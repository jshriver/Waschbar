//! Shared transposition table (3-entry buckets, generation-based replacement).
//!
//! Lazy SMP threads read and write entries without locks, as in the C++ engine. Every field
//! is a relaxed atomic, which compiles to plain loads and stores: a torn entry is possible
//! and harmless (the key check and move validation catch it), but there is no data race.

use crate::moves::Move;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicI16, AtomicU8, AtomicU16, Ordering::Relaxed};

pub const BOUND_NONE: i32 = 0;
pub const BOUND_UPPER: i32 = 1;
pub const BOUND_LOWER: i32 = 2;
pub const BOUND_EXACT: i32 = 3;

#[repr(C)]
pub struct TTEntry {
    key: AtomicU16,
    mv: AtomicU16,
    score: AtomicI16,
    eval: AtomicI16,
    depth: AtomicU8,
    gen_bound: AtomicU8,
}

impl TTEntry {
    #[inline(always)]
    pub fn mv(&self) -> Move {
        self.mv.load(Relaxed)
    }
    #[inline(always)]
    pub fn score(&self) -> i32 {
        self.score.load(Relaxed) as i32
    }
    #[inline(always)]
    pub fn eval(&self) -> i32 {
        self.eval.load(Relaxed) as i32
    }
    #[inline(always)]
    pub fn depth(&self) -> i32 {
        self.depth.load(Relaxed) as i32
    }
    #[inline(always)]
    pub fn bound(&self) -> i32 {
        (self.gen_bound.load(Relaxed) & 3) as i32
    }
    #[inline(always)]
    pub fn is_pv(&self) -> bool {
        self.gen_bound.load(Relaxed) & 4 != 0
    }
}

#[repr(C)]
pub struct TTBucket {
    e: [TTEntry; 3],
    _pad: u16,
}

const _: () = assert!(std::mem::size_of::<TTBucket>() == 32);

struct Inner {
    table: *mut TTBucket,
    bytes: usize,
    count: u64,
}

pub struct TT {
    inner: UnsafeCell<Inner>,
    generation: AtomicU8,
}

// SAFETY: `inner` is only changed by resize / ensure / clear, which the UCI layer calls only
// while no search thread runs (threads are spawned after and joined before, which orders the
// accesses). Entries themselves are atomics.
unsafe impl Sync for TT {}

pub static TT: TT = TT { inner: UnsafeCell::new(Inner { table: std::ptr::null_mut(), bytes: 0, count: 0 }), generation: AtomicU8::new(0) };

const HUGE_PAGE: usize = 2 << 20;

#[cfg(target_os = "linux")]
unsafe extern "C" {
    fn madvise(addr: *mut std::ffi::c_void, len: usize, advice: i32) -> i32;
}

impl TT {
    #[inline(always)]
    fn inner(&self) -> &Inner {
        // SAFETY: see the Sync impl.
        unsafe { &*self.inner.get() }
    }
    #[allow(clippy::mut_from_ref)]
    fn inner_mut(&self) -> &mut Inner {
        // SAFETY: see the Sync impl; callers hold no other reference into `inner`.
        unsafe { &mut *self.inner.get() }
    }

    fn free(&self) {
        let i = self.inner_mut();
        if !i.table.is_null() {
            // SAFETY: allocated in `ensure` with this exact layout.
            unsafe {
                std::alloc::dealloc(
                    i.table as *mut u8,
                    std::alloc::Layout::from_size_align_unchecked(i.bytes, HUGE_PAGE),
                )
            };
            i.table = std::ptr::null_mut();
        }
    }

    /// Sets the size in MB; memory is (re)allocated lazily by `ensure`. Not during a search.
    pub fn resize(&self, mb: usize) {
        let n = ((mb * 1024 * 1024) / std::mem::size_of::<TTBucket>()).max(1024) as u64;
        if n == self.inner().count && !self.inner().table.is_null() {
            return;
        }
        self.free();
        self.inner_mut().count = n;
    }

    /// Allocates (and clears) the table if needed. Not during a search.
    pub fn ensure(&self) {
        if !self.inner().table.is_null() {
            return;
        }
        let count = self.inner().count as usize;
        let bytes = (count * std::mem::size_of::<TTBucket>()).div_ceil(HUGE_PAGE) * HUGE_PAGE;
        let layout = std::alloc::Layout::from_size_align(bytes, HUGE_PAGE).expect("TT layout");
        // SAFETY: non-zero size; the table is filled by `clear` before use.
        let p = unsafe { std::alloc::alloc(layout) };
        if p.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        #[cfg(target_os = "linux")]
        // SAFETY: p..p+bytes is our allocation; MADV_HUGEPAGE (14) is only a hint.
        unsafe {
            madvise(p as *mut _, bytes, 14);
        }
        let i = self.inner_mut();
        i.table = p as *mut TTBucket;
        i.bytes = bytes;
        self.clear();
    }

    /// Zeroes every entry. Not during a search.
    pub fn clear(&self) {
        let i = self.inner();
        if i.table.is_null() {
            return;
        }
        // SAFETY: the table holds `count` buckets; all-zero is a valid (empty) bucket.
        unsafe { std::ptr::write_bytes(i.table, 0, i.count as usize) };
        self.generation.store(0, Relaxed);
    }

    pub fn new_search(&self) {
        let mut g = self.generation.load(Relaxed).wrapping_add(8);
        if g == 0 {
            g = 8;
        }
        self.generation.store(g, Relaxed);
    }

    #[inline(always)]
    pub fn bucket(&self, key: u64) -> &TTBucket {
        let i = self.inner();
        let idx = ((key as u128 * i.count as u128) >> 64) as usize;
        // SAFETY: idx < count, and the table is allocated before any search starts.
        unsafe { &*i.table.add(idx) }
    }

    #[inline(always)]
    pub fn prefetch(&self, key: u64) {
        #[cfg(target_arch = "x86_64")]
        // SAFETY: prefetch never faults.
        unsafe {
            use std::arch::x86_64::{_MM_HINT_T0, _mm_prefetch};
            _mm_prefetch::<_MM_HINT_T0>(self.bucket(key) as *const _ as *const i8);
        }
        #[cfg(not(target_arch = "x86_64"))]
        let _ = key;
    }

    /// The entry for `key` and whether it was found; on a miss, the entry to replace.
    #[inline]
    pub fn probe(&self, key: u64) -> (&TTEntry, bool) {
        let b = self.bucket(key);
        let k16 = key as u16;
        for e in &b.e {
            if e.key.load(Relaxed) == k16 && e.gen_bound.load(Relaxed) != 0 {
                return (e, true);
            }
        }
        let generation = self.generation.load(Relaxed);
        let worth = |e: &TTEntry| {
            e.depth.load(Relaxed) as i32 - ((generation.wrapping_sub(e.gen_bound.load(Relaxed) & 0xF8)) >> 2) as i32
        };
        let mut rep = &b.e[0];
        for e in &b.e[1..] {
            if worth(e) < worth(rep) {
                rep = e;
            }
        }
        (rep, false)
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn store(&self, e: &TTEntry, key: u64, m: Move, score: i32, eval: i32, depth: i32, bound: i32, pv: bool) {
        let k16 = key as u16;
        let generation = self.generation.load(Relaxed);
        let ekey = e.key.load(Relaxed);
        if m != 0 || ekey != k16 {
            e.mv.store(m, Relaxed);
        }
        if bound == BOUND_EXACT
            || ekey != k16
            || depth + 4 > e.depth.load(Relaxed) as i32
            || (e.gen_bound.load(Relaxed) & 0xF8) != generation
        {
            e.key.store(k16, Relaxed);
            e.score.store(score as i16, Relaxed);
            e.eval.store(eval as i16, Relaxed);
            e.depth.store(depth.max(0) as u8, Relaxed);
            e.gen_bound.store(generation | if pv { 4 } else { 0 } | bound as u8, Relaxed);
        }
    }

    /// Permille of the first 1000 buckets' entries written in this search.
    pub fn hashfull(&self) -> i32 {
        let i = self.inner();
        if i.table.is_null() {
            return 0;
        }
        let generation = self.generation.load(Relaxed);
        let mut n = 0;
        for b in 0..1000 {
            // SAFETY: count >= 1024.
            let bucket = unsafe { &*i.table.add(b) };
            for e in &bucket.e {
                let gb = e.gen_bound.load(Relaxed);
                if gb != 0 && (gb & 0xF8) == generation {
                    n += 1;
                }
            }
        }
        n / 3
    }
}
