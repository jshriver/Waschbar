//! nnue: incremental NNUE evaluation (ported from erebus-nnue-midlevel's `nnue.rs`).
//!
//! ============================== ARCHITECTURE ==============================
//! A king-bucketed perspective-pair feature transformer (the accumulators),
//! then three small layers whose weights are chosen by an output bucket
//! (material count):
//!
//!     (768 x INPUT_BUCKETS -> HL) x 2  ->  L2  ->  L3  ->  1
//!     = (768x8 -> 1024) x 2 -> 16 -> 32 -> 1, SCReLU after every hidden layer
//!
//!   * 768 inputs      = 2 colors * 6 piece types * 64 squares, repeated once
//!                       per king bucket. Each perspective reads the block for
//!                       ITS OWN king's bucket (`KING_BUCKETS`, indexed by the
//!                       king square as seen from that perspective).
//!   * HL              = hidden / accumulator width per perspective
//!   * output bucket   = (pieces on board - 2) / 4, 0..=7 -- bullet's
//!                       `MaterialCount<8>`. Picks which of the 8 weight sets
//!                       of every layer after the accumulators is used.
//!   * SCReLU          = clamp-then-square. On the accumulators it is done in
//!                       integers (clamp to [0, QA]); in the float layers it
//!                       is clamp(x, 0, 1)^2.
//!   * QA = 255, QB = 64, SCALE = 400
//!
//! Evaluation cost is dominated by the 2*HL -> L2 layer (2048 x 16 integer
//! multiply-adds), done in i16/i32 SIMD. L2 -> L3 -> 1 is ~550 f32
//! multiply-adds.
//!
//! Each search thread owns an `Evaluator` holding two fixed-perspective
//! accumulators (White's and Black's), updated incrementally -- and lazily:
//!
//!   - Accumulators live on a stack (`AccStack`), one pair per make_move
//!     depth. `Position::make_move` records what the move changed
//!     (`Position::feats`) and the search pushes it (`Evaluator::push`); the
//!     arithmetic happens when an eval is requested, from the nearest
//!     computed ancestor. Positions that are never evaluated (TT cutoffs,
//!     TT-eval and eval-cache hits, in-check nodes, pruned moves) cost no
//!     accumulator work.
//!   - A move changes at most 4 (piece, square) features, so a child is
//!     written in one pass: parent + added rows - removed rows.
//!   - A king move that changes its own perspective's king bucket changes
//!     every feature index of that perspective, so that ONE accumulator is
//!     rebuilt instead, from a cache of accumulators by king bucket
//!     (`BucketCache`); the other perspective is still toggled.
//!   - Unmaking a move just returns to the parent entry (`Evaluator::pop`).
//!   - Recent evals are kept in an `EvalCache` keyed by the position's
//!     Zobrist key, so a position reached again skips the forward pass.
//!
//! Pieces use the engine encoding `color * 6 + piece_type` (pawn 0 .. king 5,
//! white 0, black 1), which is also the net's input order.
//!
//! ============================== NET SOURCE ==============================
//! The network (`nets/net.nnue`) is embedded into the executable via
//! `include_bytes!` and parsed once at startup (`init`). It is a `bullet`
//! export for exactly this architecture -- see `parse_net` for the byte
//! layout.

use std::cell::Cell;
use std::sync::OnceLock;

use crate::board::*;

// ================= Architecture constants (must match the net) =================

/// Hidden layer / accumulator width per perspective.
pub const HL: usize = 1024;

/// Number of raw board features: 2 colors * 6 pieces * 64 squares.
const INPUT_SIZE: usize = 768;

/// King input buckets. MUST equal the trainer's bucket layout.
const INPUT_BUCKETS: usize = 8;

/// King bucket for each square of a perspective's OWN king, from that
/// perspective's side of the board (a1 = 0 .. h8 = 63; Black's king square is
/// flipped with `^ 56` first). MUST equal `KING_BUCKET_LAYOUT` in the trainer
/// (bullet `ChessBuckets`, which reads `our_ksq` / `opp_ksq` the same way).
#[rustfmt::skip]
const KING_BUCKETS: [u8; 64] = [
    0, 1, 2, 3, 3, 2, 1, 0,
    4, 4, 5, 5, 5, 5, 4, 4,
    6, 6, 6, 6, 6, 6, 6, 6,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
    7, 7, 7, 7, 7, 7, 7, 7,
];

/// Output buckets. MUST equal the trainer's `MaterialCount<OUTPUT_BUCKETS>`.
const OUTPUT_BUCKETS: usize = 8;

/// Width of the first layer after the accumulators (2*HL -> L2).
const L2: usize = 16;
/// Width of the second layer (L2 -> L3 -> 1).
const L3: usize = 32;

const QA: i64 = 255;
const QB: i64 = 64;
const SCALE: i64 = 400;

/// The 2*HL -> L2 layer sums `sq * l1w`, where `sq = clamp(acc,0,QA)^2 >> 1`
/// (halved so it fits a signed i16). Multiplying by this turns the sum back
/// into the float the trainer computed (units QA*QA*QB, halved).
const L1_SCALE: f64 = 2.0 / (QA * QA * QB) as f64;

const EMBEDDED_NET: &[u8] = include_bytes!("../nets/net.nnue");

// ================= Net =================

pub struct Net {
    /// Feature-transformer weights, feature-major: `l0w[feature * HL + h]`,
    /// `feature = 768 * king_bucket + feature_index`.
    l0w: AlignedBuf<i16>,
    /// Feature-transformer biases: `l0b[h]`.
    l0b: AlignedBuf<i16>,
    /// 2*HL -> L2 weights (x QB), rearranged from the file's
    /// `l1w[bucket][o][i]` (i < HL: stm accumulator, i >= HL: ntm) for the
    /// sparse forward pass: `l1w_pairs[bucket][p][o][k] = l1w[bucket][o][2p + k]`,
    /// so one input pair's weights for all L2 neurons are 2*L2 contiguous i16.
    l1w_pairs: AlignedBuf<i16>,
    /// L2 biases, `[bucket][L2]`, float.
    l1b: AlignedBuf<f32>,
    /// L2 -> L3 weights, rearranged from the file's `[bucket][L3][L2]` to
    /// `[bucket][L2][L3]` so the forward pass updates all L3 sums per input.
    l2w_t: AlignedBuf<f32>,
    /// L3 biases, `[bucket][L3]`, float.
    l2b: AlignedBuf<f32>,
    /// L3 -> 1 weights, `[bucket][L3]`, float.
    l3w: AlignedBuf<f32>,
    /// Output bias, `[bucket]`, float.
    l3b: AlignedBuf<f32>,
}

static NET: OnceLock<Net> = OnceLock::new();

/// Parse the network embedded in the executable (idempotent). Must run before
/// any `Evaluator` is built. Fails only if the binary was built with a broken
/// net -- there is no classical eval to fall back to.
pub fn init() -> Result<(), String> {
    if NET.get().is_none() {
        let net = parse_net(EMBEDDED_NET)?;
        let _ = NET.set(net);
    }
    Ok(())
}

/// The active net. Panics if `init` has not run, which `main` rules out.
#[inline]
pub fn net() -> &'static Net {
    NET.get().expect("NNUE net not loaded")
}

// ================= Loading =================

/// A 64-byte aligned, fixed-length buffer of plain numbers. The weight rows
/// (`l0w`: 2048 bytes, `l1w_pairs`: 64 bytes per input pair) then start on a
/// cache line, so no SIMD load splits one; a plain `Vec<i16>` is only 16-byte
/// aligned for large allocations, which cost ~10% of eval speed.
pub struct AlignedBuf<T: Copy> {
    buf: Vec<Align64>,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Align64([u8; 64]);

impl<T: Copy> AlignedBuf<T> {
    fn from_vec(v: Vec<T>) -> Self {
        let bytes = std::mem::size_of_val(v.as_slice());
        let mut buf = vec![Align64([0; 64]); bytes.div_ceil(64)];
        // SAFETY: buf holds at least `bytes` bytes; T is plain data (i16 / f32).
        unsafe { std::ptr::copy_nonoverlapping(v.as_ptr() as *const u8, buf.as_mut_ptr() as *mut u8, bytes) };
        AlignedBuf { buf, len: v.len(), _t: std::marker::PhantomData }
    }
}

impl<T: Copy> std::ops::Deref for AlignedBuf<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: the buffer holds `len` initialised T values, 64-byte aligned.
        unsafe { std::slice::from_raw_parts(self.buf.as_ptr() as *const T, self.len) }
    }
}

fn read_i16_block(buf: &[u8], offset: &mut usize, n: usize) -> Vec<i16> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        out.push(i16::from_le_bytes([buf[*offset], buf[*offset + 1]]));
        *offset += 2;
    }
    out
}

fn read_f32_block(buf: &[u8], offset: &mut usize, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let b = &buf[*offset..*offset + 4];
        out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        *offset += 4;
    }
    out
}

/// Parse a `bullet` export of the midlevel net. Little-endian, in this order
/// (`ob` = output bucket), then trailing "bullet" padding, ignored:
///   i16  l0w  [INPUT_BUCKETS][768][HL]   x QA, feature-major
///   i16  l0b  [HL]                        x QA
///   i16  l1w  [ob][L2][2*HL]              x QB, stm half then ntm half
///   f32  l1b  [ob][L2]
///   f32  l2w  [ob][L3][L2]
///   f32  l2b  [ob][L3]
///   f32  l3w  [ob][L3]
///   f32  l3b  [ob]
fn parse_net(buf: &[u8]) -> Result<Net, String> {
    let n_l0w = INPUT_BUCKETS * INPUT_SIZE * HL;
    let n_l0b = HL;
    let n_l1w = OUTPUT_BUCKETS * L2 * 2 * HL;
    let n_f32 = OUTPUT_BUCKETS * (L2 + L3 * L2 + L3 + L3 + 1);
    let expected = (n_l0w + n_l0b + n_l1w) * 2 + n_f32 * 4;

    if buf.len() < expected {
        return Err(format!(
            "net data is {} bytes, expected at least {expected} for a \
             (768x{INPUT_BUCKETS} -> {HL}) x 2 -> {L2} -> {L3} -> 1 net with {OUTPUT_BUCKETS} \
             output buckets.",
            buf.len()
        ));
    }

    let mut off = 0usize;
    let l0w = read_i16_block(buf, &mut off, n_l0w);
    let l0b = read_i16_block(buf, &mut off, n_l0b);
    let l1w = read_i16_block(buf, &mut off, n_l1w);
    let l1b = read_f32_block(buf, &mut off, OUTPUT_BUCKETS * L2);
    let l2w = read_f32_block(buf, &mut off, OUTPUT_BUCKETS * L3 * L2);
    let l2b = read_f32_block(buf, &mut off, OUTPUT_BUCKETS * L3);
    let l3w = read_f32_block(buf, &mut off, OUTPUT_BUCKETS * L3);
    let l3b = read_f32_block(buf, &mut off, OUTPUT_BUCKETS);

    // `l1_forward`'s i32 lanes are sized for |l1w| <= 128 (see FLUSH_PAIRS).
    // The trainer's weight clip (+-1.98, x QB = +-127) keeps a normal net
    // inside this; reject anything hotter with an actionable message rather
    // than evaluating garbage.
    let max_l1w = l1w.iter().map(|&x| (x as i32).abs()).max().unwrap_or(0);
    if QA as i32 * max_l1w > i16::MAX as i32 {
        return Err(format!(
            "l1w |max| = {max_l1w}: QA({QA}) * {max_l1w} = {} exceeds i16 range; the 2*HL -> L2 \
             layer's i32 sums are sized for |l1w| <= {}. Retrain with tighter l1w clipping.",
            QA as i32 * max_l1w,
            i16::MAX as i32 / QA as i32,
        ));
    }

    // A NaN or infinity in the float layers would make every eval garbage.
    let floats = [&l1b, &l2w, &l2b, &l3w, &l3b];
    if floats.iter().any(|v| v.iter().any(|x| !x.is_finite())) {
        return Err("the float layers (l1b, l2w, l2b, l3w, l3b) contain NaN or infinity -- \
                    the net is corrupt or diverged in training"
            .to_string());
    }

    // The size check only proves the file is big enough. A right-sized file
    // whose feature-transformer block is all zeros passes it and then
    // evaluates every position to a single constant (the output bias) --
    // the engine goes blind while still "running". Catch that here.
    let nz = l0w.iter().filter(|&&x| x != 0).count();
    let nz_frac = nz as f64 / l0w.len().max(1) as f64;
    if nz_frac < 0.01 {
        return Err(format!(
            "feature-transformer weights are {:.3}% non-zero ({nz}/{} values) -- this is not a \
             trained net. Every position would evaluate to the same constant.",
            nz_frac * 100.0,
            l0w.len(),
        ));
    }

    let mut l1w_pairs = vec![0i16; l1w.len()];
    for b in 0..OUTPUT_BUCKETS {
        for o in 0..L2 {
            for i in 0..2 * HL {
                let (p, k) = (i / 2, i % 2);
                l1w_pairs[((b * HL + p) * L2 + o) * 2 + k] = l1w[(b * L2 + o) * 2 * HL + i];
            }
        }
    }

    let mut l2w_t = vec![0f32; l2w.len()];
    for b in 0..OUTPUT_BUCKETS {
        for j in 0..L3 {
            for o in 0..L2 {
                l2w_t[(b * L2 + o) * L3 + j] = l2w[(b * L3 + j) * L2 + o];
            }
        }
    }

    Ok(Net {
        l0w: AlignedBuf::from_vec(l0w),
        l0b: AlignedBuf::from_vec(l0b),
        l1w_pairs: AlignedBuf::from_vec(l1w_pairs),
        l1b: AlignedBuf::from_vec(l1b),
        l2w_t: AlignedBuf::from_vec(l2w_t),
        l2b: AlignedBuf::from_vec(l2b),
        l3w: AlignedBuf::from_vec(l3w),
        l3b: AlignedBuf::from_vec(l3b),
    })
}

// ================= Feature indexing =================

/// King bucket of `perspective`, given that side's own king square (board
/// coordinates). Black's square is flipped so both sides index
/// `KING_BUCKETS` from their own back rank.
#[inline(always)]
pub fn king_bucket(perspective: usize, king_sq: usize) -> u8 {
    let rel = if perspective == WHITE { king_sq } else { king_sq ^ 56 };
    KING_BUCKETS[rel]
}

/// Output bucket for a position with `piece_count` pieces (kings included),
/// matching bullet's `MaterialCount<8>`: `(count - 2) / ceil(32 / 8)`.
#[inline(always)]
pub fn output_bucket(piece_count: u32) -> usize {
    const DIV: u32 = 32u32.div_ceil(OUTPUT_BUCKETS as u32);
    ((piece_count.saturating_sub(2) / DIV) as usize).min(OUTPUT_BUCKETS - 1)
}

/// L0 feature index for one engine piece (`color * 6 + type`) on `sq`, as seen
/// from `perspective`, whose own king is in king bucket `bucket`.
///
///   White perspective: index = 768*bucket + 384*color       + 64*type + sq
///   Black perspective: index = 768*bucket + 384*(color ^ 1) + 64*type + (sq ^ 56)
#[inline(always)]
fn feature_index(perspective: usize, bucket: u8, piece: usize, sq: usize) -> usize {
    let (rel_color, rel_sq) = if perspective == WHITE {
        (piece_color(piece), sq)
    } else {
        (piece_color(piece) ^ 1, sq ^ 56)
    };
    INPUT_SIZE * bucket as usize + 384 * rel_color + 64 * piece_type(piece) + rel_sq
}

#[inline(always)]
fn weight_row(net: &Net, feature: usize) -> &[i16] {
    let base = feature * HL;
    &net.l0w[base..base + HL]
}

// ================= Accumulators =================

/// One perspective's accumulator: feature-transformer bias plus the sum of
/// the weight rows of every active feature. Kept in **i16** -- half the
/// memory traffic of i32 per toggle and per eval load, and it lets the
/// forward pass fuse SCReLU square + output-weight multiply into a single
/// `madd_epi16`. A bullet-quantised net keeps every hidden unit well inside
/// i16 range (32 pieces * max|l0w|); `refresh_one` debug-asserts this.
/// Cache-line aligned, like the weight rows, so SIMD loads never split.
#[repr(C, align(64))]
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Accumulator {
    pub v: [i16; HL],
    /// King bucket this accumulator was built for (its own perspective's
    /// king). Set by `refresh_one`; every toggle indexes weights with it.
    pub bucket: u8,
}

impl Accumulator {
    pub fn zeroed() -> Self {
        Accumulator { v: [0; HL], bucket: 0 }
    }
}

/// Both perspectives, indexed by color (NOT by side-to-move).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct DualAccumulator {
    pub acc: [Accumulator; 2],
}

impl DualAccumulator {
    pub fn zeroed() -> Self {
        DualAccumulator { acc: [Accumulator::zeroed(), Accumulator::zeroed()] }
    }
}

/// One input-feature change: `(piece, square, added)`, piece in engine
/// encoding. A move makes at most 4 (castling moves king + rook).
pub type FeatDelta = (u8, u8, bool);

/// What `push` recorded for a stack entry: everything needed to build its
/// accumulators from the entry below, later.
#[derive(Clone)]
struct Pending {
    feats: [FeatDelta; 4],
    n: u8,
    /// Bit per perspective whose king changed bucket with this move: that
    /// accumulator is rebuilt from `pieces` instead of toggled.
    refresh: u8,
    /// King bucket of each perspective at this entry -- known at make time,
    /// without the accumulators.
    buckets: [u8; 2],
    /// The board's `pieces[color][piece]` at this entry. Only set (and only
    /// read) when `refresh != 0`.
    pieces: [[u64; 6]; 2],
}

impl Pending {
    fn empty() -> Self {
        Pending { feats: [(0, 0, false); 4], n: 0, refresh: 0, buckets: [0; 2], pieces: [[0; 6]; 2] }
    }
}

/// Lazily-updated accumulator pairs by make_move depth: entry `top` is the
/// current position, entry `top - 1` the position before the last move, and
/// so on. A move is only recorded (`push`); unmaking steps back (`pop`). The
/// accumulators are brought up to date on demand (`current`) from the
/// nearest computed ancestor. Entries above `top` are kept (not zeroed) and
/// reused.
#[derive(Clone)]
pub struct AccStack {
    accs: Vec<DualAccumulator>,
    /// `meta[i]`: how entry `i` follows from entry `i - 1`.
    meta: Vec<Pending>,
    top: usize,
    /// Entries `0..=clean` hold computed accumulators; `clean < i <= top`
    /// are pending.
    clean: usize,
}

impl AccStack {
    pub fn new() -> Self {
        AccStack { accs: vec![DualAccumulator::zeroed()], meta: vec![Pending::empty()], top: 0, clean: 0 }
    }

    /// Drop every entry and rebuild the base from scratch for `board`.
    pub fn reset(&mut self, net: &Net, board: &[u8; 64]) {
        self.top = 0;
        self.clean = 0;
        refresh_all(net, board, &mut self.accs[0]);
        self.meta[0].buckets = [self.accs[0].acc[WHITE].bucket, self.accs[0].acc[BLACK].bucket];
    }

    /// King bucket of `perspective` in the current position.
    #[inline(always)]
    pub fn bucket(&self, perspective: usize) -> u8 {
        self.meta[self.top].buckets[perspective]
    }

    /// Record a move as a new pending entry: its feature diff, which
    /// perspectives changed king bucket (`refresh`), the king buckets after
    /// the move, and the board's piece bitboards after the move.
    #[inline]
    pub fn push(&mut self, feats: &[FeatDelta], refresh: u8, buckets: [u8; 2], pieces: &[[u64; 6]; 2]) {
        if self.top + 1 == self.accs.len() {
            self.accs.push(DualAccumulator::zeroed());
            self.meta.push(Pending::empty());
        }
        self.top += 1;
        let m = &mut self.meta[self.top];
        m.feats[..feats.len()].copy_from_slice(feats);
        m.n = feats.len() as u8;
        m.refresh = refresh;
        m.buckets = buckets;
        if refresh != 0 {
            m.pieces = *pieces;
        }
    }

    /// Step back one entry. `false` if already at the base (the caller must
    /// then rebuild), which only happens if the stack was reset between a
    /// make and its unmake.
    #[inline]
    pub fn pop(&mut self) -> bool {
        if self.top == 0 {
            return false;
        }
        self.top -= 1;
        self.clean = self.clean.min(self.top);
        true
    }

    /// The current position's accumulators, building any pending entries.
    #[inline]
    pub fn current(&mut self, net: &Net, cache: &mut BucketCache) -> &DualAccumulator {
        while self.clean < self.top {
            let i = self.clean + 1;
            let (done, pending) = self.accs.split_at_mut(i);
            let (parent, child) = (&done[i - 1], &mut pending[0]);
            let m = &self.meta[i];
            for persp in [WHITE, BLACK] {
                let acc = &mut child.acc[persp];
                if m.refresh & (1 << persp) != 0 {
                    refresh_cached(net, cache, &m.pieces, persp, acc);
                } else {
                    update_from_parent(net, persp, &parent.acc[persp], acc, &m.feats[..m.n as usize]);
                }
            }
            self.clean = i;
        }
        &self.accs[self.top]
    }
}

/// log2 of the eval cache's entry count (8 bytes each: 2^16 = 512 KB).
const EVAL_CACHE_BITS: u32 = 16;

/// Recent evals by position: entry `key & mask` holds the upper 32 bits of
/// the Zobrist key and the eval. The search evaluates many positions more
/// than once (transpositions, each iterative-deepening pass), and a hit skips
/// the whole forward pass. Each search thread has its own `Evaluator`, so it
/// is never shared.
#[derive(Clone)]
pub struct EvalCache {
    slots: Box<[Cell<u64>]>,
}

impl EvalCache {
    pub fn new() -> Self {
        EvalCache { slots: (0..1usize << EVAL_CACHE_BITS).map(|_| Cell::new(0)).collect() }
    }

    #[inline(always)]
    fn slot(&self, hash: u64) -> &Cell<u64> {
        &self.slots[hash as usize & ((1usize << EVAL_CACHE_BITS) - 1)]
    }

    #[inline(always)]
    fn probe(&self, hash: u64) -> Option<i32> {
        let e = self.slot(hash).get();
        (e >> 32 == hash >> 32 && e != 0).then_some(e as u32 as i32)
    }

    #[inline(always)]
    fn store(&self, hash: u64, eval: i32) {
        self.slot(hash).set((hash & 0xFFFF_FFFF_0000_0000) | eval as u32 as u64);
    }
}

// ================= SIMD kernels =================
//
// Hot operations, all over `HL`-wide i16 vectors:
//   * acc_add / acc_sub : accumulator +/- an i16 weight row  (per feature toggle)
//   * acc_set           : accumulator := i16 bias            (per refresh)
//   * l1_forward        : sq = clamp(acc, 0, QA)^2 >> 1, then
//                         s[o] = Σ_i sq[i] * w[o][i]         (per eval)
//
// The squares are halved (max 255^2 >> 1 = 32512) so they fit a signed i16
// and one `madd_epi16` per 8 outputs does the multiply-add; dropping the low
// bit changes the layer's output by ~1e-4 relative, far below what matters.
// l1_forward exploits that most clamped accumulator values are 0 (~88% of
// inputs, ~77% of input pairs with the mini's weights). It first lists the
// input pairs (sq[2p], sq[2p+1]) with a nonzero value -- branch-free, via
// NZ_LUT -- then adds each listed pair to all L2 = 16 outputs at once, from
// `l1w_pairs`. Each i32 lane gains at most 2*32512*128 per pair, so the lanes
// are widened into i64 totals every FLUSH_PAIRS pairs (the const assert keeps
// that inside i32). The result equals the plain i64 sum exactly
// (`l1_forward_matches_dense`, `forward_pass_matches_scalar`). The AVX2
// version squares all inputs in the nonzero scan, 16 at a time, so the
// per-pair loop is just a broadcast and two madds: on Haswell the multiply,
// shift and madds all compete for port 0, and squaring per pair cost ~18%.
//
// x86_64: AVX2 when the target is built for it (`-C target-cpu=native` sets
// this), else a runtime `is_x86_feature_detected!` check. aarch64: NEON,
// baseline for the target. Everything else: scalar.

const QA_I16: i16 = QA as i16;

/// `NZ_LUT[m]` lists the set bits of the 8-bit mask `m`, lowest first: turns
/// a block's nonzero mask into pair offsets without a branch per pair.
#[cfg(target_arch = "x86_64")]
static NZ_LUT: [[u16; 8]; 256] = {
    let mut t = [[0u16; 8]; 256];
    let mut m = 0;
    while m < 256 {
        let (mut k, mut b) = (0, 0);
        while b < 8 {
            if (m >> b) & 1 == 1 {
                t[m][k] = b as u16;
                k += 1;
            }
            b += 1;
        }
        m += 1;
    }
    t
};

/// Input pairs accumulated in i32 before widening to i64.
const FLUSH_PAIRS: usize = 128;
const _: () = assert!(
    FLUSH_PAIRS as i64 * 2 * ((QA * QA) >> 1) * 128 <= i32::MAX as i64,
    "l1_forward's i32 lanes could overflow",
);
const _: () = assert!(L2 == 16, "l1_forward_avx2 is written for 16 outputs");
const _: () = assert!(L3.is_multiple_of(8), "l2_l3_avx2 works in 8-wide f32 blocks");
const _: () = assert!(HL.is_multiple_of(64), "l1_forward_avx2 scans 64 values per step");

#[inline]
fn acc_add(acc: &mut [i16], row: &[i16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return acc_addsub_avx2(acc, row, false) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return acc_addsub_avx2(acc, row, false) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return acc_addsub_neon(acc, row, false) }
    #[allow(unreachable_code)]
    acc_addsub_scalar(acc, row, false)
}

#[inline]
fn acc_sub(acc: &mut [i16], row: &[i16]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return acc_addsub_avx2(acc, row, true) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return acc_addsub_avx2(acc, row, true) }
    }
    #[cfg(target_arch = "aarch64")]
    unsafe { return acc_addsub_neon(acc, row, true) }
    #[allow(unreachable_code)]
    acc_addsub_scalar(acc, row, true)
}

#[inline]
fn acc_set(acc: &mut [i16], bias: &[i16]) {
    let n = acc.len().min(bias.len());
    acc[..n].copy_from_slice(&bias[..n]);
}

/// `out = clamp(acc, 0, QA)^2 >> 1` (0..=32512). A plain loop the compiler
/// vectorises on every target.
#[inline]
fn screlu_half(acc: &[i16], out: &mut [i16]) {
    for (o, &a) in out.iter_mut().zip(acc) {
        let c = a.clamp(0, QA_I16) as u16;
        *o = ((c * c) >> 1) as i16;
    }
}

/// The 2*HL -> L2 layer straight from the accumulators: `out[o] =
/// Σ_i sq[i] * w[o][i]`, where `sq` is `screlu_half` of `us` then `them`
/// and `w` is one output bucket's `l1w_pairs`.
#[inline]
fn l1_forward(us: &[i16; HL], them: &[i16; HL], w: &[i16], out: &mut [i64; L2]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return l1_forward_avx2(us, them, w, out) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return l1_forward_avx2(us, them, w, out) }
    }
    #[allow(unreachable_code)]
    {
        let mut c = [0i16; 2 * HL];
        screlu_half(us, &mut c[..HL]);
        screlu_half(them, &mut c[HL..]);
        l1_sparse_scalar(&c, w, out)
    }
}

// ---- scalar reference ----

fn l1_sparse_scalar(c: &[i16; 2 * HL], w: &[i16], out: &mut [i64; L2]) {
    *out = [0; L2];
    for p in 0..HL {
        let (a, b) = (c[2 * p], c[2 * p + 1]);
        if a == 0 && b == 0 {
            continue;
        }
        let row = &w[p * 2 * L2..(p + 1) * 2 * L2];
        for o in 0..L2 {
            out[o] += a as i64 * row[2 * o] as i64 + b as i64 * row[2 * o + 1] as i64;
        }
    }
}

fn acc_addsub_scalar(acc: &mut [i16], row: &[i16], sub: bool) {
    let n = acc.len().min(row.len());
    if sub {
        for h in 0..n { acc[h] = acc[h].wrapping_sub(row[h]); }
    } else {
        for h in 0..n { acc[h] = acc[h].wrapping_add(row[h]); }
    }
}

// ---- x86_64 AVX2 ----

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_addsub_avx2(acc: &mut [i16], row: &[i16], sub: bool) {
    use std::arch::x86_64::*;
    let n = acc.len().min(row.len());
    let mut h = 0;
    while h + 16 <= n {
        let a = _mm256_loadu_si256(acc.as_ptr().add(h) as *const __m256i);
        let r = _mm256_loadu_si256(row.as_ptr().add(h) as *const __m256i);
        let s = if sub { _mm256_sub_epi16(a, r) } else { _mm256_add_epi16(a, r) };
        _mm256_storeu_si256(acc.as_mut_ptr().add(h) as *mut __m256i, s);
        h += 16;
    }
    if sub {
        while h < n { acc[h] = acc[h].wrapping_sub(row[h]); h += 1; }
    } else {
        while h < n { acc[h] = acc[h].wrapping_add(row[h]); h += 1; }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l1_forward_avx2(us: &[i16; HL], them: &[i16; HL], w: &[i16], out: &mut [i64; L2]) {
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;
    debug_assert!(w.len() >= HL * 2 * L2);
    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA_I16);

    #[repr(C, align(64))]
    struct Sq([i16; 2 * HL]);
    // Both buffers are fully written by pass 1 before pass 2 reads them.
    let mut sq = MaybeUninit::<Sq>::uninit();
    let sqp = sq.as_mut_ptr() as *mut i16;
    let mut list = MaybeUninit::<[u16; HL + 8]>::uninit();
    let lp = list.as_mut_ptr() as *mut u16;

    // Pass 1: SCReLU all 2*HL inputs 16 at a time (clamp, square, halve),
    // store them, and list the nonzero input pairs branch-free. Squaring
    // here instead of per pair in pass 2 takes the multiply and shift off
    // port 0, which pass 2's madds saturate on Haswell.
    let mut count = 0usize;
    for (half, src) in [us, them].into_iter().enumerate() {
        let sp = src.as_ptr();
        for step in 0..HL / 64 {
            let mut m = [0usize; 4];
            for (b, mb) in m.iter_mut().enumerate() {
                let i = step * 64 + b * 16;
                let c = _mm256_min_epi16(_mm256_max_epi16(_mm256_loadu_si256(sp.add(i) as *const __m256i), zero), qa);
                // c^2 <= 65025 fits u16; the logical shift halves it exactly.
                let v = _mm256_srli_epi16::<1>(_mm256_mullo_epi16(c, c));
                _mm256_store_si256(sqp.add(half * HL + i) as *mut __m256i, v);
                // a pair (one i32 lane) is nonzero iff either square is
                *mb = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpgt_epi32(v, zero))) as usize;
            }
            for (b, &mb) in m.iter().enumerate() {
                let idx = _mm_add_epi16(
                    _mm_loadu_si128(NZ_LUT[mb].as_ptr() as *const __m128i),
                    _mm_set1_epi16(((half * HL + step * 64 + b * 16) / 2) as i16),
                );
                _mm_storeu_si128(lp.add(count) as *mut __m128i, idx);
                count += mb.count_ones() as usize;
            }
        }
    }

    // Pass 2: add each listed pair to all 16 outputs; widen the i32 lanes
    // to i64 totals every FLUSH_PAIRS pairs.
    let pairs = sqp as *const i32;
    let wp = w.as_ptr();
    let mut t = [zero; 4];
    let (mut a0, mut a1) = (zero, zero);
    let mut k = 0;
    while k < count {
        let end = (k + FLUSH_PAIRS).min(count);
        while k < end {
            let p = *lp.add(k) as usize;
            let cp = _mm256_set1_epi32(*pairs.add(p));
            let row = wp.add(p * 2 * L2) as *const __m256i;
            a0 = _mm256_add_epi32(a0, _mm256_madd_epi16(cp, _mm256_loadu_si256(row)));
            a1 = _mm256_add_epi32(a1, _mm256_madd_epi16(cp, _mm256_loadu_si256(row.add(1))));
            k += 1;
        }
        t[0] = _mm256_add_epi64(t[0], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a0)));
        t[1] = _mm256_add_epi64(t[1], _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(a0)));
        t[2] = _mm256_add_epi64(t[2], _mm256_cvtepi32_epi64(_mm256_castsi256_si128(a1)));
        t[3] = _mm256_add_epi64(t[3], _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(a1)));
        a0 = zero;
        a1 = zero;
    }
    for (k, v) in t.iter().enumerate() {
        _mm256_storeu_si256(out.as_mut_ptr().add(k * 4) as *mut __m256i, *v);
    }
}

// ---- aarch64 NEON ----

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn acc_addsub_neon(acc: &mut [i16], row: &[i16], sub: bool) {
    use std::arch::aarch64::*;
    let n = acc.len().min(row.len());
    let mut h = 0;
    while h + 8 <= n {
        let a = vld1q_s16(acc.as_ptr().add(h));
        let r = vld1q_s16(row.as_ptr().add(h));
        let s = if sub { vsubq_s16(a, r) } else { vaddq_s16(a, r) };
        vst1q_s16(acc.as_mut_ptr().add(h), s);
        h += 8;
    }
    if sub {
        while h < n { acc[h] = acc[h].wrapping_sub(row[h]); h += 1; }
    } else {
        while h < n { acc[h] = acc[h].wrapping_add(row[h]); h += 1; }
    }
}

// ================= Accumulator maintenance =================

/// Full from-scratch build of one perspective's accumulator from the board,
/// including its king bucket.
pub fn refresh_one(net: &Net, board: &[u8; 64], perspective: usize, acc: &mut Accumulator) {
    let king = make_piece(perspective, KING) as u8;
    let king_sq = board.iter().position(|&p| p == king).unwrap_or(0);
    let bucket = king_bucket(perspective, king_sq);
    acc.bucket = bucket;

    acc_set(&mut acc.v, &net.l0b);
    for (sq, &p) in board.iter().enumerate() {
        if p as usize == NO_PIECE {
            continue;
        }
        let f = feature_index(perspective, bucket, p as usize, sq);
        acc_add(&mut acc.v, weight_row(net, f));
    }

    // DEBUG-ONLY: recompute in full i32 precision and confirm no hidden unit
    // came near the i16 boundary. A net that trips this needs retraining;
    // the incremental-vs-refresh assert in `Evaluator::evaluate` would NOT
    // catch it, since both wrap identically.
    #[cfg(debug_assertions)]
    {
        let mut wide = [0i32; HL];
        for h in 0..HL {
            wide[h] = net.l0b[h] as i32;
        }
        for (sq, &p) in board.iter().enumerate() {
            if p as usize != NO_PIECE {
                let row = weight_row(net, feature_index(perspective, bucket, p as usize, sq));
                for h in 0..HL {
                    wide[h] += row[h] as i32;
                }
            }
        }
        for h in 0..HL {
            debug_assert!(
                wide[h] > i16::MIN as i32 + 256 && wide[h] < i16::MAX as i32 - 256,
                "NNUE hidden unit {h} = {} is within 256 of the i16 boundary -- this net is too \
                 hot for i16 accumulators",
                wide[h],
            );
            debug_assert_eq!(wide[h] as i16, acc.v[h], "i16 accumulator diverged at unit {h}");
        }
    }
}

/// Accumulator cache for king-bucket changes (a "Finny table"): for each
/// (perspective, king bucket), the accumulator last built for that bucket and
/// the piece bitboards it was built from. Moving a king into a bucket then
/// only adds/removes the pieces that differ from that snapshot -- usually a
/// handful -- instead of rebuilding from all ~30 pieces.
#[derive(Clone)]
pub struct BucketCache {
    entries: Box<[CacheEntry; 2 * INPUT_BUCKETS]>,
}

#[repr(C, align(64))]
#[derive(Clone)]
struct CacheEntry {
    acc: [i16; HL],
    /// pieces[color][piece] the accumulator reflects
    pieces: [[u64; 6]; 2],
    valid: bool,
}

impl BucketCache {
    pub fn new() -> Self {
        let empty = CacheEntry { acc: [0; HL], pieces: [[0; 6]; 2], valid: false };
        BucketCache { entries: Box::new(std::array::from_fn(|_| empty.clone())) }
    }

    pub fn clear(&mut self) {
        for e in self.entries.iter_mut() {
            e.valid = false;
        }
    }
}

/// Rebuild one perspective's accumulator (after its king changed bucket) via
/// the cache: bring the cached entry for the new bucket up to date with
/// `pieces` (the board's `pieces[color][piece]` bitboards), then copy it out.
pub fn refresh_cached(
    net: &Net,
    cache: &mut BucketCache,
    pieces: &[[u64; 6]; 2],
    perspective: usize,
    acc: &mut Accumulator,
) {
    let king_sq = pieces[perspective][KING].trailing_zeros() as usize;
    let bucket = king_bucket(perspective, king_sq);
    let e = &mut cache.entries[perspective * INPUT_BUCKETS + bucket as usize];
    if !e.valid {
        e.acc.copy_from_slice(&net.l0b);
        e.pieces = [[0; 6]; 2];
        e.valid = true;
    }
    for color in 0..2 {
        for pt in 0..6 {
            let piece = make_piece(color, pt);
            let now = pieces[color][pt];
            let old = e.pieces[color][pt];
            let mut added = now & !old;
            while added != 0 {
                let sq = added.trailing_zeros() as usize;
                added &= added - 1;
                acc_add(&mut e.acc, weight_row(net, feature_index(perspective, bucket, piece, sq)));
            }
            let mut removed = old & !now;
            while removed != 0 {
                let sq = removed.trailing_zeros() as usize;
                removed &= removed - 1;
                acc_sub(&mut e.acc, weight_row(net, feature_index(perspective, bucket, piece, sq)));
            }
            e.pieces[color][pt] = now;
        }
    }
    acc.v.copy_from_slice(&e.acc);
    acc.bucket = bucket;
}

/// Write one perspective's child accumulator from its parent after a move:
/// `child = parent + Σ added rows - Σ removed rows`, in a single pass over
/// the accumulator (one read of the parent, one write of the child, instead
/// of a copy plus a read-modify-write per feature). `feats` is the move's
/// `(piece, square, added)` list, at most 2 added and 2 removed.
pub fn update_from_parent(
    net: &Net,
    perspective: usize,
    parent: &Accumulator,
    child: &mut Accumulator,
    feats: &[FeatDelta],
) {
    let bucket = parent.bucket;
    let mut adds: [&[i16; HL]; 2] = [&[0; HL]; 2];
    let mut subs: [&[i16; HL]; 2] = [&[0; HL]; 2];
    let (mut na, mut ns) = (0, 0);
    for &(p, sq, add) in feats {
        let row: &[i16; HL] =
            weight_row(net, feature_index(perspective, bucket, p as usize, sq as usize)).try_into().unwrap();
        if add && na < 2 {
            adds[na] = row;
            na += 1;
        } else if !add && ns < 2 {
            subs[ns] = row;
            ns += 1;
        } else {
            // more than 2 of a kind: not produced by any chess move
            unreachable!("update_from_parent: unexpected feature list {feats:?}");
        }
    }
    child.bucket = bucket;
    let (src, dst) = (&parent.v, &mut child.v);
    match (na, ns) {
        (1, 1) => fused::<1, 1>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        (1, 2) => fused::<1, 2>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        (2, 2) => fused::<2, 2>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        (2, 1) => fused::<2, 1>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        (1, 0) => fused::<1, 0>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        (0, 1) => fused::<0, 1>(dst, src, adds[0], adds[1], subs[0], subs[1]),
        _ => fused::<0, 0>(dst, src, adds[0], adds[1], subs[0], subs[1]),
    }
}

/// `dst = src + a0 [+ a1] - s0 [- s1]`, wrapping, with the row counts fixed
/// at compile time so the loop has no branches and vectorises.
#[inline(always)]
fn fused<const NA: usize, const NS: usize>(
    dst: &mut [i16; HL],
    src: &[i16; HL],
    a0: &[i16; HL],
    a1: &[i16; HL],
    s0: &[i16; HL],
    s1: &[i16; HL],
) {
    for h in 0..HL {
        let mut v = src[h];
        if NA > 0 {
            v = v.wrapping_add(a0[h]);
        }
        if NA > 1 {
            v = v.wrapping_add(a1[h]);
        }
        if NS > 0 {
            v = v.wrapping_sub(s0[h]);
        }
        if NS > 1 {
            v = v.wrapping_sub(s1[h]);
        }
        dst[h] = v;
    }
}

/// Full from-scratch build of both perspectives' accumulators.
pub fn refresh_all(net: &Net, board: &[u8; 64], dual: &mut DualAccumulator) {
    refresh_one(net, board, WHITE, &mut dual.acc[WHITE]);
    refresh_one(net, board, BLACK, &mut dual.acc[BLACK]);
}

// ================= Forward pass =================

/// Float SCReLU, as bullet trains it.
#[inline(always)]
fn screlu_f(x: f32) -> f32 {
    let c = x.clamp(0.0, 1.0);
    c * c
}

/// Forward pass from a pair of accumulators, relative to `stm`.
/// `piece_count` (every piece on the board, kings included) picks the
/// output bucket.
pub fn evaluate_with_net(net: &Net, acc: &DualAccumulator, stm: usize, piece_count: u32) -> i32 {
    let (us, them) = (&acc.acc[stm], &acc.acc[stm ^ 1]);
    let ob = output_bucket(piece_count);

    // SCReLU both accumulators (stm first) into the 2*HL -> L2 layer:
    // sparse integer sums, then to float in the trainer's units.
    let mut s = [0i64; L2];
    l1_forward(&us.v, &them.v, &net.l1w_pairs[ob * L2 * 2 * HL..(ob + 1) * L2 * 2 * HL], &mut s);
    float_layers(net, ob, &s)
}

/// Everything after the integer 2*HL -> L2 sums `s`: to float, SCReLU,
/// L2 -> L3 -> 1, centipawns.
#[inline]
fn float_layers(net: &Net, ob: usize, s: &[i64; L2]) -> i32 {
    let l1b = &net.l1b[ob * L2..(ob + 1) * L2];
    let mut h2 = [0f32; L2];
    l1_activate(s, l1b, &mut h2);

    // L2 -> L3 -> 1 in f32. Each L3 sum still adds its terms in input order
    // (bias, then o = 0..L2), but all L3 sums advance together, so the loop
    // vectorises instead of waiting on one long chain of dependent adds.
    let l2w = &net.l2w_t[ob * L2 * L3..(ob + 1) * L2 * L3];
    let l3w = &net.l3w[ob * L3..(ob + 1) * L3];
    let mut x = [0f32; L3];
    x.copy_from_slice(&net.l2b[ob * L3..(ob + 1) * L3]);
    let mut prod = [0f32; L3];
    l2_l3(&mut x, l2w, &h2, l3w, &mut prod);

    // The output sum stays a sequential chain in j order (bit-exact).
    let mut out = net.l3b[ob];
    for p in prod {
        out += p;
    }

    (out * SCALE as f32) as i32
}

/// `h2[o] = screlu_f((s[o] as f64 * L1_SCALE) as f32 + l1b[o])`: the 2*HL -> L2
/// layer's integer sums to its float activations. The AVX2 version is bit-identical.
#[inline]
fn l1_activate(s: &[i64; L2], l1b: &[f32], h2: &mut [f32; L2]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return l1_activate_avx2(s, l1b, h2) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return l1_activate_avx2(s, l1b, h2) }
    }
    #[allow(unreachable_code)]
    for o in 0..L2 {
        h2[o] = screlu_f((s[o] as f64 * L1_SCALE) as f32 + l1b[o]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l1_activate_avx2(s: &[i64; L2], l1b: &[f32], h2: &mut [f32; L2]) {
    use std::arch::x86_64::*;
    debug_assert!(l1b.len() >= L2);
    // Exact i64 -> f64 for |x| < 2^51 (the sums stay below 2^34): add the
    // bits of 2^52 + 2^51 as an integer, reinterpret, subtract it as a double.
    let magic_i = _mm256_set1_epi64x(0x4338_0000_0000_0000);
    let magic_d = _mm256_set1_pd(f64::from_bits(0x4338_0000_0000_0000));
    let scale = _mm256_set1_pd(L1_SCALE);
    let (zero, one) = (_mm_setzero_ps(), _mm_set1_ps(1.0));
    for q in 0..L2 / 4 {
        let x = _mm256_loadu_si256(s.as_ptr().add(q * 4) as *const __m256i);
        let d = _mm256_sub_pd(_mm256_castsi256_pd(_mm256_add_epi64(x, magic_i)), magic_d);
        let f = _mm_add_ps(_mm256_cvtpd_ps(_mm256_mul_pd(d, scale)), _mm_loadu_ps(l1b.as_ptr().add(q * 4)));
        let c = _mm_min_ps(_mm_max_ps(f, zero), one);
        _mm_storeu_ps(h2.as_mut_ptr().add(q * 4), _mm_mul_ps(c, c));
    }
}

/// `x[j] += w[o][j] * h[o]` for o = 0..L2 in order, then
/// `prod[j] = l3w[j] * screlu_f(x[j])`. The AVX2 version does the same
/// separate multiply and add per lane (no FMA), so results are bit-identical
/// to the scalar loop -- which the compiler does not vectorise on its own.
#[inline]
fn l2_l3(x: &mut [f32; L3], w: &[f32], h: &[f32; L2], l3w: &[f32], prod: &mut [f32; L3]) {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    unsafe { return l2_l3_avx2(x, w, h, l3w, prod) }
    #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
    if std::is_x86_feature_detected!("avx2") {
        unsafe { return l2_l3_avx2(x, w, h, l3w, prod) }
    }
    #[allow(unreachable_code)]
    {
        for o in 0..L2 {
            let w = &w[o * L3..(o + 1) * L3];
            for j in 0..L3 {
                x[j] += w[j] * h[o];
            }
        }
        for j in 0..L3 {
            prod[j] = l3w[j] * screlu_f(x[j]);
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn l2_l3_avx2(x: &mut [f32; L3], w: &[f32], h: &[f32; L2], l3w: &[f32], prod: &mut [f32; L3]) {
    use std::arch::x86_64::*;
    debug_assert!(w.len() >= L2 * L3 && l3w.len() >= L3);
    let mut v = [_mm256_setzero_ps(); L3 / 8];
    for (q, vq) in v.iter_mut().enumerate() {
        *vq = _mm256_loadu_ps(x.as_ptr().add(q * 8));
    }
    for (o, &ho) in h.iter().enumerate() {
        let hb = _mm256_set1_ps(ho);
        let row = w.as_ptr().add(o * L3);
        for (q, vq) in v.iter_mut().enumerate() {
            *vq = _mm256_add_ps(*vq, _mm256_mul_ps(_mm256_loadu_ps(row.add(q * 8)), hb));
        }
    }
    let (zero, one) = (_mm256_setzero_ps(), _mm256_set1_ps(1.0));
    for (q, vq) in v.iter().enumerate() {
        _mm256_storeu_ps(x.as_mut_ptr().add(q * 8), *vq);
        let c = _mm256_min_ps(_mm256_max_ps(*vq, zero), one);
        let p = _mm256_mul_ps(_mm256_loadu_ps(l3w.as_ptr().add(q * 8)), _mm256_mul_ps(c, c));
        _mm256_storeu_ps(prod.as_mut_ptr().add(q * 8), p);
    }
}

// ================= Evaluator =================

/// A search thread's evaluation state: the lazy accumulator stack, its
/// king-bucket cache and the eval cache, all tied to the one embedded net.
pub struct Evaluator {
    net: &'static Net,
    stack: AccStack,
    cache: BucketCache,
    eval_cache: EvalCache,
}

impl Default for Evaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl Evaluator {
    pub fn new() -> Self {
        Evaluator { net: net(), stack: AccStack::new(), cache: BucketCache::new(), eval_cache: EvalCache::new() }
    }

    /// Full refresh for `pos`; drops the accumulator stack and the bucket cache.
    pub fn reset(&mut self, pos: &Position) {
        self.cache.clear();
        self.stack.reset(self.net, &pos.board);
    }

    /// Record the move `pos.make_move` just made (from `pos.feats`).
    #[inline]
    pub fn push(&mut self, pos: &Position) {
        let mut buckets = [self.stack.bucket(WHITE), self.stack.bucket(BLACK)];
        let mut refresh = 0u8;
        // only the mover's own king can have changed bucket
        let us = pos.stm ^ 1;
        let b = king_bucket(us, pos.king_sq(us));
        if b != buckets[us] {
            buckets[us] = b;
            refresh = 1 << us;
        }
        let mut pieces = [[0u64; 6]; 2];
        if refresh != 0 {
            for c in 0..2 {
                for pt in 0..6 {
                    pieces[c][pt] = pos.pieces(c, pt);
                }
            }
        }
        self.stack.push(&pos.feats[..pos.nfeats], refresh, buckets, &pieces);
    }

    /// Step back to the position before the last pushed move.
    #[inline]
    pub fn pop(&mut self) {
        self.stack.pop();
    }

    /// Static evaluation of `pos` (which must match the pushed moves), side to move relative.
    pub fn evaluate(&mut self, pos: &Position) -> i32 {
        let key = pos.s().key;
        if let Some(e) = self.eval_cache.probe(key) {
            return e;
        }
        let acc = self.stack.current(self.net, &mut self.cache);

        // DEBUG-ONLY consistency check: the lazily-updated accumulators
        // must exactly match a full from-scratch refresh of this position.
        #[cfg(debug_assertions)]
        {
            let mut check = DualAccumulator::zeroed();
            refresh_all(self.net, &pos.board, &mut check);
            debug_assert_eq!(check, *acc, "lazy NNUE accumulators desynced from a full refresh");
        }

        let e = evaluate_with_net(self.net, acc, pos.stm, pos.occ().count_ones());
        self.eval_cache.store(key, e);
        e
    }
}

/// Slow reference evaluation from scratch, for the `eval` command and tests.
pub fn evaluate_full(pos: &Position) -> i32 {
    let mut acc = DualAccumulator::zeroed();
    refresh_all(net(), &pos.board, &mut acc);
    evaluate_with_net(net(), &acc, pos.stm, pos.occ().count_ones())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::moves::{GenType, MoveList};

    fn setup() {
        crate::bitboard::init();
        crate::zobrist::init_keys();
        init().unwrap();
    }

    fn pos_from(fen: &str) -> Position {
        let mut p = Position::new();
        assert!(p.set_fen(fen), "bad fen {fen}");
        p
    }

    #[test]
    fn embedded_net_parses() {
        // parse_net returning Ok also runs the all-zeros / range / NaN guards.
        assert!(parse_net(EMBEDDED_NET).is_ok());
    }

    /// Must match bullet's `MaterialCount<8>`: (count - 2) / ceil(32 / 8).
    #[test]
    fn output_bucket_matches_material_count() {
        for count in 2..=32u32 {
            assert_eq!(output_bucket(count), ((count - 2) / 4) as usize, "count {count}");
        }
    }

    /// Black reads the bucket table from its own back rank.
    #[test]
    fn king_bucket_is_perspective_relative() {
        for sq in 0..64 {
            assert_eq!(king_bucket(WHITE, sq), king_bucket(BLACK, sq ^ 56));
        }
        assert_eq!(king_bucket(WHITE, 6), 1); // g1
        assert_eq!(king_bucket(BLACK, 62), 1); // g8
    }

    /// The SIMD kernels must be bit-identical to the scalar reference.
    #[test]
    fn simd_matches_scalar() {
        // xorshift PRNG -- deterministic, no dependency
        let mut s: u64 = 0x9E3779B97F4A7C15;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };

        for trial in 0..64 {
            let mut acc_ref = [0i16; HL];
            let mut acc_simd = [0i16; HL];
            for h in 0..HL {
                let v = ((next() % 4096) as i32 - 1024) as i16;
                acc_ref[h] = v;
                acc_simd[h] = v;
            }
            let row: Vec<i16> = (0..HL).map(|_| (next() % 512) as i16 - 256).collect();

            let sub = trial & 1 == 0;
            acc_addsub_scalar(&mut acc_ref, &row, sub);
            if sub {
                acc_sub(&mut acc_simd, &row)
            } else {
                acc_add(&mut acc_simd, &row)
            }
            assert_eq!(acc_ref, acc_simd, "acc_{} mismatch on trial {trial}", if sub { "sub" } else { "add" });
        }
    }

    /// The sparse 2*HL -> L2 kernel (SIMD and scalar) must equal a plain
    /// dense i64 sum, for any sparsity -- including every accumulator at QA
    /// with every weight at the +128 bound, the worst case for its i32 lanes.
    #[test]
    fn l1_forward_matches_dense() {
        let mut s: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let dense = |c: &[i16; 2 * HL], w: &[i16]| -> [i64; L2] {
            let mut out = [0i64; L2];
            for o in 0..L2 {
                for i in 0..2 * HL {
                    let (p, k) = (i / 2, i % 2);
                    out[o] += c[i] as i64 * w[(p * L2 + o) * 2 + k] as i64;
                }
            }
            out
        };
        let halves = |a: &[i16; 2 * HL]| -> ([i16; HL], [i16; HL]) {
            (a[..HL].try_into().unwrap(), a[HL..].try_into().unwrap())
        };
        for trial in 0..40 {
            let zero_pct = [0u64, 50, 88, 99, 100][trial % 5];
            let mut acc = [0i16; 2 * HL];
            for v in acc.iter_mut() {
                *v = if next() % 100 < zero_pct { -5 } else { (next() % 300) as i16 };
            }
            let mut c = [0i16; 2 * HL];
            screlu_half(&acc, &mut c);
            let w: Vec<i16> = (0..2 * HL * L2).map(|_| (next() % 257) as i16 - 128).collect();
            let want = dense(&c, &w);
            let mut got = [0i64; L2];
            l1_sparse_scalar(&c, &w, &mut got);
            assert_eq!(got, want, "scalar, trial {trial}");
            let (us, them) = halves(&acc);
            l1_forward(&us, &them, &w, &mut got);
            assert_eq!(got, want, "dispatched, trial {trial}");
        }
        let w = vec![128i16; 2 * HL * L2];
        let mut got = [0i64; L2];
        l1_forward(&[QA_I16; HL], &[QA_I16; HL], &w, &mut got);
        assert_eq!(got, [2 * HL as i64 * ((QA * QA) >> 1) * 128; L2]);
    }

    /// Reference evaluations of this net (from its inference reference, erebus `--probe-fen`).
    #[test]
    fn inference_reference_vectors() {
        setup();
        for (fen, want) in [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 47),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1", 738),
            ("r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8", -1191),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 64),
            ("6k1/5pp1/7p/8/8/7P/5PP1/3R2K1 w - - 0 1", 1912),
            ("4k3/8/8/8/8/8/8/QQQQK3 b - - 0 1", -2048),
        ] {
            assert_eq!(evaluate_full(&pos_from(fen)), want, "{fen}");
        }
    }

    /// The SIMD activation of the L2 sums must equal the scalar formula bit for bit,
    /// across the whole range of the sums (including negatives and the clamp edges).
    #[test]
    fn l1_activate_matches_scalar() {
        let mut st: u64 = 0x1234_5678_9ABC_DEF1;
        let mut next = || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        for trial in 0..2000 {
            let span = [1i64 << 12, 1 << 22, 1 << 33][trial % 3];
            let s: [i64; L2] = std::array::from_fn(|_| (next() % (2 * span as u64)) as i64 - span);
            let l1b: Vec<f32> = (0..L2).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
            let mut got = [0f32; L2];
            l1_activate(&s, &l1b, &mut got);
            for o in 0..L2 {
                let want = screlu_f((s[o] as f64 * L1_SCALE) as f32 + l1b[o]);
                assert_eq!(got[o].to_bits(), want.to_bits(), "s={} b={}", s[o], l1b[o]);
            }
        }
    }

    /// End-to-end: the SIMD forward pass equals a plain i64 / f32 reference
    /// built straight from the net's arrays.
    #[test]
    fn forward_pass_matches_scalar() {
        setup();
        let net = net();
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ] {
            let p = pos_from(fen);
            let ob = output_bucket(p.occ().count_ones());
            let mut acc = DualAccumulator::zeroed();
            refresh_all(net, &p.board, &mut acc);
            let (us, them) = (&acc.acc[p.stm], &acc.acc[p.stm ^ 1]);
            let sq = |a: i16| -> i64 {
                let c = a.clamp(0, QA_I16) as i64;
                (c * c) >> 1
            };
            let mut h2 = [0f32; L2];
            // l1w[ob][o][i], read back through the pair layout
            let w = |o: usize, i: usize| net.l1w_pairs[((ob * HL + i / 2) * L2 + o) * 2 + i % 2] as i64;
            for o in 0..L2 {
                let mut s: i64 = 0;
                for h in 0..HL {
                    s += sq(us.v[h]) * w(o, h) + sq(them.v[h]) * w(o, HL + h);
                }
                h2[o] = screlu_f((s as f64 * L1_SCALE) as f32 + net.l1b[ob * L2 + o]);
            }
            let mut out = net.l3b[ob];
            for j in 0..L3 {
                let mut x = net.l2b[ob * L3 + j];
                for o in 0..L2 {
                    x += net.l2w_t[(ob * L2 + o) * L3 + j] * h2[o];
                }
                out += net.l3w[ob * L3 + j] * screlu_f(x);
            }
            let want = (out * SCALE as f32) as i32;
            assert_eq!(want, evaluate_full(&p), "forward pass mismatch on {fen}");
        }
    }

    /// The lazy accumulators must equal a from-scratch refresh.
    fn assert_acc_fresh(ev: &mut Evaluator, p: &Position) {
        let net = ev.net;
        let acc = ev.stack.current(net, &mut ev.cache).clone();
        let mut want = DualAccumulator::zeroed();
        refresh_all(net, &p.board, &mut want);
        assert!(acc == want, "lazy accumulator differs from a full refresh");
    }

    /// Walk the move tree reading the accumulators only at some nodes, so
    /// pending chains of every length build up and are resolved both on the
    /// way down and after unmakes.
    fn lazy_walk(ev: &mut Evaluator, p: &mut Position, depth: u32, visited: &mut u64) {
        *visited += 1;
        if depth == 0 || visited.is_multiple_of(3) {
            assert_acc_fresh(ev, p);
        }
        if depth == 0 {
            return;
        }
        let mut moves = MoveList::new();
        p.generate(GenType::All, &mut moves);
        for &m in moves.as_slice() {
            if !p.is_legal(m) {
                continue;
            }
            p.make_move(m);
            ev.push(p);
            lazy_walk(ev, p, depth - 1, visited);
            p.unmake_move(m);
            ev.pop();
        }
        if visited.is_multiple_of(5) {
            assert_acc_fresh(ev, p);
        }
    }

    #[test]
    fn lazy_accumulators_match_refresh() {
        setup();
        for (fen, depth) in [
            // kiwipete: castling, en passant, captures
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3),
            // promotions and promotion-captures
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3),
            // Chess960 castling, king and rook squares overlapping
            ("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9", 3),
            // open kings: most moves are king moves, many across bucket
            // boundaries, often several in one pending chain
            ("8/8/3k4/8/8/4K3/8/R6r w - - 0 1", 4),
        ] {
            let mut p = pos_from(fen);
            let mut ev = Evaluator::new();
            ev.reset(&p);
            let mut visited = 0;
            lazy_walk(&mut ev, &mut p, depth, &mut visited);
            assert_acc_fresh(&mut ev, &p);
        }
    }

    #[test]
    fn incremental_eval_matches_full() {
        setup();
        let mut p = pos_from("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        let mut ev = Evaluator::new();
        ev.reset(&p);
        let mut ml = MoveList::new();
        p.generate(GenType::All, &mut ml);
        for &m in ml.as_slice() {
            if !p.is_legal(m) {
                continue;
            }
            p.make_move(m);
            ev.push(&p);
            assert_eq!(ev.evaluate(&p), evaluate_full(&p));
            p.unmake_move(m);
            ev.pop();
        }
        assert_eq!(ev.evaluate(&p), evaluate_full(&p));
    }

    #[test]
    fn massive_material_advantage_is_detected() {
        setup();
        assert!(evaluate_full(&pos_from("4k3/8/8/8/8/8/8/QQQQK3 w - - 0 1")) > 800);
        assert!(evaluate_full(&pos_from("4k3/8/8/8/8/8/8/QQQQK3 b - - 0 1")) < -800);
    }
}
