//! Attack tables and square-set helpers. Sliders use PEXT on x86-64 with BMI2 and magic
//! bitboards elsewhere (ARM, or x86 without BMI2); both index the same tables.

use crate::board::*;
use std::sync::Once;

#[inline(always)]
pub const fn lsb(b: u64) -> usize {
    b.trailing_zeros() as usize
}
#[inline(always)]
pub const fn popcount(b: u64) -> i32 {
    b.count_ones() as i32
}
#[inline(always)]
pub fn pop_lsb(b: &mut u64) -> usize {
    let s = lsb(*b);
    *b &= *b - 1;
    s
}
#[inline(always)]
pub const fn sqbb(s: usize) -> u64 {
    1u64 << s
}
#[inline(always)]
pub const fn rank_of(s: usize) -> usize {
    s >> 3
}
#[inline(always)]
pub const fn file_of(s: usize) -> usize {
    s & 7
}

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const RANK_1: u64 = 0xFF;

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
const USE_PEXT: bool = true;
#[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
const USE_PEXT: bool = false;

const ROOK_TABLE_SIZE: usize = 102400;
const BISHOP_TABLE_SIZE: usize = 5248;

pub struct Tables {
    pub pawn_att: [[u64; 64]; 2],
    pub knight_att: [u64; 64],
    pub king_att: [u64; 64],
    rook_mask: [u64; 64],
    bishop_mask: [u64; 64],
    rook_off: [usize; 64],
    bishop_off: [usize; 64],
    rook_magic: [u64; 64],
    bishop_magic: [u64; 64],
    rook_shift: [u32; 64],
    bishop_shift: [u32; 64],
    pub between: [[u64; 64]; 64],
    pub line: [[u64; 64]; 64],
    rook_table: [u64; ROOK_TABLE_SIZE],
    bishop_table: [u64; BISHOP_TABLE_SIZE],
}

static mut TABLES: Tables = Tables {
    pawn_att: [[0; 64]; 2],
    knight_att: [0; 64],
    king_att: [0; 64],
    rook_mask: [0; 64],
    bishop_mask: [0; 64],
    rook_off: [0; 64],
    bishop_off: [0; 64],
    rook_magic: [0; 64],
    bishop_magic: [0; 64],
    rook_shift: [0; 64],
    bishop_shift: [0; 64],
    between: [[0; 64]; 64],
    line: [[0; 64]; 64],
    rook_table: [0; ROOK_TABLE_SIZE],
    bishop_table: [0; BISHOP_TABLE_SIZE],
};
static INIT: Once = Once::new();

/// The attack tables. Written once by `init` (before any other thread exists), read-only after.
#[inline(always)]
pub fn tables() -> &'static Tables {
    // SAFETY: only `init` writes TABLES, exactly once, before any reader runs.
    unsafe { &*(&raw const TABLES) }
}

#[inline(always)]
pub fn pawn_att(c: usize, s: usize) -> u64 {
    tables().pawn_att[c][s]
}
#[inline(always)]
pub fn knight_att(s: usize) -> u64 {
    tables().knight_att[s]
}
#[inline(always)]
pub fn king_att(s: usize) -> u64 {
    tables().king_att[s]
}
#[inline(always)]
pub fn between(a: usize, b: usize) -> u64 {
    tables().between[a][b]
}
#[inline(always)]
pub fn line(a: usize, b: usize) -> u64 {
    tables().line[a][b]
}

#[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
#[inline(always)]
fn pext(occ: u64, mask: u64) -> usize {
    // SAFETY: compiled only when the target has BMI2.
    unsafe { std::arch::x86_64::_pext_u64(occ, mask) as usize }
}

#[inline(always)]
pub fn rook_attacks(s: usize, occ: u64) -> u64 {
    let t = tables();
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let idx = pext(occ, t.rook_mask[s]);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let idx = ((occ & t.rook_mask[s]).wrapping_mul(t.rook_magic[s]) >> t.rook_shift[s]) as usize;
    t.rook_table[t.rook_off[s] + idx]
}

#[inline(always)]
pub fn bishop_attacks(s: usize, occ: u64) -> u64 {
    let t = tables();
    #[cfg(all(target_arch = "x86_64", target_feature = "bmi2"))]
    let idx = pext(occ, t.bishop_mask[s]);
    #[cfg(not(all(target_arch = "x86_64", target_feature = "bmi2")))]
    let idx = ((occ & t.bishop_mask[s]).wrapping_mul(t.bishop_magic[s]) >> t.bishop_shift[s]) as usize;
    t.bishop_table[t.bishop_off[s] + idx]
}

#[inline(always)]
pub fn queen_attacks(s: usize, occ: u64) -> u64 {
    rook_attacks(s, occ) | bishop_attacks(s, occ)
}

#[inline(always)]
pub fn piece_attacks(pt: usize, s: usize, occ: u64) -> u64 {
    match pt {
        KNIGHT => knight_att(s),
        BISHOP => bishop_attacks(s, occ),
        ROOK => rook_attacks(s, occ),
        QUEEN => queen_attacks(s, occ),
        KING => king_att(s),
        _ => 0,
    }
}

const ROOK_DIRS: [(i32, i32); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i32, i32); 4] = [(1, 1), (-1, 1), (1, -1), (-1, -1)];

fn slide(s: usize, occ: u64, dirs: &[(i32, i32); 4]) -> u64 {
    let mut r = 0;
    for &(df, dr) in dirs {
        let (mut f, mut k) = (file_of(s) as i32, rank_of(s) as i32);
        loop {
            f += df;
            k += dr;
            if !(0..8).contains(&f) || !(0..8).contains(&k) {
                break;
            }
            let b = sqbb((k * 8 + f) as usize);
            r |= b;
            if occ & b != 0 {
                break;
            }
        }
    }
    r
}

/// Fills a slider's table for square s. With PEXT the index is pext(occ, mask), i.e. the
/// subset's position in carry-rippler order. Otherwise searches a magic multiplier (fixed seed,
/// so the result is deterministic) mapping every subset to an index without a wrong collision.
fn init_slider(
    s: usize,
    mask: u64,
    dirs: &[(i32, i32); 4],
    table: &mut [u64],
    magic: &mut u64,
    shift: &mut u32,
    seed: &mut u64,
) {
    let mut occ = Vec::with_capacity(4096);
    let mut att = Vec::with_capacity(4096);
    let mut b = 0u64;
    loop {
        occ.push(b);
        att.push(slide(s, b, dirs));
        b = b.wrapping_sub(mask) & mask;
        if b == 0 {
            break;
        }
    }
    *shift = 64 - mask.count_ones();
    if USE_PEXT {
        table[..att.len()].copy_from_slice(&att);
        *magic = 0;
        return;
    }
    let mut stamp = vec![0u32; att.len()];
    let mut epoch = 0u32;
    let mut rnd = || {
        *seed ^= *seed >> 12;
        *seed ^= *seed << 25;
        *seed ^= *seed >> 27;
        seed.wrapping_mul(2685821657736338717)
    };
    loop {
        let m = rnd() & rnd() & rnd();
        if (mask.wrapping_mul(m) >> 56).count_ones() < 6 {
            continue;
        }
        epoch += 1;
        let mut ok = true;
        for i in 0..occ.len() {
            let idx = (occ[i].wrapping_mul(m) >> *shift) as usize;
            if stamp[idx] != epoch {
                stamp[idx] = epoch;
                table[idx] = att[i];
            } else if table[idx] != att[i] {
                ok = false;
                break;
            }
        }
        if ok {
            *magic = m;
            return;
        }
    }
}

/// Builds the attack tables (idempotent).
pub fn init() {
    INIT.call_once(|| {
        // SAFETY: runs once, before any reader (call_once also orders it before later calls).
        let t = unsafe { &mut *(&raw mut TABLES) };
        for s in 0..64 {
            let (f, r) = (file_of(s) as i32, rank_of(s) as i32);
            let add = |b: &mut u64, df: i32, dr: i32| {
                let (nf, nr) = (f + df, r + dr);
                if (0..8).contains(&nf) && (0..8).contains(&nr) {
                    *b |= sqbb((nr * 8 + nf) as usize);
                }
            };
            add(&mut t.pawn_att[WHITE][s], -1, 1);
            add(&mut t.pawn_att[WHITE][s], 1, 1);
            add(&mut t.pawn_att[BLACK][s], -1, -1);
            add(&mut t.pawn_att[BLACK][s], 1, -1);
            for (df, dr) in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)] {
                add(&mut t.knight_att[s], df, dr);
            }
            for df in -1..=1 {
                for dr in -1..=1 {
                    if df != 0 || dr != 0 {
                        add(&mut t.king_att[s], df, dr);
                    }
                }
            }
        }
        let (mut rp, mut bp) = (0usize, 0usize);
        let mut seed = 0x5EED_5EED_1234_5678u64;
        for s in 0..64 {
            let edges = ((RANK_1 | (RANK_1 << 56)) & !(RANK_1 << (8 * rank_of(s))))
                | ((FILE_A | (FILE_A << 7)) & !(FILE_A << file_of(s)));
            t.rook_mask[s] = slide(s, 0, &ROOK_DIRS) & !edges;
            t.bishop_mask[s] = slide(s, 0, &BISHOP_DIRS) & !edges;
            t.rook_off[s] = rp;
            t.bishop_off[s] = bp;
            let rn = 1usize << t.rook_mask[s].count_ones();
            let bn = 1usize << t.bishop_mask[s].count_ones();
            init_slider(
                s,
                t.rook_mask[s],
                &ROOK_DIRS,
                &mut t.rook_table[rp..rp + rn],
                &mut t.rook_magic[s],
                &mut t.rook_shift[s],
                &mut seed,
            );
            init_slider(
                s,
                t.bishop_mask[s],
                &BISHOP_DIRS,
                &mut t.bishop_table[bp..bp + bn],
                &mut t.bishop_magic[s],
                &mut t.bishop_shift[s],
                &mut seed,
            );
            rp += rn;
            bp += bn;
        }
        for a in 0..64 {
            for b in 0..64 {
                t.between[a][b] = 0;
                t.line[a][b] = 0;
                if a == b {
                    continue;
                }
                if slide(a, 0, &ROOK_DIRS) & sqbb(b) != 0 {
                    t.between[a][b] = slide(a, sqbb(b), &ROOK_DIRS) & slide(b, sqbb(a), &ROOK_DIRS);
                    t.line[a][b] = (slide(a, 0, &ROOK_DIRS) & slide(b, 0, &ROOK_DIRS)) | sqbb(a) | sqbb(b);
                } else if slide(a, 0, &BISHOP_DIRS) & sqbb(b) != 0 {
                    t.between[a][b] = slide(a, sqbb(b), &BISHOP_DIRS) & slide(b, sqbb(a), &BISHOP_DIRS);
                    t.line[a][b] = (slide(a, 0, &BISHOP_DIRS) & slide(b, 0, &BISHOP_DIRS)) | sqbb(a) | sqbb(b);
                }
            }
        }
    });
}
