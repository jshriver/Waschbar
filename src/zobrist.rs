//! Zobrist keys and the cuckoo tables for upcoming-repetition detection.

use crate::bitboard::*;
use crate::board::*;
use crate::moves::*;
use std::sync::Once;

pub(crate) struct Keys {
    pub(crate) piece: [[u64; 64]; 12],
    pub(crate) castle: [u64; 16],
    pub(crate) ep: [u64; 8],
    pub(crate) side: u64,
    pub(crate) cuckoo: [u64; 8192],
    pub(crate) cuckoo_move: [Move; 8192],
}

static mut KEYS: Keys = Keys {
    piece: [[0; 64]; 12],
    castle: [0; 16],
    ep: [0; 8],
    side: 0,
    cuckoo: [0; 8192],
    cuckoo_move: [0; 8192],
};
static KEYS_INIT: Once = Once::new();

#[inline(always)]
pub(crate) fn keys() -> &'static Keys {
    // SAFETY: only `init_keys` writes KEYS, exactly once, before any reader runs.
    unsafe { &*(&raw const KEYS) }
}
#[inline(always)]
pub(crate) fn zpiece(p: usize, s: usize) -> u64 {
    keys().piece[p][s]
}

#[inline(always)]
pub(crate) fn cuckoo_h1(k: u64) -> usize {
    (k & 0x1FFF) as usize
}
#[inline(always)]
pub(crate) fn cuckoo_h2(k: u64) -> usize {
    ((k >> 16) & 0x1FFF) as usize
}

/// Zobrist keys and the cuckoo tables for upcoming-repetition detection (idempotent;
/// needs the attack tables).
pub fn init_keys() {
    KEYS_INIT.call_once(|| {
        // SAFETY: runs once, before any reader.
        let k = unsafe { &mut *(&raw mut KEYS) };
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        for p in 0..12 {
            for s in 0..64 {
                k.piece[p][s] = rng();
            }
        }
        let cr: [u64; 4] = std::array::from_fn(|_| rng());
        for c in 0..16 {
            k.castle[c] = (0..4).filter(|i| c & (1 << i) != 0).fold(0, |acc, i| acc ^ cr[i]);
        }
        for f in 0..8 {
            k.ep[f] = rng();
        }
        k.side = rng();

        for p in 0..12 {
            let pt = piece_type(p);
            if pt == PAWN {
                continue;
            }
            for s1 in 0..64 {
                for s2 in s1 + 1..64 {
                    if piece_attacks(pt, s1, 0) & sqbb(s2) == 0 {
                        continue;
                    }
                    let mut mv = mk_normal(s1, s2);
                    let mut key = k.piece[p][s1] ^ k.piece[p][s2] ^ k.side;
                    let mut i = cuckoo_h1(key);
                    loop {
                        std::mem::swap(&mut k.cuckoo[i], &mut key);
                        std::mem::swap(&mut k.cuckoo_move[i], &mut mv);
                        if mv == MOVE_NONE {
                            break;
                        }
                        i = if i == cuckoo_h1(key) { cuckoo_h2(key) } else { cuckoo_h1(key) };
                    }
                }
            }
        }
    });
}
