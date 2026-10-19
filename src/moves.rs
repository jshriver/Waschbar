//! Move encoding, `MoveList` and generation types.

use crate::board::KNIGHT;
use std::mem::MaybeUninit;

pub type Move = u16;

// Move encoding: from(6) | to(6) | type(2) | promo(2)
pub const MT_NORMAL: usize = 0;
pub const MT_PROMO: usize = 1;
pub const MT_EP: usize = 2;
pub const MT_CASTLE: usize = 3;

#[inline(always)]
pub const fn mk_move(from: usize, to: usize, ty: usize, promo_pt: usize) -> Move {
    (from | (to << 6) | (ty << 12) | ((promo_pt - KNIGHT) << 14)) as Move
}
#[inline(always)]
pub const fn mk_normal(from: usize, to: usize) -> Move {
    mk_move(from, to, MT_NORMAL, KNIGHT)
}
#[inline(always)]
pub const fn move_from(m: Move) -> usize {
    (m & 63) as usize
}
#[inline(always)]
pub const fn move_to(m: Move) -> usize {
    ((m >> 6) & 63) as usize
}
#[inline(always)]
pub const fn move_type(m: Move) -> usize {
    ((m >> 12) & 3) as usize
}
#[inline(always)]
pub const fn move_promo(m: Move) -> usize {
    ((m >> 14) & 3) as usize + KNIGHT
}
pub const MOVE_NONE: Move = 0;
pub const MOVE_NULL: Move = 65;

/// Up to 256 moves. The buffer is left uninitialised (a list is built at every node);
/// only `moves[..size]` is ever read.
pub struct MoveList {
    moves: [MaybeUninit<Move>; 256],
    size: usize,
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl MoveList {
    #[inline(always)]
    pub fn new() -> Self {
        MoveList { moves: [const { MaybeUninit::uninit() }; 256], size: 0 }
    }
    #[inline(always)]
    pub fn add(&mut self, m: Move) {
        self.moves[self.size] = MaybeUninit::new(m);
        self.size += 1;
    }
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.size
    }
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }
    #[inline(always)]
    pub fn clear(&mut self) {
        self.size = 0;
    }
    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: moves[..size] were all written by `add`.
        unsafe { std::slice::from_raw_parts(self.moves.as_ptr() as *const Move, self.size) }
    }
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        // SAFETY: as in `as_slice`.
        unsafe { std::slice::from_raw_parts_mut(self.moves.as_mut_ptr() as *mut Move, self.size) }
    }
}

impl Clone for MoveList {
    fn clone(&self) -> Self {
        let mut l = MoveList::new();
        for &m in self.as_slice() {
            l.add(m);
        }
        l
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum GenType {
    Captures,
    Quiets,
    All,
}
