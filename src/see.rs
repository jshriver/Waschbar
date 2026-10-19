//! Static exchange evaluation.

use crate::bitboard::*;
use crate::board::*;
use crate::moves::*;

impl Position {
    /// Static exchange evaluation: does m win at least `threshold`?
    pub fn see_ge(&self, m: Move, threshold: i32) -> bool {
        const SEE_VAL: [i32; 7] = [100, 300, 300, 500, 900, 20000, 0];
        if move_type(m) != MT_NORMAL {
            return 0 >= threshold;
        }
        let (from, to) = (move_from(m), move_to(m));
        let victim = if self.piece_on(to) == NO_PIECE { 6 } else { piece_type(self.piece_on(to)) };
        let mut swap = SEE_VAL[victim] - threshold;
        if swap < 0 {
            return false;
        }
        swap = SEE_VAL[piece_type(self.piece_on(from))] - swap;
        if swap <= 0 {
            return true;
        }
        let mut o = self.occ() ^ sqbb(from) ^ sqbb(to);
        let mut side = self.stm;
        let mut attackers = self.attackers_to(to, o);
        let bq = self.by_type[BISHOP] | self.by_type[QUEEN];
        let rq = self.by_type[ROOK] | self.by_type[QUEEN];
        let mut res = 1;
        loop {
            side ^= 1;
            attackers &= o;
            let sa = attackers & self.by_color[side];
            if sa == 0 {
                break;
            }
            res ^= 1;
            let mut b = sa & self.by_type[PAWN];
            if b != 0 {
                swap = SEE_VAL[PAWN] - swap;
                if swap < res {
                    break;
                }
                o ^= b & b.wrapping_neg();
                attackers |= bishop_attacks(to, o) & bq;
                continue;
            }
            b = sa & self.by_type[KNIGHT];
            if b != 0 {
                swap = SEE_VAL[KNIGHT] - swap;
                if swap < res {
                    break;
                }
                o ^= b & b.wrapping_neg();
                continue;
            }
            b = sa & self.by_type[BISHOP];
            if b != 0 {
                swap = SEE_VAL[BISHOP] - swap;
                if swap < res {
                    break;
                }
                o ^= b & b.wrapping_neg();
                attackers |= bishop_attacks(to, o) & bq;
                continue;
            }
            b = sa & self.by_type[ROOK];
            if b != 0 {
                swap = SEE_VAL[ROOK] - swap;
                if swap < res {
                    break;
                }
                o ^= b & b.wrapping_neg();
                attackers |= rook_attacks(to, o) & rq;
                continue;
            }
            b = sa & self.by_type[QUEEN];
            if b != 0 {
                swap = SEE_VAL[QUEEN] - swap;
                if swap < res {
                    break;
                }
                o ^= b & b.wrapping_neg();
                attackers |= (bishop_attacks(to, o) & bq) | (rook_attacks(to, o) & rq);
                continue;
            }
            // king: it may only capture if the other side has no attacker left
            return if attackers & !self.by_color[side] != 0 { res ^ 1 != 0 } else { res != 0 };
        }
        res != 0
    }
}
