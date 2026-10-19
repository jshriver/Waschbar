//! Pseudo-legal move generation, legality checks and perft.

use crate::bitboard::*;
use crate::board::*;
use crate::moves::*;

impl Position {
    // ---------------- move generation ----------------

    pub fn generate(&self, gt: GenType, ml: &mut MoveList) {
        let us = self.stm;
        let them = us ^ 1;
        let o = self.occ();
        let empty = !o;
        let enemy = self.by_color[them];
        let pawns = self.pieces(us, PAWN);
        let promo_rank = if us == WHITE { RANK_1 << 56 } else { RANK_1 };
        let rank3 = if us == WHITE { RANK_1 << 16 } else { RANK_1 << 40 };
        let up: i32 = if us == WHITE { 8 } else { -8 };
        let shift = |b: u64, d: i32| if d > 0 { b << d } else { b >> -d };
        let not_a = !FILE_A;
        let not_h = !(FILE_A << 7);
        let back = |t: usize, d: i32| (t as i32 - d) as usize;

        let push1 = shift(pawns, up) & empty;
        let push2 = shift(push1 & rank3, up) & empty;
        // left capture (towards file a): white +7, black -9; right: white +9, black -7
        let (dl, dr) = if us == WHITE { (7, 9) } else { (-9, -7) };
        let cap_l = shift(pawns & not_a, dl) & enemy;
        let cap_r = shift(pawns & not_h, dr) & enemy;

        let add_promos = |ml: &mut MoveList, from: usize, to: usize| {
            if gt != GenType::Quiets {
                ml.add(mk_move(from, to, MT_PROMO, QUEEN));
            }
            if gt != GenType::Captures {
                ml.add(mk_move(from, to, MT_PROMO, KNIGHT));
                ml.add(mk_move(from, to, MT_PROMO, ROOK));
                ml.add(mk_move(from, to, MT_PROMO, BISHOP));
            }
        };
        // promotions
        for (mut b, d) in [(push1 & promo_rank, up), (cap_l & promo_rank, dl), (cap_r & promo_rank, dr)] {
            while b != 0 {
                let t = pop_lsb(&mut b);
                add_promos(ml, back(t, d), t);
            }
        }
        if gt != GenType::Quiets {
            for (mut b, d) in [(cap_l & !promo_rank, dl), (cap_r & !promo_rank, dr)] {
                while b != 0 {
                    let t = pop_lsb(&mut b);
                    ml.add(mk_normal(back(t, d), t));
                }
            }
            let ep = self.s().ep;
            if ep != NO_SQ {
                let mut att = pawn_att(them, ep) & pawns;
                while att != 0 {
                    ml.add(mk_move(pop_lsb(&mut att), ep, MT_EP, KNIGHT));
                }
            }
        }
        if gt != GenType::Captures {
            for (mut b, d) in [(push1 & !promo_rank, up), (push2, 2 * up)] {
                while b != 0 {
                    let t = pop_lsb(&mut b);
                    ml.add(mk_normal(back(t, d), t));
                }
            }
        }
        let targets = match gt {
            GenType::Captures => enemy,
            GenType::Quiets => empty,
            GenType::All => !self.by_color[us],
        };
        for pt in KNIGHT..=KING {
            let mut b = self.pieces(us, pt);
            while b != 0 {
                let f = pop_lsb(&mut b);
                let mut a = piece_attacks(pt, f, o) & targets;
                while a != 0 {
                    ml.add(mk_normal(f, pop_lsb(&mut a)));
                }
            }
        }
        if gt != GenType::Captures && self.s().checkers == 0 {
            let c = self.castling();
            let ksq = self.king_sq(us);
            for r in 2 * us..2 * us + 2 {
                if c & (1 << r) == 0 || o & self.castle_path[r] != 0 {
                    continue;
                }
                let rsq = self.castle_rook_sq[r];
                // attacks are tested without king and castling rook (in Chess960 the rook may shield a square)
                let o2 = o ^ sqbb(ksq) ^ sqbb(rsq);
                let mut path = self.castle_king_path[r];
                let mut ok = true;
                while path != 0 && ok {
                    ok = !self.sq_attacked(pop_lsb(&mut path), them, o2);
                }
                if ok {
                    ml.add(mk_move(ksq, rsq, MT_CASTLE, KNIGHT));
                }
            }
        }
    }

    /// Legality of a pseudo-legal move.
    pub fn is_legal(&self, m: Move) -> bool {
        let (from, to) = (move_from(m), move_to(m));
        let us = self.stm;
        let them = us ^ 1;
        let ksq = self.king_sq(us);
        let x = self.s();
        if move_type(m) == MT_EP {
            let cap = if us == WHITE { to - 8 } else { to + 8 };
            let o = (self.occ() ^ sqbb(from) ^ sqbb(cap)) | sqbb(to);
            if rook_attacks(ksq, o) & (self.pieces(them, ROOK) | self.pieces(them, QUEEN)) != 0 {
                return false;
            }
            if bishop_attacks(ksq, o) & (self.pieces(them, BISHOP) | self.pieces(them, QUEEN)) != 0 {
                return false;
            }
            if x.checkers & !sqbb(cap) & (self.by_type[KNIGHT] | self.by_type[PAWN]) != 0 {
                return false;
            }
            return true;
        }
        if from == ksq {
            if move_type(m) == MT_CASTLE {
                return true; // generator verified the path
            }
            return !self.sq_attacked(to, them, self.occ() ^ sqbb(from));
        }
        if x.checkers != 0 {
            if x.checkers & (x.checkers - 1) != 0 {
                return false;
            }
            let chk = lsb(x.checkers);
            if (between(ksq, chk) | x.checkers) & sqbb(to) == 0 {
                return false;
            }
        }
        if x.pinned & sqbb(from) != 0 && line(ksq, from) & sqbb(to) == 0 {
            return false;
        }
        true
    }

    /// Could `m` (e.g. from the TT or a killer slot) be generated in this position?
    pub fn is_pseudo_legal(&self, m: Move) -> bool {
        if m == MOVE_NONE || m == MOVE_NULL {
            return false;
        }
        let (from, to) = (move_from(m), move_to(m));
        let us = self.stm;
        let p = self.piece_on(from);
        if p == NO_PIECE || piece_color(p) != us {
            return false;
        }
        if move_type(m) != MT_NORMAL {
            let mut ml = MoveList::new();
            self.generate(GenType::All, &mut ml);
            return ml.as_slice().contains(&m);
        }
        if self.by_color[us] & sqbb(to) != 0 {
            return false;
        }
        let pt = piece_type(p);
        if pt == PAWN {
            if rank_of(to) == 7 || rank_of(to) == 0 {
                return false;
            }
            let up: i32 = if us == WHITE { 8 } else { -8 };
            let (f, t) = (from as i32, to as i32);
            let ok = if self.piece_on(to) != NO_PIECE {
                pawn_att(us, from) & sqbb(to) != 0
            } else {
                t == f + up
                    || (t == f + 2 * up
                        && rank_of(from) == (if us == WHITE { 1 } else { 6 })
                        && self.piece_on((f + up) as usize) == NO_PIECE)
            };
            if !ok {
                return false;
            }
        } else if piece_attacks(pt, from, self.occ()) & sqbb(to) == 0 {
            return false;
        }
        // in double check only king moves; handled by is_legal
        true
    }

    /// Direct checks + discovered checks for normal moves (not exact for special moves).
    pub fn gives_check_approx(&self, m: Move) -> bool {
        let (from, to) = (move_from(m), move_to(m));
        let them = self.stm ^ 1;
        let eksq = self.king_sq(them);
        let mut pt = piece_type(self.piece_on(from));
        if move_type(m) == MT_PROMO {
            pt = move_promo(m);
        }
        let o = (self.occ() ^ sqbb(from)) | sqbb(to);
        if pt == PAWN {
            if pawn_att(self.stm, to) & sqbb(eksq) != 0 {
                return true;
            }
        } else if pt != KING && piece_attacks(pt, to, o) & sqbb(eksq) != 0 {
            return true;
        }
        // discovered
        (rook_attacks(eksq, o) & (self.pieces(self.stm, ROOK) | self.pieces(self.stm, QUEEN)) & !sqbb(from)) != 0
            || (bishop_attacks(eksq, o) & (self.pieces(self.stm, BISHOP) | self.pieces(self.stm, QUEEN)) & !sqbb(from))
                != 0
    }

}

pub fn perft(pos: &mut Position, depth: i32) -> u64 {
    let mut ml = MoveList::new();
    pos.generate(GenType::All, &mut ml);
    let mut n = 0;
    for &m in ml.as_slice() {
        if !pos.is_legal(m) {
            continue;
        }
        if depth == 1 {
            n += 1;
            continue;
        }
        pos.make_move(m);
        n += perft(pos, depth - 1);
        pos.unmake_move(m);
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        crate::bitboard::init();
        crate::zobrist::init_keys();
    }

    #[test]
    fn perft_standard() {
        init();
        for (fen, d, want) in [
            (START_FEN, 5, 4865609u64),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4, 4085603),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 4, 422333),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 4, 2103487),
        ] {
            let mut p = Position::new();
            assert!(p.set_fen(fen));
            assert_eq!(perft(&mut p, d), want, "{fen}");
        }
    }

    #[test]
    fn perft_chess960() {
        init();
        let mut p = Position::new();
        assert!(p.set_fen("bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9"));
        assert_eq!(perft(&mut p, 4), 326672);
    }
}
