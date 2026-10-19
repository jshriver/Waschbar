//! Position: board state, FEN, make/unmake, draw and repetition detection.

use crate::bitboard::*;
use crate::moves::*;
use crate::nnue::FeatDelta;
use crate::zobrist::*;
use std::sync::atomic::{AtomicBool, Ordering};

pub const WHITE: usize = 0;
pub const BLACK: usize = 1;

pub const PAWN: usize = 0;
pub const KNIGHT: usize = 1;
pub const BISHOP: usize = 2;
pub const ROOK: usize = 3;
pub const QUEEN: usize = 4;
pub const KING: usize = 5;
pub const NO_PT: usize = 6;

/// Pieces are `color * 6 + piece_type` (0..12); an empty square holds `NO_PIECE`.
pub const NO_PIECE: usize = 12;
pub const NO_SQ: usize = 64;

#[inline(always)]
pub const fn make_piece(c: usize, pt: usize) -> usize {
    c * 6 + pt
}
#[inline(always)]
pub const fn piece_type(p: usize) -> usize {
    p % 6
}
#[inline(always)]
pub const fn piece_color(p: usize) -> usize {
    p / 6
}

#[derive(Clone, Copy, Default)]
pub struct StateInfo {
    pub key: u64,
    pub pawn_key: u64,
    pub np_key: [u64; 2],
    pub checkers: u64,
    /// Pieces of the side to move pinned to their own king.
    pub pinned: u64,
    pub castling: usize,
    pub ep: usize,
    pub halfmove: i32,
    /// Piece captured by the move that led here.
    pub captured: usize,
    pub plies_from_null: i32,
}

pub const MAX_STATES: usize = 4096;

/// UCI_Chess960: castling is written king-takes-rook, FEN uses rook files.
pub static CHESS960: AtomicBool = AtomicBool::new(false);

pub fn chess960() -> bool {
    CHESS960.load(Ordering::Relaxed)
}

/// Castling right index r (bit 1 << r): 0 = white O-O, 1 = white O-O-O, 2 = black O-O, 3 = black O-O-O.
/// A castling move is encoded king-from -> own-rook-square with MT_CASTLE (works for Chess960 and DFRC).
#[inline(always)]
pub const fn castle_king_to(r: usize) -> usize {
    (if r >= 2 { 56 } else { 0 }) + if r & 1 != 0 { 2 } else { 6 }
}
#[inline(always)]
pub const fn castle_rook_to(r: usize) -> usize {
    (if r >= 2 { 56 } else { 0 }) + if r & 1 != 0 { 3 } else { 5 }
}

#[derive(Clone)]
pub struct Position {
    pub by_type: [u64; 6],
    pub by_color: [u64; 2],
    pub board: [u8; 64],
    pub stm: usize,
    /// Index of the current state in `st`.
    pub sp: usize,
    pub game_ply: i32,
    pub st: Vec<StateInfo>,
    // castling geometry, fixed for the game (set by set_fen)
    pub(crate) castle_rook_sq: [usize; 4],
    /// Squares that must be empty (excluding king and rook).
    pub(crate) castle_path: [u64; 4],
    /// Squares the king crosses incl. destination (must not be attacked).
    pub(crate) castle_king_path: [u64; 4],
    /// Rights kept when a piece moves from / to the square.
    castle_mask: [u8; 64],
    /// NNUE input changes made by the last make_move, as `(piece, square, added)`.
    pub feats: [FeatDelta; 4],
    pub nfeats: usize,
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}

impl Position {
    pub fn new() -> Self {
        let mut p = Position {
            by_type: [0; 6],
            by_color: [0; 2],
            board: [NO_PIECE as u8; 64],
            stm: WHITE,
            sp: 0,
            game_ply: 0,
            st: vec![StateInfo::default(); MAX_STATES],
            castle_rook_sq: [NO_SQ; 4],
            castle_path: [0; 4],
            castle_king_path: [0; 4],
            castle_mask: [15; 64],
            feats: [(0, 0, false); 4],
            nfeats: 0,
        };
        p.clear();
        p
    }

    pub fn startpos() -> Self {
        let mut p = Position::new();
        p.set_fen(START_FEN);
        p
    }

    #[inline(always)]
    pub fn s(&self) -> &StateInfo {
        &self.st[self.sp]
    }
    #[inline(always)]
    pub fn occ(&self) -> u64 {
        self.by_color[0] | self.by_color[1]
    }
    #[inline(always)]
    pub fn pieces(&self, c: usize, pt: usize) -> u64 {
        self.by_color[c] & self.by_type[pt]
    }
    #[inline(always)]
    pub fn piece_on(&self, s: usize) -> usize {
        self.board[s] as usize
    }
    #[inline(always)]
    pub fn king_sq(&self, c: usize) -> usize {
        lsb(self.pieces(c, KING))
    }
    #[inline(always)]
    pub fn in_check(&self) -> bool {
        self.st[self.sp].checkers != 0
    }
    #[inline(always)]
    pub fn has_non_pawn(&self, c: usize) -> bool {
        self.by_color[c] & (self.by_type[KNIGHT] | self.by_type[BISHOP] | self.by_type[ROOK] | self.by_type[QUEEN]) != 0
    }
    #[inline(always)]
    pub fn castling(&self) -> usize {
        self.st[self.sp].castling
    }

    #[inline(always)]
    fn put(&mut self, p: usize, s: usize) {
        self.board[s] = p as u8;
        self.by_type[piece_type(p)] |= sqbb(s);
        self.by_color[piece_color(p)] |= sqbb(s);
    }
    #[inline(always)]
    fn remove(&mut self, s: usize) {
        let p = self.piece_on(s);
        self.by_type[piece_type(p)] ^= sqbb(s);
        self.by_color[piece_color(p)] ^= sqbb(s);
        self.board[s] = NO_PIECE as u8;
    }
    #[inline(always)]
    fn move_piece(&mut self, f: usize, t: usize) {
        let p = self.piece_on(f);
        let ft = sqbb(f) | sqbb(t);
        self.by_type[piece_type(p)] ^= ft;
        self.by_color[piece_color(p)] ^= ft;
        self.board[t] = p as u8;
        self.board[f] = NO_PIECE as u8;
    }

    #[inline]
    pub fn attackers_to(&self, s: usize, occ: u64) -> u64 {
        (pawn_att(BLACK, s) & self.pieces(WHITE, PAWN))
            | (pawn_att(WHITE, s) & self.pieces(BLACK, PAWN))
            | (knight_att(s) & self.by_type[KNIGHT])
            | (king_att(s) & self.by_type[KING])
            | (bishop_attacks(s, occ) & (self.by_type[BISHOP] | self.by_type[QUEEN]))
            | (rook_attacks(s, occ) & (self.by_type[ROOK] | self.by_type[QUEEN]))
    }

    #[inline]
    pub fn sq_attacked(&self, s: usize, by_c: usize, occ: u64) -> bool {
        (pawn_att(by_c ^ 1, s) & self.pieces(by_c, PAWN)) != 0
            || (knight_att(s) & self.pieces(by_c, KNIGHT)) != 0
            || (king_att(s) & self.pieces(by_c, KING)) != 0
            || (bishop_attacks(s, occ) & (self.pieces(by_c, BISHOP) | self.pieces(by_c, QUEEN))) != 0
            || (rook_attacks(s, occ) & (self.pieces(by_c, ROOK) | self.pieces(by_c, QUEEN))) != 0
    }

    #[inline(always)]
    pub fn is_capture(&self, m: Move) -> bool {
        (self.piece_on(move_to(m)) != NO_PIECE && move_type(m) != MT_CASTLE) || move_type(m) == MT_EP
    }
    /// Castling right index of an MT_CASTLE move.
    #[inline(always)]
    fn castle_right(&self, m: Move) -> usize {
        2 * self.stm + (move_to(m) < move_from(m)) as usize
    }
    #[inline(always)]
    pub fn is_tactical(&self, m: Move) -> bool {
        self.is_capture(m) || (move_type(m) == MT_PROMO && move_promo(m) == QUEEN)
    }

    /// Copies position + recent history (enough for repetition detection) from `o`.
    pub fn copy_from(&mut self, o: &Position) {
        self.by_type = o.by_type;
        self.by_color = o.by_color;
        self.board = o.board;
        self.stm = o.stm;
        self.game_ply = o.game_ply;
        self.castle_rook_sq = o.castle_rook_sq;
        self.castle_path = o.castle_path;
        self.castle_king_path = o.castle_king_path;
        self.castle_mask = o.castle_mask;
        let keep = o.sp.min((o.st[o.sp].halfmove.max(0) as usize).min(200));
        self.st[..=keep].copy_from_slice(&o.st[o.sp - keep..=o.sp]);
        self.sp = keep;
        self.nfeats = 0;
    }

    /// Keeps only the last `keep` plies of history (the caller ensures sp >= keep).
    pub fn compact_history(&mut self, keep: usize) {
        let sp = self.sp;
        self.st.copy_within(sp - keep..=sp, 0);
        self.sp = keep;
    }

    fn clear(&mut self) {
        self.by_type = [0; 6];
        self.by_color = [0; 2];
        self.board = [NO_PIECE as u8; 64];
        self.stm = WHITE;
        self.sp = 0;
        self.game_ply = 0;
        self.st[0] = StateInfo { ep: NO_SQ, ..Default::default() };
        self.castle_rook_sq = [NO_SQ; 4];
        self.castle_path = [0; 4];
        self.castle_king_path = [0; 4];
        self.castle_mask = [15; 64];
        self.nfeats = 0;
    }

    fn add_castling(&mut self, c: usize, rsq: usize) {
        let ksq = self.king_sq(c);
        let back = if c == WHITE { 0 } else { 7 };
        if rank_of(ksq) != back || rank_of(rsq) != back || self.piece_on(rsq) != make_piece(c, ROOK) {
            return;
        }
        let r = 2 * c + (rsq < ksq) as usize;
        let (kto, rto) = (castle_king_to(r), castle_rook_to(r));
        self.castle_rook_sq[r] = rsq;
        self.castle_path[r] =
            (between(rsq, rto) | between(ksq, kto) | sqbb(rto) | sqbb(kto)) & !(sqbb(ksq) | sqbb(rsq));
        self.castle_king_path[r] = between(ksq, kto) | sqbb(kto);
        self.castle_mask[ksq] &= !(if c == WHITE { 3 } else { 12 });
        self.castle_mask[rsq] &= !(1 << r);
        self.st[0].castling |= 1 << r;
    }

    fn compute_key(&self) -> u64 {
        let mut k = 0;
        for s in 0..64 {
            if self.piece_on(s) != NO_PIECE {
                k ^= zpiece(self.piece_on(s), s);
            }
        }
        let x = self.s();
        k ^= keys().castle[x.castling];
        if x.ep != NO_SQ {
            k ^= keys().ep[file_of(x.ep)];
        }
        if self.stm == BLACK {
            k ^= keys().side;
        }
        k
    }

    fn compute_pawn_key(&self) -> u64 {
        let mut k = 0;
        let mut b = self.by_type[PAWN];
        while b != 0 {
            let s = pop_lsb(&mut b);
            k ^= zpiece(self.piece_on(s), s);
        }
        k
    }

    fn compute_np_key(&self, c: usize) -> u64 {
        let mut k = 0;
        let mut b = self.by_color[c] & !self.by_type[PAWN];
        while b != 0 {
            let s = pop_lsb(&mut b);
            k ^= zpiece(self.piece_on(s), s);
        }
        k
    }

    fn compute_check_info(&mut self) {
        let ksq = self.king_sq(self.stm);
        let them = self.stm ^ 1;
        let o = self.occ();
        let checkers = self.attackers_to(ksq, o) & self.by_color[them];
        let mut pinned = 0;
        let mut snipers = (rook_attacks(ksq, 0) & (self.pieces(them, ROOK) | self.pieces(them, QUEEN)))
            | (bishop_attacks(ksq, 0) & (self.pieces(them, BISHOP) | self.pieces(them, QUEEN)));
        while snipers != 0 {
            let sn = pop_lsb(&mut snipers);
            let b = between(ksq, sn) & o;
            if b != 0 && b & (b - 1) == 0 {
                pinned |= b & self.by_color[self.stm];
            }
        }
        let x = &mut self.st[self.sp];
        x.checkers = checkers;
        x.pinned = pinned;
    }

    /// Sets up a FEN (X-FEN / Shredder-FEN castling). Returns false if it is not usable.
    pub fn set_fen(&mut self, fen: &str) -> bool {
        self.clear();
        let mut it = fen.split_whitespace();
        let bd = it.next().unwrap_or("");
        let side = it.next().unwrap_or("");
        let cast = it.next().unwrap_or("");
        let ep = it.next().unwrap_or("");
        let hm: i32 = it.next().and_then(|t| t.parse().ok()).unwrap_or(0);
        let fm: i32 = it.next().and_then(|t| t.parse().ok()).unwrap_or(1);
        let (mut r, mut f) = (7i32, 0i32);
        for c in bd.chars() {
            if c == '/' {
                r -= 1;
                f = 0;
            } else if ('1'..='8').contains(&c) {
                f += c as i32 - '0' as i32;
            } else {
                let Some(idx) = "PNBRQKpnbrqk".find(c) else { return false };
                if r < 0 || f > 7 {
                    return false;
                }
                self.put(make_piece(idx / 6, idx % 6), (r * 8 + f) as usize);
                f += 1;
            }
        }
        if popcount(self.pieces(WHITE, KING)) != 1 || popcount(self.pieces(BLACK, KING)) != 1 {
            return false;
        }
        self.stm = if side == "b" { BLACK } else { WHITE };
        // the side not to move must not be in check
        if self.sq_attacked(self.king_sq(self.stm ^ 1), self.stm, self.occ()) {
            return false;
        }
        // castling: KQkq (outermost rook, X-FEN) or Shredder-FEN rook files (AHah)
        self.st[0].castling = 0;
        for ch in cast.chars() {
            if ch == '-' {
                break;
            }
            let c = if ch.is_ascii_uppercase() { WHITE } else { BLACK };
            let u = ch.to_ascii_uppercase();
            let back = if c == WHITE { 0 } else { 56 };
            let ksq = self.king_sq(c);
            let rooks = self.pieces(c, ROOK) & (RANK_1 << back);
            let mut rsq = NO_SQ;
            if u == 'K' {
                rsq = (ksq + 1..=back + 7).rev().find(|&s| rooks & sqbb(s) != 0).unwrap_or(NO_SQ);
            } else if u == 'Q' {
                rsq = (back..ksq).find(|&s| rooks & sqbb(s) != 0).unwrap_or(NO_SQ);
            } else if ('A'..='H').contains(&u) {
                rsq = back + (u as usize - 'A' as usize);
            }
            if rsq != NO_SQ {
                self.add_castling(c, rsq);
            }
        }
        self.st[0].ep = NO_SQ;
        let eb = ep.as_bytes();
        if eb.len() == 2 && (b'a'..=b'h').contains(&eb[0]) && (b'1'..=b'8').contains(&eb[1]) {
            let s = (eb[1] - b'1') as usize * 8 + (eb[0] - b'a') as usize;
            if pawn_att(self.stm ^ 1, s) & self.pieces(self.stm, PAWN) != 0 {
                self.st[0].ep = s;
            }
        }
        self.st[0].halfmove = hm;
        self.st[0].captured = NO_PIECE;
        self.st[0].plies_from_null = 0;
        self.game_ply = 2 * (fm - 1) + (self.stm == BLACK) as i32;
        self.st[0].key = self.compute_key();
        self.st[0].pawn_key = self.compute_pawn_key();
        self.st[0].np_key = [self.compute_np_key(WHITE), self.compute_np_key(BLACK)];
        self.compute_check_info();
        true
    }

    pub fn fen(&self) -> String {
        let mut r = String::new();
        for rk in (0..8).rev() {
            let mut e = 0;
            for f in 0..8 {
                let p = self.piece_on(rk * 8 + f);
                if p == NO_PIECE {
                    e += 1;
                    continue;
                }
                if e > 0 {
                    r.push((b'0' + e) as char);
                    e = 0;
                }
                r.push(b"PNBRQKpnbrqk"[p] as char);
            }
            if e > 0 {
                r.push((b'0' + e) as char);
            }
            if rk > 0 {
                r.push('/');
            }
        }
        r += if self.stm == WHITE { " w " } else { " b " };
        let c = self.castling();
        if c == 0 {
            r.push('-');
        }
        for i in 0..4 {
            if c & (1 << i) != 0 {
                let f = (b'a' + file_of(self.castle_rook_sq[i]) as u8) as char;
                r.push(if chess960() {
                    if i < 2 { f.to_ascii_uppercase() } else { f }
                } else {
                    b"KQkq"[i] as char
                });
            }
        }
        r.push(' ');
        let x = self.s();
        if x.ep == NO_SQ {
            r.push('-');
        } else {
            r.push((b'a' + file_of(x.ep) as u8) as char);
            r.push((b'1' + rank_of(x.ep) as u8) as char);
        }
        r += &format!(" {} {}", x.halfmove, self.game_ply / 2 + 1);
        r
    }

    pub fn make_move(&mut self, m: Move) {
        let prev = self.st[self.sp];
        self.sp += 1;
        self.game_ply += 1;
        let mut x = StateInfo {
            castling: prev.castling,
            halfmove: prev.halfmove + 1,
            plies_from_null: prev.plies_from_null + 1,
            np_key: prev.np_key,
            ep: NO_SQ,
            ..Default::default()
        };
        let mut key = prev.key ^ keys().side;
        let mut pkey = prev.pawn_key;
        if prev.ep != NO_SQ {
            key ^= keys().ep[file_of(prev.ep)];
        }

        let (from, to, mt) = (move_from(m), move_to(m), move_type(m));
        let us = self.stm;
        let them = us ^ 1;
        let p = self.piece_on(from);
        let pt = piece_type(p);
        let mut nf = 0;
        macro_rules! feat {
            ($pc:expr, $sq:expr, $add:expr) => {{
                self.feats[nf] = ($pc as u8, $sq as u8, $add);
                nf += 1;
            }};
        }

        let mut captured = NO_PIECE;
        if mt == MT_CASTLE {
            let cr = self.castle_right(m);
            let (rf, rt, kt) = (to, castle_rook_to(cr), castle_king_to(cr));
            let rook = self.piece_on(rf);
            self.remove(from);
            self.remove(rf);
            self.put(p, kt);
            self.put(rook, rt);
            let d = zpiece(p, from) ^ zpiece(p, kt) ^ zpiece(rook, rf) ^ zpiece(rook, rt);
            key ^= d;
            x.np_key[us] ^= d;
            feat!(p, from, false);
            feat!(p, kt, true);
            feat!(rook, rf, false);
            feat!(rook, rt, true);
        } else {
            let capsq = if mt == MT_EP { if us == WHITE { to - 8 } else { to + 8 } } else { to };
            captured = self.piece_on(capsq);
            feat!(p, from, false);
            if captured != NO_PIECE {
                self.remove(capsq);
                key ^= zpiece(captured, capsq);
                if piece_type(captured) == PAWN {
                    pkey ^= zpiece(captured, capsq);
                } else {
                    x.np_key[them] ^= zpiece(captured, capsq);
                }
                x.halfmove = 0;
                feat!(captured, capsq, false);
            }
            self.move_piece(from, to);
            key ^= zpiece(p, from) ^ zpiece(p, to);
            if pt != PAWN {
                x.np_key[us] ^= zpiece(p, from) ^ zpiece(p, to);
                feat!(p, to, true);
            } else {
                x.halfmove = 0;
                pkey ^= zpiece(p, from);
                if mt == MT_PROMO {
                    let np = make_piece(us, move_promo(m));
                    self.remove(to);
                    self.put(np, to);
                    key ^= zpiece(p, to) ^ zpiece(np, to);
                    x.np_key[us] ^= zpiece(np, to);
                    feat!(np, to, true);
                } else {
                    feat!(p, to, true);
                    pkey ^= zpiece(p, to);
                    if to ^ from == 16 {
                        let epsq = (to + from) / 2;
                        if pawn_att(us, epsq) & self.pieces(them, PAWN) != 0 {
                            x.ep = epsq;
                            key ^= keys().ep[file_of(epsq)];
                        }
                    }
                }
            }
        }
        self.nfeats = nf;
        x.captured = captured;
        if x.castling != 0 {
            let nc = x.castling & self.castle_mask[from] as usize & self.castle_mask[to] as usize;
            if nc != x.castling {
                key ^= keys().castle[x.castling] ^ keys().castle[nc];
                x.castling = nc;
            }
        }
        x.key = key;
        x.pawn_key = pkey;
        self.st[self.sp] = x;
        self.stm = them;
        self.compute_check_info();
    }

    pub fn unmake_move(&mut self, m: Move) {
        let (from, to, mt) = (move_from(m), move_to(m), move_type(m));
        self.stm ^= 1;
        let us = self.stm;
        if mt == MT_CASTLE {
            let cr = self.castle_right(m);
            let (rt, kt) = (castle_rook_to(cr), castle_king_to(cr));
            let (king, rook) = (self.piece_on(kt), self.piece_on(rt));
            self.remove(kt);
            self.remove(rt);
            self.put(king, from);
            self.put(rook, to);
        } else {
            if mt == MT_PROMO {
                self.remove(to);
                self.put(make_piece(us, PAWN), to);
            }
            self.move_piece(to, from);
            let captured = self.st[self.sp].captured;
            if captured != NO_PIECE {
                let capsq = if mt == MT_EP { if us == WHITE { to - 8 } else { to + 8 } } else { to };
                self.put(captured, capsq);
            }
        }
        self.sp -= 1;
        self.game_ply -= 1;
    }

    pub fn make_null(&mut self) {
        let prev = self.st[self.sp];
        self.sp += 1;
        self.game_ply += 1;
        let mut x = prev;
        x.key ^= keys().side;
        if prev.ep != NO_SQ {
            x.key ^= keys().ep[file_of(prev.ep)];
        }
        x.ep = NO_SQ;
        x.halfmove = prev.halfmove + 1;
        x.plies_from_null = 0;
        x.captured = NO_PIECE;
        self.st[self.sp] = x;
        self.nfeats = 0;
        self.stm ^= 1;
        self.compute_check_info();
    }

    pub fn unmake_null(&mut self) {
        self.stm ^= 1;
        self.sp -= 1;
        self.game_ply -= 1;
    }

    /// 50-move rule and repetitions (inside the search tree a single repetition is a draw;
    /// before the root it takes a threefold).
    pub fn is_draw(&self, ply: i32) -> bool {
        let x = self.s();
        if x.halfmove >= 100 {
            if x.checkers == 0 {
                return true;
            }
            let mut ml = MoveList::new();
            self.generate(GenType::All, &mut ml);
            return ml.as_slice().iter().any(|&m| self.is_legal(m));
        }
        let end = (x.halfmove.min(x.plies_from_null).max(0) as usize).min(self.sp);
        let mut reps = 0;
        let mut i = 4;
        while i <= end {
            if self.st[self.sp - i].key == x.key {
                if (i as i32) < ply {
                    return true;
                }
                reps += 1;
                if reps >= 2 {
                    return true;
                }
            }
            i += 2;
        }
        false
    }

    /// Can the side to move force a repetition with a single reversible move (cuckoo tables)?
    pub fn upcoming_repetition(&self, ply: i32) -> bool {
        let x = self.s();
        let end = (x.halfmove.min(x.plies_from_null).max(0) as usize).min(self.sp);
        if end < 3 {
            return false;
        }
        let sp = self.sp;
        let orig = x.key;
        let mut other = orig ^ self.st[sp - 1].key ^ keys().side;
        let o = self.occ();
        let k = keys();
        let mut i = 3;
        while i <= end {
            other ^= self.st[sp - i + 1].key ^ self.st[sp - i].key ^ k.side;
            if other == 0 {
                let move_key = orig ^ self.st[sp - i].key;
                let mut j = cuckoo_h1(move_key);
                if k.cuckoo[j] != move_key {
                    j = cuckoo_h2(move_key);
                }
                if k.cuckoo[j] == move_key {
                    let mv = k.cuckoo_move[j];
                    if between(move_from(mv), move_to(mv)) & o == 0 && ply > i as i32 {
                        return true;
                    }
                }
            }
            i += 2;
        }
        false
    }

}

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

#[cfg(test)]
mod tests {
    use super::*;

    fn init() {
        crate::bitboard::init();
        crate::zobrist::init_keys();
    }

    #[test]
    fn fen_round_trip_and_keys() {
        init();
        let mut p = Position::new();
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        assert!(p.set_fen(fen));
        assert_eq!(p.fen(), fen);
        // incremental keys match a fresh computation after every legal move
        let mut ml = MoveList::new();
        p.generate(GenType::All, &mut ml);
        for &m in ml.as_slice() {
            if !p.is_legal(m) {
                continue;
            }
            p.make_move(m);
            assert_eq!(p.s().key, p.compute_key());
            assert_eq!(p.s().pawn_key, p.compute_pawn_key());
            assert_eq!(p.s().np_key, [p.compute_np_key(WHITE), p.compute_np_key(BLACK)]);
            p.unmake_move(m);
        }
        assert_eq!(p.fen(), fen);
    }
}
