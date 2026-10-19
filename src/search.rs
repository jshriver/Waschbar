//! Alpha-beta search: PVS with null move, LMR, singular extensions, probcut, TB probes.

use std::mem::MaybeUninit;
use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::bitboard::*;
use crate::board::*;
use crate::eval::*;
use crate::moves::*;
use crate::nnue::Evaluator;
use crate::syzygy::{self, TbSearch};
use crate::tt::*;
use crate::uci::{self, Limits, PONDERING, STOP, STOP_ON_PONDERHIT, TimeManager};

// ---------------------------------------------------------------- history

pub const PH_SIZE: usize = 1024;
pub const CORR_SIZE: usize = 16384;
pub const CORR_GRAIN: i32 = 256;

/// Continuation-history row of the sentinel "no previous move": all zeros, never written.
pub const CH_NONE: usize = 12 * 64;

#[derive(Clone, Copy)]
pub struct SStack {
    pub pv: [Move; MAX_PLY as usize + 1],
    pub pv_len: usize,
    pub killer: [Move; 2],
    pub eval: i32,
    pub mv: Move,
    pub piece: usize,
    pub excluded: Move,
    /// Continuation-history row (`piece * 64 + to`) of the move made at this ply, or CH_NONE.
    pub ch: usize,
    pub double_ext: i32,
    pub cutoff_cnt: i32,
}

impl SStack {
    pub const ZERO: SStack = SStack {
        pv: [0; MAX_PLY as usize + 1],
        pv_len: 0,
        killer: [0; 2],
        eval: 0,
        mv: 0,
        piece: 0,
        excluded: 0,
        ch: 0,
        double_ext: 0,
        cutoff_cnt: 0,
    };
}

pub struct Histories {
    pub main_hist: [[i16; 4096]; 2],
    pub cap_hist: [[[i16; 6]; 64]; 12],
    pub pawn_hist: [[[i16; 64]; 12]; PH_SIZE],
    /// `[piece * 64 + to of the earlier move][piece][to]`; row CH_NONE stays zero.
    pub cont_hist: [[[i16; 64]; 12]; 12 * 64 + 1],
    pub counter_move: [[Move; 64]; 12],
    pub corr_hist: [[i32; CORR_SIZE]; 2],
    pub np_corr: [[[i32; CORR_SIZE]; 2]; 2],
    pub cont_corr: [[[i32; 64]; 12]; 2],
}

impl Histories {
    pub fn new_boxed() -> Box<Histories> {
        zeroed_box()
    }

    pub fn clear(&mut self) {
        // SAFETY: Histories is plain integer arrays; all-zero is valid.
        unsafe { std::ptr::write_bytes(self as *mut Histories, 0, 1) };
    }
}

/// Heap-allocates a zero-filled `T` without building it on the stack first (the history
/// tables are megabytes). Only for plain-old-data types whose all-zero bit pattern is valid.
pub fn zeroed_box<T>() -> Box<T> {
    use std::alloc::{Layout, alloc_zeroed, handle_alloc_error};
    let layout = Layout::new::<T>();
    assert!(layout.size() > 0);
    // SAFETY: non-zero size; callers only use integer arrays, for which zero is valid.
    unsafe {
        let p = alloc_zeroed(layout) as *mut T;
        if p.is_null() {
            handle_alloc_error(layout);
        }
        Box::from_raw(p)
    }
}

// ---------------------------------------------------------------- move picker

pub const PIECE_VAL: [i32; 7] = [100, 300, 300, 500, 900, 0, 0];

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Stage {
    Tt,
    CapGen,
    Cap,
    K1,
    K2,
    Cm,
    QGen,
    Q,
    Bad,
    Done,
    QsTt,
    QsGen,
    Qs,
    EvTt,
    EvGen,
    Ev,
}

impl Stage {
    fn next(self) -> Stage {
        match self {
            Stage::Tt => Stage::CapGen,
            Stage::QsTt => Stage::QsGen,
            Stage::EvTt => Stage::EvGen,
            s => s,
        }
    }
}

/// The picker keeps no reference to the position or histories (the search changes the
/// position between calls and restores it); `next` gets them each time.
pub struct Picker {
    ttm: Move,
    pub k1: Move,
    pub k2: Move,
    pub cm: Move,
    stage: Stage,
    list: MoveList,
    /// scores[..list.len()], written before they are read
    scores: [MaybeUninit<i32>; 256],
    idx: usize,
    bad: MoveList,
    bad_idx: usize,
    pub skip_quiets: bool,
    /// Continuation-history rows of the moves 1, 2 and 4 plies back.
    ch: [usize; 3],
}

impl Picker {
    /// `si` is this node's index in `stack` (at least 4).
    pub fn new(pos: &Position, h: &Histories, stack: &[SStack], si: usize, tm: Move, qs: bool) -> Picker {
        let ss = &stack[si];
        let prev = &stack[si - 1];
        let pm = prev.mv;
        let cm = if pm != 0 && pm != MOVE_NULL { h.counter_move[prev.piece][move_to(pm)] } else { MOVE_NONE };
        let mut stage = if pos.in_check() {
            Stage::EvTt
        } else if qs {
            Stage::QsTt
        } else {
            Stage::Tt
        };
        let mut ttm = tm;
        if ttm == 0 || !pos.is_pseudo_legal(ttm) || (qs && !pos.in_check() && !pos.is_tactical(ttm)) {
            ttm = MOVE_NONE;
            stage = stage.next();
        }
        Picker {
            ttm,
            k1: ss.killer[0],
            k2: ss.killer[1],
            cm,
            stage,
            list: MoveList::new(),
            scores: [const { MaybeUninit::uninit() }; 256],
            idx: 0,
            bad: MoveList::new(),
            bad_idx: 0,
            skip_quiets: false,
            ch: [stack[si - 1].ch, stack[si - 2].ch, stack[si - 4].ch],
        }
    }

    fn cap_score(&self, pos: &Position, h: &Histories, m: Move) -> i32 {
        let to = move_to(m);
        let victim = if move_type(m) == MT_EP {
            PAWN
        } else if pos.piece_on(to) == NO_PIECE {
            NO_PT
        } else {
            piece_type(pos.piece_on(to))
        };
        let vt = if victim == NO_PT { 0 } else { victim };
        let mut s = PIECE_VAL[victim] * 16;
        if move_type(m) == MT_PROMO {
            s += 8000;
        }
        s + h.cap_hist[pos.piece_on(move_from(m))][to][vt] as i32
    }

    fn quiet_score(&self, pos: &Position, h: &Histories, m: Move) -> i32 {
        let (pc, to) = (pos.piece_on(move_from(m)), move_to(m));
        h.main_hist[pos.stm][(m & 4095) as usize] as i32
            + h.pawn_hist[pos.s().pawn_key as usize & (PH_SIZE - 1)][pc][to] as i32
            + h.cont_hist[self.ch[0]][pc][to] as i32
            + h.cont_hist[self.ch[1]][pc][to] as i32
            + h.cont_hist[self.ch[2]][pc][to] as i32 / 2
    }

    fn score_caps(&mut self, pos: &Position, h: &Histories) {
        for i in 0..self.list.len() {
            self.scores[i] = MaybeUninit::new(self.cap_score(pos, h, self.list.as_slice()[i]));
        }
    }

    fn score_quiets(&mut self, pos: &Position, h: &Histories) {
        // squares attacked by enemy pawns (tp), by minors or better (tm), by rooks or better (tr)
        let them = pos.stm ^ 1;
        let occ = pos.occ();
        let ep = pos.pieces(them, PAWN);
        let tp = if them == WHITE {
            ((ep & !FILE_A) << 7) | ((ep & !(FILE_A << 7)) << 9)
        } else {
            ((ep & !FILE_A) >> 9) | ((ep & !(FILE_A << 7)) >> 7)
        };
        let mut tm = tp;
        let mut b = pos.pieces(them, KNIGHT);
        while b != 0 {
            tm |= knight_att(pop_lsb(&mut b));
        }
        b = pos.pieces(them, BISHOP);
        while b != 0 {
            tm |= bishop_attacks(pop_lsb(&mut b), occ);
        }
        let mut tr = tm;
        b = pos.pieces(them, ROOK);
        while b != 0 {
            tr |= rook_attacks(pop_lsb(&mut b), occ);
        }
        for i in 0..self.list.len() {
            let m = self.list.as_slice()[i];
            let mut s = self.quiet_score(pos, h, m);
            let pt = piece_type(pos.piece_on(move_from(m)));
            let (f, t) = (sqbb(move_from(m)), sqbb(move_to(m)));
            let (threat, escape, enter) = match pt {
                QUEEN => (tr, 16000, 14000),
                ROOK => (tm, 12000, 10000),
                KNIGHT | BISHOP => (tp, 8000, 7000),
                _ => (0, 0, 0),
            };
            if f & threat != 0 {
                s += escape;
            }
            if t & threat != 0 {
                s -= enter;
            }
            self.scores[i] = MaybeUninit::new(s);
        }
    }

    #[inline(always)]
    fn score(&self, i: usize) -> i32 {
        debug_assert!(i < self.list.len());
        // SAFETY: every scores[i] with i < list.len() is written right after generation.
        unsafe { self.scores[i].assume_init() }
    }

    fn pick_best(&mut self) -> Move {
        let mut b = self.idx;
        let mut bs = self.score(b);
        for i in self.idx + 1..self.list.len() {
            let s = self.score(i);
            if s > bs {
                b = i;
                bs = s;
            }
        }
        self.list.as_mut_slice().swap(b, self.idx);
        self.scores.swap(b, self.idx);
        self.idx += 1;
        self.list.as_slice()[self.idx - 1]
    }

    fn is_quiet_ok(&self, pos: &Position, m: Move) -> bool {
        m != 0 && m != self.ttm && !pos.is_tactical(m) && pos.is_pseudo_legal(m)
    }

    /// The next pseudo-legal move, or MOVE_NONE when exhausted.
    pub fn next(&mut self, pos: &Position, h: &Histories) -> Move {
        loop {
            match self.stage {
                Stage::Tt | Stage::QsTt | Stage::EvTt => {
                    self.stage = self.stage.next();
                    return self.ttm;
                }
                Stage::CapGen => {
                    self.list.clear();
                    pos.generate(GenType::Captures, &mut self.list);
                    self.score_caps(pos, h);
                    self.idx = 0;
                    self.stage = Stage::Cap;
                }
                Stage::Cap => {
                    while self.idx < self.list.len() {
                        let m = self.pick_best();
                        if m == self.ttm {
                            continue;
                        }
                        if !pos.see_ge(m, -self.score(self.idx - 1) / 32) {
                            self.bad.add(m);
                            continue;
                        }
                        return m;
                    }
                    self.stage = Stage::K1;
                }
                Stage::K1 => {
                    self.stage = Stage::K2;
                    if !self.skip_quiets && self.is_quiet_ok(pos, self.k1) {
                        return self.k1;
                    }
                }
                Stage::K2 => {
                    self.stage = Stage::Cm;
                    if !self.skip_quiets && self.k2 != self.k1 && self.is_quiet_ok(pos, self.k2) {
                        return self.k2;
                    }
                }
                Stage::Cm => {
                    self.stage = Stage::QGen;
                    if !self.skip_quiets && self.cm != self.k1 && self.cm != self.k2 && self.is_quiet_ok(pos, self.cm)
                    {
                        return self.cm;
                    }
                }
                Stage::QGen => {
                    self.list.clear();
                    self.idx = 0;
                    if !self.skip_quiets {
                        pos.generate(GenType::Quiets, &mut self.list);
                        self.score_quiets(pos, h);
                    }
                    self.stage = Stage::Q;
                }
                Stage::Q => {
                    while !self.skip_quiets && self.idx < self.list.len() {
                        let m = self.pick_best();
                        if m == self.ttm || m == self.k1 || m == self.k2 || m == self.cm {
                            continue;
                        }
                        return m;
                    }
                    self.stage = Stage::Bad;
                }
                Stage::Bad => {
                    while self.bad_idx < self.bad.len() {
                        let m = self.bad.as_slice()[self.bad_idx];
                        self.bad_idx += 1;
                        if m != self.ttm {
                            return m;
                        }
                    }
                    self.stage = Stage::Done;
                    return MOVE_NONE;
                }
                Stage::QsGen => {
                    self.list.clear();
                    pos.generate(GenType::Captures, &mut self.list);
                    self.score_caps(pos, h);
                    self.idx = 0;
                    self.stage = Stage::Qs;
                }
                Stage::Qs | Stage::Ev => {
                    while self.idx < self.list.len() {
                        let m = self.pick_best();
                        if m == self.ttm {
                            continue;
                        }
                        return m;
                    }
                    self.stage = Stage::Done;
                    return MOVE_NONE;
                }
                Stage::EvGen => {
                    self.list.clear();
                    pos.generate(GenType::All, &mut self.list);
                    for i in 0..self.list.len() {
                        let m = self.list.as_slice()[i];
                        self.scores[i] = MaybeUninit::new(if pos.is_capture(m) {
                            1_000_000 + self.cap_score(pos, h, m)
                        } else {
                            self.quiet_score(pos, h, m)
                        });
                    }
                    self.idx = 0;
                    self.stage = Stage::Ev;
                }
                Stage::Done => return MOVE_NONE,
            }
        }
    }
}

// ---------------------------------------------------------------- search

static LMR: LazyLock<[[i32; 64]; 64]> = LazyLock::new(|| {
    let mut t = [[0; 64]; 64];
    for d in 1..64 {
        for m in 1..64 {
            t[d][m] = (1.1 + (d as f64).ln() * (m as f64).ln() / 1.85) as i32;
        }
    }
    t
});

#[inline(always)]
fn lmr(depth: i32, move_count: i32) -> i32 {
    LMR[depth.min(63) as usize][move_count.min(63) as usize]
}

#[inline(always)]
fn score_to_tt(s: i32, ply: i32) -> i32 {
    if s >= TB_WIN_BOUND {
        s + ply
    } else if s <= -TB_WIN_BOUND {
        s - ply
    } else {
        s
    }
}
#[inline(always)]
fn score_from_tt(s: i32, ply: i32) -> i32 {
    if s == VALUE_NONE {
        s
    } else if s >= TB_WIN_BOUND {
        s - ply
    } else if s <= -TB_WIN_BOUND {
        s + ply
    } else {
        s
    }
}

#[inline(always)]
fn hist_update(h: &mut i16, bonus: i32) {
    let bonus = bonus.clamp(-16000, 16000);
    let v = *h as i32;
    *h = (v + bonus - v * bonus.abs() / 16384) as i16;
}

#[inline(always)]
fn stat_bonus(d: i32) -> i32 {
    (170 * d - 80).min(1700)
}

#[inline(always)]
fn corr_entry(e: &mut i32, diff: i32, w: i32) {
    *e = (*e * (256 - w) + diff * CORR_GRAIN * w) / 256;
    *e = (*e).clamp(-64 * CORR_GRAIN, 64 * CORR_GRAIN);
}

/// Node and tablebase-hit counters of one thread, readable by the others.
#[repr(align(64))]
#[derive(Default)]
pub struct Counters {
    pub nodes: AtomicU64,
    pub tb_hits: AtomicU64,
}

const STACK_SIZE: usize = MAX_PLY as usize + 16;

/// Per-thread search state (Lazy SMP: one Worker per thread, all sharing the TT).
pub struct Worker {
    pub id: usize,
    pub pos: Position,
    pub ev: Evaluator,
    pub h: Box<Histories>,
    stack: Vec<SStack>,
    pub ctr: Arc<Counters>,
    pub seldepth: i32,
    root_depth: i32,
    root_move_nodes: Vec<u64>,
    pub root_moves: MoveList,

    // result of the last completed iteration
    pub completed_depth: i32,
    pub best_score: i32,
    pub best_move: Move,
    pub pv: [Move; MAX_PLY as usize + 1],
    pub pv_len: usize,

    // per-search settings, copied in before the search starts
    pub time: TimeManager,
    pub tb: TbSearch,
}

impl Worker {
    pub fn new(id: usize, ctr: Arc<Counters>) -> Worker {
        Worker {
            id,
            pos: Position::new(),
            ev: Evaluator::new(),
            h: Histories::new_boxed(),
            stack: vec![SStack::ZERO; STACK_SIZE],
            ctr,
            seldepth: 0,
            root_depth: 0,
            root_move_nodes: vec![0; 4096],
            root_moves: MoveList::new(),
            completed_depth: 0,
            best_score: 0,
            best_move: MOVE_NONE,
            pv: [0; MAX_PLY as usize + 1],
            pv_len: 0,
            time: TimeManager::default(),
            tb: TbSearch::default(),
        }
    }

    pub fn clear(&mut self) {
        self.h.clear();
    }

    #[inline(always)]
    pub fn nodes(&self) -> u64 {
        self.ctr.nodes.load(Relaxed)
    }

    #[inline(always)]
    fn inc_nodes(&self) {
        self.ctr.nodes.store(self.ctr.nodes.load(Relaxed) + 1, Relaxed);
    }

    fn is_root_move(&self, m: Move) -> bool {
        self.root_moves.as_slice().contains(&m)
    }

    #[inline(always)]
    fn stopped() -> bool {
        STOP.load(Relaxed)
    }

    #[inline]
    fn check_time(&self) -> bool {
        if STOP.load(Relaxed) {
            return true;
        }
        if self.id == 0
            && self.nodes() & 1023 == 0
            && self.time.timed
            && self.root_depth > 1
            && !PONDERING.load(Relaxed)
            && self.time.elapsed() >= self.time.hard_limit
        {
            STOP.store(true, Relaxed);
        }
        STOP.load(Relaxed)
    }

    #[inline]
    fn do_move(&mut self, m: Move, si: usize) {
        let piece = self.pos.piece_on(move_from(m));
        let ss = &mut self.stack[si];
        ss.mv = m;
        ss.piece = piece;
        ss.ch = piece * 64 + move_to(m);
        self.pos.make_move(m);
        TT.prefetch(self.pos.s().key);
        self.ev.push(&self.pos);
        self.inc_nodes();
    }

    #[inline]
    fn undo_move(&mut self, m: Move) {
        self.pos.unmake_move(m);
        self.ev.pop();
    }

    #[inline]
    fn evaluate(&mut self) -> i32 {
        self.ev.evaluate(&self.pos)
    }

    fn corrected_eval(&self, raw: i32, si: usize) -> i32 {
        let x = self.pos.s();
        let stm = self.pos.stm;
        let h = &self.h;
        let mut c = (2 * h.corr_hist[stm][x.pawn_key as usize & (CORR_SIZE - 1)]
            + h.np_corr[stm][0][x.np_key[0] as usize & (CORR_SIZE - 1)]
            + h.np_corr[stm][1][x.np_key[1] as usize & (CORR_SIZE - 1)])
            / (3 * CORR_GRAIN);
        let prev = &self.stack[si - 1];
        if prev.mv != 0 && prev.mv != MOVE_NULL {
            c += h.cont_corr[stm][prev.piece][move_to(prev.mv)] / (2 * CORR_GRAIN);
        }
        let v = raw * (200 - x.halfmove) / 200 + c;
        v.clamp(-TB_WIN_BOUND + 1, TB_WIN_BOUND - 1)
    }

    fn update_corr(&mut self, diff: i32, depth: i32, si: usize) {
        let x = *self.pos.s();
        let stm = self.pos.stm;
        let w = (depth + 1).min(16);
        let (pm, ppiece) = (self.stack[si - 1].mv, self.stack[si - 1].piece);
        let h = &mut self.h;
        corr_entry(&mut h.corr_hist[stm][x.pawn_key as usize & (CORR_SIZE - 1)], diff, w);
        corr_entry(&mut h.np_corr[stm][0][x.np_key[0] as usize & (CORR_SIZE - 1)], diff, w);
        corr_entry(&mut h.np_corr[stm][1][x.np_key[1] as usize & (CORR_SIZE - 1)], diff, w);
        if pm != 0 && pm != MOVE_NULL {
            corr_entry(&mut h.cont_corr[stm][ppiece][move_to(pm)], diff, w);
        }
    }

    /// Copies the child's PV behind `m` into this ply's PV.
    #[inline]
    fn update_pv(&mut self, si: usize, m: Move) {
        let (lo, hi) = self.stack.split_at_mut(si + 1);
        let (ss, ch) = (&mut lo[si], &hi[0]);
        ss.pv[0] = m;
        ss.pv[1..=ch.pv_len].copy_from_slice(&ch.pv[..ch.pv_len]);
        ss.pv_len = ch.pv_len + 1;
    }

    // ---------------------------------------------------------------- qsearch
    fn qsearch<const PV: bool>(&mut self, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        let si = ply as usize + 4;
        if PV {
            self.stack[si].pv_len = 0;
        }
        if self.check_time() {
            return 0;
        }
        if ply > self.seldepth {
            self.seldepth = ply;
        }
        if self.pos.is_draw(ply) {
            return 0;
        }
        let in_check = self.pos.in_check();
        if ply >= MAX_PLY - 1 {
            return if in_check { 0 } else { let e = self.evaluate(); self.corrected_eval(e, si) };
        }

        let key = self.pos.s().key;
        let (te, hit) = TT.probe(key);
        let tt_score = if hit { score_from_tt(te.score(), ply) } else { VALUE_NONE };
        let tt_move = if hit { te.mv() } else { MOVE_NONE };
        let tt_bound = if hit { te.bound() } else { BOUND_NONE };
        if !PV
            && hit
            && tt_score != VALUE_NONE
            && (tt_bound == BOUND_EXACT
                || (tt_bound == BOUND_LOWER && tt_score >= beta)
                || (tt_bound == BOUND_UPPER && tt_score <= alpha))
        {
            return tt_score;
        }

        let mut best_score;
        let mut raw_eval = VALUE_NONE;
        let fut_base;
        if in_check {
            best_score = -INF;
            fut_base = -INF;
            self.stack[si].eval = VALUE_NONE;
        } else {
            raw_eval = if hit && te.eval() != VALUE_NONE { te.eval() } else { self.evaluate() };
            best_score = self.corrected_eval(raw_eval, si);
            self.stack[si].eval = best_score;
            if hit
                && tt_score != VALUE_NONE
                && ((tt_bound == BOUND_LOWER && tt_score > best_score)
                    || (tt_bound == BOUND_UPPER && tt_score < best_score))
            {
                best_score = tt_score;
            }
            if best_score >= beta {
                if !hit {
                    TT.store(te, key, MOVE_NONE, score_to_tt(best_score, ply), raw_eval, 0, BOUND_LOWER, false);
                }
                return best_score;
            }
            if best_score > alpha {
                alpha = best_score;
            }
            fut_base = best_score + 160;
        }

        let mut mp = Picker::new(&self.pos, &self.h, &self.stack, si, tt_move, true);
        let mut best_move = MOVE_NONE;
        let mut move_count = 0;
        let old_alpha = alpha;
        loop {
            let m = mp.next(&self.pos, &self.h);
            if m == MOVE_NONE {
                break;
            }
            if !self.pos.is_legal(m) {
                continue;
            }
            move_count += 1;
            if best_score > -TB_WIN_BOUND {
                if !in_check && move_type(m) != MT_PROMO && fut_base > -INF {
                    let victim =
                        if move_type(m) == MT_EP { PAWN } else { piece_type(self.pos.piece_on(move_to(m))) };
                    if fut_base + PIECE_VAL[victim] <= alpha && !self.pos.gives_check_approx(m) {
                        best_score = best_score.max(fut_base + PIECE_VAL[victim]);
                        continue;
                    }
                }
                if in_check && move_count > 2 && !self.pos.is_capture(m) {
                    continue; // limited quiet evasions
                }
                if !self.pos.see_ge(m, -50) {
                    continue;
                }
            }
            self.do_move(m, si);
            let score = -self.qsearch::<PV>(-beta, -alpha, ply + 1);
            self.undo_move(m);
            if Self::stopped() {
                return 0;
            }
            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    alpha = score;
                    if PV {
                        self.update_pv(si, m);
                    }
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        if in_check && best_score == -INF {
            return -MATE + ply;
        }
        let bound = if best_score >= beta {
            BOUND_LOWER
        } else if PV && best_score > old_alpha {
            BOUND_EXACT
        } else {
            BOUND_UPPER
        };
        TT.store(te, key, best_move, score_to_tt(best_score, ply), raw_eval, 0, bound, false);
        best_score
    }

    // ---------------------------------------------------------------- main search
    fn search<const PV: bool>(&mut self, mut alpha: i32, mut beta: i32, mut depth: i32, ply: i32, cutnode: bool) -> i32 {
        let si = ply as usize + 4;
        let root = ply == 0;
        if PV {
            self.stack[si].pv_len = 0;
        }
        if depth <= 0 {
            return self.qsearch::<PV>(alpha, beta, ply);
        }
        if self.check_time() {
            return 0;
        }
        if PV && ply > self.seldepth {
            self.seldepth = ply;
        }
        let in_check = self.pos.in_check();

        if !root {
            if self.pos.is_draw(ply) {
                return 0;
            }
            if alpha < 0 && self.pos.upcoming_repetition(ply) {
                alpha = 0;
                if alpha >= beta {
                    return alpha;
                }
            }
            if ply >= MAX_PLY - 1 {
                return if in_check { 0 } else { let e = self.evaluate(); self.corrected_eval(e, si) };
            }
            alpha = alpha.max(-MATE + ply);
            beta = beta.min(MATE - ply - 1);
            if alpha >= beta {
                return alpha;
            }
        }

        let excluded = self.stack[si].excluded;
        let key = self.pos.s().key;
        let (te, mut hit) = TT.probe(key);
        if excluded != 0 {
            hit = false;
        }
        let tt_score = if hit { score_from_tt(te.score(), ply) } else { VALUE_NONE };
        let mut tt_move = if root {
            self.stack[si].pv[0]
        } else if hit {
            te.mv()
        } else {
            MOVE_NONE
        };
        if root && tt_move == 0 && hit {
            tt_move = te.mv();
        }
        let tt_bound = if hit { te.bound() } else { BOUND_NONE };
        let tt_depth = if hit { te.depth() } else { 0 };
        let tt_pv = PV || (hit && te.is_pv());
        let tt_capture = tt_move != 0 && self.pos.is_tactical(tt_move);

        if !PV
            && hit
            && tt_depth >= depth
            && tt_score != VALUE_NONE
            && self.pos.s().halfmove < 90
            && (tt_bound == BOUND_EXACT
                || (tt_bound == BOUND_LOWER && tt_score >= beta)
                || (tt_bound == BOUND_UPPER && tt_score <= alpha))
        {
            // reward quiet tt move on fail high
            if tt_move != 0 && tt_score >= beta && !tt_capture && self.pos.is_pseudo_legal(tt_move) {
                hist_update(&mut self.h.main_hist[self.pos.stm][(tt_move & 4095) as usize], stat_bonus(depth));
            }
            return tt_score;
        }

        let mut best_score = -INF;
        let mut max_score = INF;

        // Syzygy WDL probe
        if !root && excluded == 0 && self.tb.card != 0 {
            let pc = popcount(self.pos.occ());
            if pc <= self.tb.card
                && (pc < self.tb.card || depth >= self.tb.probe_depth)
                && self.pos.s().halfmove == 0
                && self.pos.castling() == 0
                && let Some(wdl) = self.tb.tables.as_deref().and_then(|t| syzygy::probe_wdl(t, &self.pos))
            {
                self.ctr.tb_hits.store(self.ctr.tb_hits.load(Relaxed) + 1, Relaxed);
                let draw_score = self.tb.rule50 as i32;
                let v = if wdl < -draw_score {
                    -TB_WIN_SCORE + ply
                } else if wdl > draw_score {
                    TB_WIN_SCORE - ply
                } else {
                    2 * wdl * draw_score
                };
                let b = if wdl < -draw_score {
                    BOUND_UPPER
                } else if wdl > draw_score {
                    BOUND_LOWER
                } else {
                    BOUND_EXACT
                };
                if b == BOUND_EXACT || (if b == BOUND_LOWER { v >= beta } else { v <= alpha }) {
                    TT.store(te, key, MOVE_NONE, score_to_tt(v, ply), VALUE_NONE, (MAX_PLY - 1).min(depth + 6), b, tt_pv);
                    return v;
                }
                if PV {
                    if b == BOUND_LOWER {
                        best_score = v;
                        alpha = alpha.max(v);
                    } else {
                        max_score = v;
                    }
                }
            }
        }

        let mut raw_eval = VALUE_NONE;
        let eval;
        let mut improving = false;
        {
            let child = &mut self.stack[si + 1];
            child.killer = [MOVE_NONE; 2];
            child.excluded = MOVE_NONE;
            child.cutoff_cnt = 0;
        }
        if in_check {
            self.stack[si].eval = VALUE_NONE;
            eval = VALUE_NONE;
        } else if excluded != 0 {
            eval = self.stack[si].eval;
            raw_eval = eval;
        } else {
            raw_eval = if hit && te.eval() != VALUE_NONE { te.eval() } else { self.evaluate() };
            let e = self.corrected_eval(raw_eval, si);
            self.stack[si].eval = e;
            eval = if hit
                && tt_score != VALUE_NONE
                && ((tt_bound == BOUND_LOWER && tt_score > e) || (tt_bound == BOUND_UPPER && tt_score < e))
            {
                tt_score
            } else {
                e
            };
            if !hit {
                TT.store(te, key, MOVE_NONE, VALUE_NONE, raw_eval, -1, BOUND_NONE, tt_pv);
            }
        }
        let ss_eval = self.stack[si].eval;
        if !in_check {
            improving = if self.stack[si - 2].eval != VALUE_NONE {
                ss_eval > self.stack[si - 2].eval
            } else if self.stack[si - 4].eval != VALUE_NONE {
                ss_eval > self.stack[si - 4].eval
            } else {
                true
            };
            // opponent's quiet move: reward/punish by static eval swing
            let (pm, prev_eval) = (self.stack[si - 1].mv, self.stack[si - 1].eval);
            if pm != 0
                && pm != MOVE_NULL
                && prev_eval != VALUE_NONE
                && excluded == 0
                && self.pos.s().captured == NO_PIECE
                && move_type(pm) != MT_PROMO
            {
                let bonus = (-10 * (prev_eval + ss_eval)).clamp(-1000, 1000);
                hist_update(&mut self.h.main_hist[self.pos.stm ^ 1][(pm & 4095) as usize], bonus);
            }
        }

        if !PV && !in_check && excluded == 0 {
            // reverse futility
            if depth <= 8 && eval - 80 * (depth - improving as i32) >= beta && eval < TB_WIN_BOUND {
                return (eval + beta) / 2;
            }
            // razoring
            if depth <= 3 && eval + 250 * depth < alpha {
                let s = self.qsearch::<false>(alpha, alpha + 1, ply);
                if s <= alpha {
                    return s;
                }
            }
            // null move
            if depth >= 3
                && eval >= beta
                && self.stack[si].eval >= beta - 20 * depth + 160
                && self.stack[si - 1].mv != MOVE_NULL
                && self.pos.has_non_pawn(self.pos.stm)
                && beta > -TB_WIN_BOUND
            {
                let r = 5 + depth / 3 + ((eval - beta) / 200).min(3);
                let ss = &mut self.stack[si];
                ss.mv = MOVE_NULL;
                ss.piece = 0;
                ss.ch = CH_NONE;
                self.pos.make_null();
                self.inc_nodes();
                let s = -self.search::<false>(-beta, -beta + 1, depth - r, ply + 1, !cutnode);
                self.pos.unmake_null();
                if Self::stopped() {
                    return 0;
                }
                if s >= beta {
                    return if s >= TB_WIN_BOUND { beta } else { s };
                }
            }
            // probcut
            let pc_beta = beta + 200;
            if depth >= 5
                && beta.abs() < TB_WIN_BOUND
                && !(hit && tt_depth >= depth - 3 && tt_score != VALUE_NONE && tt_score < pc_beta)
            {
                let tm = if tt_capture { tt_move } else { MOVE_NONE };
                let mut mp = Picker::new(&self.pos, &self.h, &self.stack, si, tm, true);
                loop {
                    let m = mp.next(&self.pos, &self.h);
                    if m == MOVE_NONE {
                        break;
                    }
                    if !self.pos.is_legal(m) || !self.pos.see_ge(m, pc_beta - self.stack[si].eval) {
                        continue;
                    }
                    self.do_move(m, si);
                    let mut s = -self.qsearch::<false>(-pc_beta, -pc_beta + 1, ply + 1);
                    if s >= pc_beta {
                        s = -self.search::<false>(-pc_beta, -pc_beta + 1, depth - 4, ply + 1, !cutnode);
                    }
                    self.undo_move(m);
                    if Self::stopped() {
                        return 0;
                    }
                    if s >= pc_beta {
                        TT.store(te, key, m, score_to_tt(s, ply), raw_eval, depth - 3, BOUND_LOWER, tt_pv);
                        return s;
                    }
                }
            }
        }

        // internal iterative reduction
        if (PV || cutnode) && depth >= 4 && tt_move == 0 {
            depth -= 1;
        }

        let mut mp = Picker::new(&self.pos, &self.h, &self.stack, si, tt_move, false);
        let mut best_move = MOVE_NONE;
        let mut move_count = 0;
        let mut quiets = [MOVE_NONE; 64];
        let mut caps = [MOVE_NONE; 32];
        let (mut nq, mut nc) = (0, 0);
        let mut nodes_before = 0;

        loop {
            let m = mp.next(&self.pos, &self.h);
            if m == MOVE_NONE {
                break;
            }
            if m == excluded {
                continue;
            }
            if root && !self.is_root_move(m) {
                continue;
            }
            if !self.pos.is_legal(m) {
                continue;
            }
            move_count += 1;
            let quiet = !self.pos.is_tactical(m);
            let (from, to) = (move_from(m), move_to(m));
            let pc = self.pos.piece_on(from);
            let hist = if quiet {
                self.h.main_hist[self.pos.stm][(m & 4095) as usize] as i32
                    + self.h.cont_hist[self.stack[si - 1].ch][pc][to] as i32
                    + self.h.cont_hist[self.stack[si - 2].ch][pc][to] as i32
            } else {
                let victim = self.pos.piece_on(to);
                self.h.cap_hist[pc][to][if victim == NO_PIECE { 0 } else { piece_type(victim) }] as i32
            };

            if !root && best_score > -TB_WIN_BOUND && self.pos.has_non_pawn(self.pos.stm) {
                let lmr_depth = (depth - 1 - lmr(depth, move_count)).max(0);
                if quiet {
                    if move_count >= (3 + depth * depth) / (2 - improving as i32) {
                        mp.skip_quiets = true;
                    }
                    if !in_check && lmr_depth <= 8 && self.stack[si].eval + 100 + 110 * lmr_depth <= alpha {
                        mp.skip_quiets = true;
                        continue;
                    }
                    if lmr_depth <= 4 && hist < -3000 * depth {
                        continue;
                    }
                    if !self.pos.see_ge(m, -30 * lmr_depth * lmr_depth) {
                        continue;
                    }
                } else {
                    if !in_check && lmr_depth < 7 && move_type(m) != MT_PROMO {
                        let victim = if move_type(m) == MT_EP { PAWN } else { piece_type(self.pos.piece_on(to)) };
                        if self.stack[si].eval + 250 + 180 * lmr_depth + PIECE_VAL[victim] + hist / 8 <= alpha {
                            continue;
                        }
                    }
                    if !self.pos.see_ge(m, -100 * depth) {
                        continue;
                    }
                }
            }

            // singular extension
            let mut ext = 0;
            if !root
                && depth >= 6
                && m == tt_move
                && excluded == 0
                && tt_depth >= depth - 3
                && (tt_bound & BOUND_LOWER) != 0
                && tt_score.abs() < TB_WIN_BOUND
                && ply < 2 * self.root_depth
            {
                let s_beta = tt_score - depth;
                self.stack[si].excluded = m;
                let s = self.search::<false>(s_beta - 1, s_beta, (depth - 1) / 2, ply, cutnode);
                self.stack[si].excluded = MOVE_NONE;
                if Self::stopped() {
                    return 0;
                }
                if s < s_beta {
                    ext = 1;
                    if !PV && s < s_beta - 20 && self.stack[si].double_ext <= 6 {
                        ext = 2;
                    }
                } else if s_beta >= beta {
                    return s_beta;
                } else if tt_score >= beta || cutnode {
                    ext = -1;
                }
            }
            self.stack[si + 1].double_ext = self.stack[si].double_ext + (ext >= 2) as i32;

            if root {
                nodes_before = self.nodes();
            }
            self.do_move(m, si);
            let mut new_depth = depth - 1 + ext;
            let mut score = 0;
            // late move reductions
            if depth >= 2 && move_count > 1 + root as i32 && (quiet || !PV) {
                let mut r = lmr(depth, move_count);
                r += !PV as i32;
                r += cutnode as i32;
                r += !improving as i32;
                r += (tt_capture && quiet) as i32;
                r -= tt_pv as i32;
                r += (self.stack[si + 1].cutoff_cnt > 3) as i32;
                r -= self.pos.in_check() as i32;
                r -= hist / 8000;
                r -= (m == mp.k1 || m == mp.k2 || m == mp.cm) as i32;
                if !quiet {
                    r = (r - 1).max(0);
                }
                let d = (new_depth - r).clamp(1, new_depth + 1);
                score = -self.search::<false>(-alpha - 1, -alpha, d, ply + 1, true);
                if score > alpha && d < new_depth {
                    let deeper = score > best_score + 50 + 2 * new_depth;
                    let shallower = score < best_score + 8;
                    new_depth += deeper as i32 - shallower as i32;
                    if d < new_depth {
                        score = -self.search::<false>(-alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
                    }
                }
            } else if !PV || move_count > 1 {
                score = -self.search::<false>(-alpha - 1, -alpha, new_depth, ply + 1, !cutnode);
            }
            if PV && (move_count == 1 || score > alpha) {
                score = -self.search::<true>(-beta, -alpha, new_depth, ply + 1, false);
            }
            self.undo_move(m);
            if Self::stopped() {
                return 0;
            }
            if root {
                self.root_move_nodes[(m & 4095) as usize] += self.nodes() - nodes_before;
            }

            if score > best_score {
                best_score = score;
                if score > alpha {
                    best_move = m;
                    if PV {
                        self.update_pv(si, m);
                    }
                    if score >= beta {
                        self.stack[si].cutoff_cnt += 1;
                        break;
                    }
                    alpha = score;
                }
            }
            if m != best_move {
                if quiet && nq < 64 {
                    quiets[nq] = m;
                    nq += 1;
                } else if !quiet && nc < 32 {
                    caps[nc] = m;
                    nc += 1;
                }
            }
        }

        if move_count == 0 {
            return if excluded != 0 {
                alpha
            } else if in_check {
                -MATE + ply
            } else {
                0
            };
        }

        if best_score >= beta && best_move != 0 {
            let bonus = stat_bonus(depth);
            let malus = stat_bonus(depth);
            let stm = self.pos.stm;
            if !self.pos.is_tactical(best_move) {
                let ss = &mut self.stack[si];
                if ss.killer[0] != best_move {
                    ss.killer[1] = ss.killer[0];
                    ss.killer[0] = best_move;
                }
                let (pm, ppiece) = (self.stack[si - 1].mv, self.stack[si - 1].piece);
                if pm != 0 && pm != MOVE_NULL {
                    self.h.counter_move[ppiece][move_to(pm)] = best_move;
                }
                let chs = [self.stack[si - 1].ch, self.stack[si - 2].ch, self.stack[si - 4].ch];
                let pawn_idx = self.pos.s().pawn_key as usize & (PH_SIZE - 1);
                let pos = &self.pos;
                let h = &mut self.h;
                let mut upd = |mv: Move, b: i32| {
                    let (pc, to) = (pos.piece_on(move_from(mv)), move_to(mv));
                    hist_update(&mut h.main_hist[stm][(mv & 4095) as usize], b);
                    hist_update(&mut h.pawn_hist[pawn_idx][pc][to], b);
                    if chs[0] != CH_NONE {
                        hist_update(&mut h.cont_hist[chs[0]][pc][to], b);
                    }
                    if chs[1] != CH_NONE {
                        hist_update(&mut h.cont_hist[chs[1]][pc][to], b);
                    }
                    if chs[2] != CH_NONE {
                        hist_update(&mut h.cont_hist[chs[2]][pc][to], b / 2);
                    }
                };
                upd(best_move, bonus);
                for &q in &quiets[..nq] {
                    upd(q, -malus);
                }
            } else {
                let victim = self.pos.piece_on(move_to(best_move));
                let vt = if victim == NO_PIECE { 0 } else { piece_type(victim) };
                hist_update(&mut self.h.cap_hist[self.pos.piece_on(move_from(best_move))][move_to(best_move)][vt], bonus);
            }
            for &c in &caps[..nc] {
                let victim = self.pos.piece_on(move_to(c));
                let vt = if victim == NO_PIECE { 0 } else { piece_type(victim) };
                hist_update(&mut self.h.cap_hist[self.pos.piece_on(move_from(c))][move_to(c)][vt], -malus);
            }
        }

        if PV {
            best_score = best_score.min(max_score);
        }

        if excluded == 0 {
            let ss_eval = self.stack[si].eval;
            let bound = if best_score >= beta {
                BOUND_LOWER
            } else if PV && best_move != 0 {
                BOUND_EXACT
            } else {
                BOUND_UPPER
            };
            if !in_check
                && (best_move == 0 || !self.pos.is_tactical(best_move))
                && !(bound == BOUND_LOWER && best_score <= ss_eval)
                && !(bound == BOUND_UPPER && best_score >= ss_eval)
            {
                self.update_corr(best_score - ss_eval, depth, si);
            }
            TT.store(te, key, best_move, score_to_tt(best_score, ply), raw_eval, depth, bound, tt_pv);
        }
        best_score
    }

    /// Iterative deepening. Thread 0 owns time management and output; helpers just search and
    /// share what they find through the TT.
    pub fn iterate(&mut self, lim: &Limits, silent: bool) {
        let main_thread = self.id == 0;
        for ss in self.stack.iter_mut() {
            ss.killer = [MOVE_NONE; 2];
            ss.excluded = MOVE_NONE;
            ss.eval = VALUE_NONE;
            ss.mv = MOVE_NONE;
            ss.ch = CH_NONE;
            ss.double_ext = 0;
            ss.pv_len = 0;
        }
        self.root_move_nodes.fill(0);
        self.ev.reset(&self.pos);
        self.ctr.nodes.store(0, Relaxed);
        self.ctr.tb_hits.store(0, Relaxed);
        self.completed_depth = 0;
        self.best_score = -INF;
        self.pv_len = 0;
        self.best_move = self.root_moves.as_slice().first().copied().unwrap_or(MOVE_NONE);
        if self.best_move == 0 {
            return;
        }

        const RS: usize = 4;
        let mut prev_score = 0;
        let mut score;
        let max_depth = if lim.depth > 0 { lim.depth.min(MAX_PLY - 1) } else { MAX_PLY - 1 };
        let mut stable = 0;
        let mut last_scores = [0i32; MAX_PLY as usize];

        self.root_depth = 1;
        while self.root_depth <= max_depth {
            let rd = self.root_depth;
            self.seldepth = 0;
            let mut delta = 12;
            let (mut alpha, mut beta) = (-INF, INF);
            if rd >= 4 {
                alpha = (prev_score - delta).max(-INF);
                beta = (prev_score + delta).min(INF);
            }
            let mut fail_high_reduce = 0;
            loop {
                let d = (rd - fail_high_reduce).max(1);
                score = self.search::<true>(alpha, beta, d, 0, false);
                if Self::stopped() {
                    break;
                }
                if score <= alpha {
                    beta = (alpha + beta) / 2;
                    alpha = (score - delta).max(-INF);
                    fail_high_reduce = 0;
                } else if score >= beta {
                    beta = (score + delta).min(INF);
                    if score.abs() < TB_WIN_BOUND {
                        fail_high_reduce = (fail_high_reduce + 1).min(3);
                    }
                } else {
                    break;
                }
                delta += delta / 2;
                if delta > 1000 {
                    alpha = -INF;
                    beta = INF;
                }
            }
            if Self::stopped() && rd > 1 {
                break;
            }
            // completed iteration
            let rs = &self.stack[RS];
            let nb = if rs.pv_len > 0 { rs.pv[0] } else { self.best_move };
            if nb == self.best_move {
                stable += 1;
            } else {
                stable = 0;
            }
            self.best_move = nb;
            self.pv_len = rs.pv_len;
            self.pv[..rs.pv_len].copy_from_slice(&rs.pv[..rs.pv_len]);
            self.best_score = score;
            self.completed_depth = rd;
            last_scores[rd as usize] = score;
            prev_score = score;
            if !main_thread {
                if Self::stopped() {
                    break;
                }
                self.root_depth += 1;
                continue;
            }
            let el = self.time.elapsed();
            if !silent {
                print_info(self, rd, self.seldepth, score);
            }
            if Self::stopped() {
                break;
            }
            let mut done = false;
            if score.abs() >= MATE_BOUND && rd > 10 && self.time.timed && MATE - score.abs() < rd {
                done = true;
            } else if self.time.timed {
                if self.root_moves.len() == 1 && rd >= 4 {
                    done = true;
                } else {
                    let stab_f = match stable {
                        s if s >= 6 => 0.65,
                        s if s >= 4 => 0.8,
                        s if s >= 2 => 0.95,
                        s if s >= 1 => 1.15,
                        _ => 1.4,
                    };
                    let mut score_f = 1.0;
                    if rd >= 4 {
                        let drop = last_scores[rd as usize - 3] - score;
                        score_f = (1.0 + drop as f64 / 120.0).clamp(0.85, 1.6);
                    }
                    let n = self.nodes();
                    let frac = if n > 0 {
                        self.root_move_nodes[(self.best_move & 4095) as usize] as f64 / n as f64
                    } else {
                        0.5
                    };
                    let node_f = ((1.6 - frac) * 1.1).clamp(0.6, 1.8);
                    let opt = self.time.soft_limit as f64 * stab_f * score_f * node_f;
                    if el as f64 >= opt.min(self.time.hard_limit as f64 * 0.9) {
                        done = true;
                    }
                }
            }
            if done {
                // while pondering, keep searching; the answer goes out as soon as ponderhit arrives
                if !PONDERING.load(Relaxed) {
                    break;
                }
                STOP_ON_PONDERHIT.store(true, Relaxed);
            }
            self.root_depth += 1;
        }
    }
}

/// One "info depth ..." line for w's last completed iteration.
pub fn print_info(w: &Worker, depth: i32, seldepth: i32, score: i32) {
    let el = w.time.elapsed();
    let n = uci::total_nodes();
    let score = if w.tb.root_in_tb && score.abs() < MATE_BOUND { w.tb.root_tb_score } else { score };
    let mut s = format!(
        "info depth {depth} seldepth {seldepth} score {} nodes {n} nps {} time {el} hashfull {} tbhits {} pv",
        uci::uci_score(score),
        n as i64 * 1000 / el.max(1),
        TT.hashfull(),
        uci::total_tbhits(),
    );
    for &m in &w.pv[..w.pv_len] {
        s.push(' ');
        s += &uci::move_to_uci(m);
    }
    uci::uci_out(&s);
}
