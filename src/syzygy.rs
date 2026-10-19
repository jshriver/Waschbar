//! Syzygy tablebases via `shakmaty-syzygy`: the DTZ root filter (ported from erebus) and
//! in-search WDL probes.

use std::num::NonZeroU32;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering::Relaxed};
use std::sync::{Arc, Mutex};

use shakmaty::{
    Bitboard as SmBB, Board as SmBoard, ByColor, ByRole, CastlingMode, Chess, Color as SmColor, FromSetup,
    PositionError, Setup, Square as SmSquare,
};
use shakmaty_syzygy::{MaybeRounded, Tablebase, Wdl};

use crate::bitboard::*;
use crate::board::*;
use crate::eval::*;
use crate::moves::*;

// UCI options
pub static TB_PROBE_DEPTH: AtomicI32 = AtomicI32::new(1);
pub static TB_PROBE_LIMIT: AtomicI32 = AtomicI32::new(7);
pub static TB_50_MOVE_RULE: AtomicBool = AtomicBool::new(true);

pub type Tables = Tablebase<Chess>;

static TABLES: Mutex<Option<Arc<Tables>>> = Mutex::new(None);

/// What one search needs from the tablebases, set up by `rank_root`.
#[derive(Clone, Default)]
pub struct TbSearch {
    pub tables: Option<Arc<Tables>>,
    /// In-search probing limit in pieces (0 = off).
    pub card: i32,
    pub probe_depth: i32,
    pub rule50: bool,
    /// Root moves were filtered by the tablebases.
    pub root_in_tb: bool,
    /// Score to report while `root_in_tb`.
    pub root_tb_score: i32,
}

#[derive(Default)]
pub struct TbInfo {
    /// .rtbw / .rtbz files found
    pub wdl: usize,
    pub dtz: usize,
    /// Largest piece count covered (0 = none).
    pub pieces: usize,
}

/// (Re)loads tablebases from a path list (':' separated, ';' on Windows). "<empty>" or "" unloads.
pub fn set_path(path: &str) -> TbInfo {
    let mut info = TbInfo::default();
    let mut slot = TABLES.lock().unwrap();
    *slot = None;
    if path.is_empty() || path == "<empty>" {
        return info;
    }
    let sep = if cfg!(windows) { ';' } else { ':' };
    let mut tb = Tables::new();
    for dir in path.split(sep).filter(|d| !d.is_empty()) {
        if tb.add_directory(Path::new(dir)).is_err() {
            continue;
        }
        if let Ok(rd) = std::fs::read_dir(dir) {
            for e in rd.flatten() {
                match e.path().extension().and_then(|x| x.to_str()) {
                    Some("rtbw") => info.wdl += 1,
                    Some("rtbz") => info.dtz += 1,
                    _ => {}
                }
            }
        }
    }
    info.pieces = tb.max_pieces();
    if info.pieces > 0 {
        *slot = Some(Arc::new(tb));
    }
    info
}

fn loaded() -> Option<Arc<Tables>> {
    TABLES.lock().unwrap().clone()
}

/// Converts the engine position (without castling rights) into a shakmaty position.
#[allow(clippy::result_large_err)]
fn to_chess(p: &Position) -> Option<Chess> {
    let bb = |b: u64| SmBB(b);
    let by_role = ByRole {
        pawn: bb(p.by_type[PAWN]),
        knight: bb(p.by_type[KNIGHT]),
        bishop: bb(p.by_type[BISHOP]),
        rook: bb(p.by_type[ROOK]),
        queen: bb(p.by_type[QUEEN]),
        king: bb(p.by_type[KING]),
    };
    let by_color = ByColor { white: bb(p.by_color[WHITE]), black: bb(p.by_color[BLACK]) };
    let board = SmBoard::try_from_bitboards(by_role, by_color).ok()?;
    let ep = p.s().ep;
    let setup = Setup {
        board,
        turn: if p.stm == WHITE { SmColor::White } else { SmColor::Black },
        castling_rights: SmBB::EMPTY,
        ep_square: if ep == NO_SQ { None } else { SmSquare::try_from(ep as u32).ok() },
        halfmoves: p.s().halfmove.max(0) as u32,
        fullmoves: NonZeroU32::MIN,
        promoted: SmBB::EMPTY,
        pockets: None,
        remaining_checks: None,
    };
    Chess::from_setup(setup, CastlingMode::Chess960)
        .or_else(|e: PositionError<Chess>| e.ignore_invalid_ep_square())
        .ok()
}

fn wdl_value(w: Wdl) -> i32 {
    match w {
        Wdl::Loss => -2,
        Wdl::BlessedLoss => -1,
        Wdl::Draw => 0,
        Wdl::CursedWin => 1,
        Wdl::Win => 2,
    }
}

/// WDL probe (-2 loss .. +2 win, +-1 cursed/blessed) of a position reached by a zeroing move.
/// Needs rule50 == 0 and no castling rights.
pub fn probe_wdl(tables: &Tables, p: &Position) -> Option<i32> {
    let pos = to_chess(p)?;
    tables.probe_wdl_after_zeroing(&pos).ok().map(wdl_value)
}

fn probe_dtz(tables: &Tables, p: &Position) -> Option<i32> {
    let pos = to_chess(p)?;
    match tables.probe_dtz(&pos).ok()? {
        MaybeRounded::Precise(d) | MaybeRounded::Rounded(d) => Some(d.0),
    }
}

/// The position occurred earlier in the game or tree (same side to move, within the
/// reversible stretch).
fn seen_before(p: &Position) -> bool {
    let x = p.s();
    let lookback = (x.halfmove.min(x.plies_from_null).max(0) as usize).min(p.sp);
    (2..=lookback).step_by(2).any(|k| p.st[p.sp - k].key == x.key)
}

/// Syzygy root filter (ported from erebus): when the root is in the tablebases, keep the
/// root moves worth searching, best first:
///
/// * winning: the moves that zero the 50-move clock (pawn move, capture) and keep the win,
///   else those leaving the opponent the shortest DTZ, ties broken by the shortest DTZ after
///   the move. This is DTZ-optimal play: it always makes progress and, in pawnless endings
///   where DTZ counts to mate, mates fast. Many tables store DTZ rounded, so a tied move
///   that recreates an earlier position loses the tie (no threefold shuffles, e.g. KBNK).
///   A win the 50-move clock would cut off counts as a cursed win;
/// * drawing: every move that keeps the draw, for the search to choose among (a cursed win,
///   which might still win, first);
/// * losing: the moves that hold out longest.
///
/// With `Syzygy50MoveRule` off the clock is ignored: cursed wins rank as wins and blessed
/// losses as losses. Returns the moves (empty if the root is not covered or a probe fails)
/// and the score to report.
fn root_filter(t: &Tables, p: &mut Position, legal: &MoveList, rule50: bool) -> Option<(Vec<Move>, i32)> {
    probe_dtz(t, p)?;
    // Larger key = better for us: a category band, then progress within it.
    const MATE: i64 = 4_000_000;
    const WIN: i64 = 3_000_000;
    const CURSED_WIN: i64 = 2_000_000;
    const DRAW: i64 = 1_000_000;
    const BLESSED_LOSS: i64 = 500_000;
    let mut ranked: Vec<(i64, Move)> = Vec::with_capacity(legal.len());
    for &m in legal.as_slice() {
        let zeroing = p.is_capture(m) || piece_type(p.piece_on(move_from(m))) == PAWN;
        let clock = if zeroing { 0 } else { p.s().halfmove as i64 + 1 };
        p.make_move(m);
        let repeats = seen_before(p) as i64;
        // The opponent's DTZ: negative = they lose. Mate is a win at once; stalemate a draw.
        let mut replies = MoveList::new();
        p.generate(GenType::All, &mut replies);
        let has_reply = replies.as_slice().iter().any(|&r| p.is_legal(r));
        let dtz = if has_reply { probe_dtz(t, p) } else { Some(if p.in_check() { -1 } else { 0 }) };
        let mated = !has_reply && p.in_check();
        p.unmake_move(m);
        let dtz = dtz?;
        let d = dtz.unsigned_abs() as i64;
        let key = if dtz < 0 && (mated || !rule50 || d + clock <= 100) {
            // win: mate, then zeroing moves, then shortest DTZ
            if mated { MATE } else { WIN - if zeroing { 0 } else { 2000 } - 2 * d - repeats }
        } else if dtz < 0 {
            CURSED_WIN
        } else if dtz == 0 {
            DRAW
        } else if rule50 && d + clock > 100 {
            BLESSED_LOSS
        } else {
            d // loss: longest resistance
        };
        ranked.push((key, m));
    }
    let best = ranked.iter().map(|r| r.0).max()?;
    // A win keeps only the best moves; a draw every move that does not lose (a cursed win
    // first in the order); a loss the longest.
    let keep = |key: i64| if (DRAW..WIN / 2).contains(&best) { key >= DRAW } else { key == best };
    // stable sort: equal keys keep move-generation order
    ranked.sort_by_key(|r| std::cmp::Reverse(r.0));
    let moves = ranked.into_iter().filter(|r| keep(r.0)).map(|r| r.1).collect();
    let score = if best >= WIN / 2 {
        TB_WIN_SCORE
    } else if best >= CURSED_WIN {
        1
    } else if best >= DRAW {
        0
    } else if best >= BLESSED_LOSS {
        -1
    } else {
        -TB_WIN_SCORE
    };
    Some((moves, score))
}

/// Filters the root moves with the tablebases (see `root_filter`) and returns the
/// tablebase state for this search.
pub fn rank_root(p: &mut Position, legal: &mut MoveList) -> TbSearch {
    let tables = loaded();
    let largest = tables.as_ref().map_or(0, |t| t.max_pieces() as i32);
    let rule50 = TB_50_MOVE_RULE.load(Relaxed);
    let mut tb = TbSearch {
        card: if largest > 0 { largest.min(TB_PROBE_LIMIT.load(Relaxed)) } else { 0 },
        probe_depth: TB_PROBE_DEPTH.load(Relaxed),
        rule50,
        tables,
        ..Default::default()
    };
    let Some(t) = tb.tables.clone() else { return tb };
    if tb.card == 0 || popcount(p.occ()) > tb.card || p.castling() != 0 || legal.is_empty() {
        return tb;
    }
    let Some((moves, score)) = root_filter(&t, p, legal, rule50) else { return tb };
    legal.clear();
    for m in moves {
        legal.add(m);
    }
    tb.root_in_tb = true;
    tb.root_tb_score = score;
    tb
}
