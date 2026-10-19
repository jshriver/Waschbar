//! UCI protocol: command loop, options, move and score notation, time management, and the
//! Lazy SMP thread pool (go / stop / ponderhit).

use std::io::{BufRead, Write};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::bitboard::*;
use crate::board::*;
use crate::eval::*;
use crate::movegen::perft;
use crate::moves::*;
use crate::nnue;
use crate::search::{Counters, Worker, print_info};
use crate::syzygy;
use crate::tt::TT;

const ENGINE_NAME: &str = concat!("Waschbar ", env!("CARGO_PKG_VERSION"));
const ENGINE_AUTHOR: &str = "Joshua Shriver";

/// Thread-safe line output.
pub fn uci_out(s: &str) {
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{s}");
    let _ = out.flush();
}

/// Honours UCI_Chess960.
pub fn move_to_uci(m: Move) -> String {
    move_to_uci_mode(m, chess960())
}

pub fn move_to_uci_mode(m: Move, c960: bool) -> String {
    if m == MOVE_NONE {
        return "0000".to_string();
    }
    let from = move_from(m);
    let mut to = move_to(m);
    if move_type(m) == MT_CASTLE && !c960 {
        // standard notation: the king's destination
        to = rank_of(from) * 8 + if to > from { 6 } else { 2 };
    }
    let mut s = String::with_capacity(5);
    s.push((b'a' + file_of(from) as u8) as char);
    s.push((b'1' + rank_of(from) as u8) as char);
    s.push((b'a' + file_of(to) as u8) as char);
    s.push((b'1' + rank_of(to) as u8) as char);
    if move_type(m) == MT_PROMO {
        s.push(b"nbrq"[move_promo(m) - KNIGHT] as char);
    }
    s
}

pub fn parse_uci_move(pos: &Position, s: &str) -> Move {
    let mut ml = MoveList::new();
    pos.generate(GenType::All, &mut ml);
    let c960 = chess960();
    for &m in ml.as_slice() {
        if move_to_uci_mode(m, c960) == s && pos.is_legal(m) {
            return m;
        }
    }
    // also accept the other castling notation (king-takes-rook vs. king destination)
    for &m in ml.as_slice() {
        if move_type(m) == MT_CASTLE && move_to_uci_mode(m, !c960) == s && pos.is_legal(m) {
            return m;
        }
    }
    MOVE_NONE
}

/// "cp N" / "mate N"
pub fn uci_score(s: i32) -> String {
    if s >= MATE_BOUND {
        format!("mate {}", (MATE - s + 1) / 2)
    } else if s <= -MATE_BOUND {
        format!("mate {}", -(MATE + s) / 2)
    } else if s >= TB_WIN_BOUND {
        format!("cp {}", 20000 - (TB_WIN_SCORE - s))
    } else if s <= -TB_WIN_BOUND {
        format!("cp {}", -20000 + (TB_WIN_SCORE + s))
    } else {
        format!("cp {s}")
    }
}

fn set_position(root: &mut Position, toks: &[&str]) {
    let mut it = toks.iter().copied();
    let fen = match it.next() {
        Some("startpos") => {
            it.next(); // maybe "moves"
            START_FEN.to_string()
        }
        Some("fen") => it.by_ref().take_while(|&t| t != "moves").collect::<Vec<_>>().join(" "),
        _ => return,
    };
    if !root.set_fen(&fen) {
        root.set_fen(START_FEN);
        return;
    }
    for tok in it {
        let m = parse_uci_move(root, tok);
        if m == 0 {
            break;
        }
        if root.sp >= MAX_STATES - MAX_PLY as usize - 8 {
            // compact history: keep only the last 100 plies (enough for repetition / 50-move rule)
            root.compact_history(100);
        }
        root.make_move(m);
    }
}

fn parse_go(toks: &[&str]) -> Limits {
    let mut lim = Limits::default();
    let mut it = toks.iter().copied();
    while let Some(tok) = it.next() {
        let mut num = || it.next().and_then(|v| v.parse::<i64>().ok());
        match tok {
            "wtime" => lim.wtime = num().unwrap_or(lim.wtime),
            "btime" => lim.btime = num().unwrap_or(lim.btime),
            "winc" => lim.winc = num().unwrap_or(lim.winc),
            "binc" => lim.binc = num().unwrap_or(lim.binc),
            "movetime" => lim.movetime = num().unwrap_or(lim.movetime),
            "movestogo" => lim.movestogo = num().map_or(lim.movestogo, |v| v as i32),
            "depth" => lim.depth = num().map_or(lim.depth, |v| v as i32),
            "infinite" => lim.infinite = true,
            "ponder" => lim.ponder = true,
            _ => {}
        }
    }
    lim
}

fn set_option(pool: &mut Pool, toks: &[&str]) {
    let (mut name, mut value) = (Vec::new(), Vec::new());
    let mut target = None;
    for &t in toks {
        match t {
            "name" => target = Some(0),
            "value" => target = Some(1),
            _ => match target {
                Some(0) => name.push(t),
                Some(1) => value.push(t),
                _ => {}
            },
        }
    }
    let name = name.join(" ").to_lowercase();
    let value = value.join(" ");
    let lval = value.to_lowercase();
    let int = || value.trim().parse::<i64>().ok();
    match name.as_str() {
        "hash" => {
            if let Some(mb) = int() {
                TT.resize(mb.clamp(1, 65536) as usize);
            }
        }
        "move overhead" => {
            if let Some(v) = int() {
                MOVE_OVERHEAD.store(v.clamp(0, 5000), Relaxed);
            }
        }
        "threads" => {
            if let Some(v) = int() {
                pool.set_threads(v.clamp(1, 1024) as usize);
            }
        }
        "uci_chess960" => CHESS960.store(lval == "true", Relaxed),
        "syzygypath" => {
            let tb = syzygy::set_path(&value);
            if value.is_empty() || value == "<empty>" {
                uci_out("info string Syzygy: tablebases unloaded");
            } else if tb.pieces == 0 {
                uci_out(&format!("info string Syzygy: no tablebase files found in {value}"));
            } else {
                uci_out(&format!(
                    "info string Syzygy: found {} WDL and {} DTZ files, up to {}-piece tablebases",
                    tb.wdl, tb.dtz, tb.pieces
                ));
            }
        }
        "syzygyprobedepth" => {
            if let Some(v) = int() {
                syzygy::TB_PROBE_DEPTH.store(v.clamp(1, 100) as i32, Relaxed);
            }
        }
        "syzygy50moverule" => syzygy::TB_50_MOVE_RULE.store(lval == "true", Relaxed),
        "syzygyprobelimit" => {
            if let Some(v) = int() {
                syzygy::TB_PROBE_LIMIT.store(v.clamp(0, 7) as i32, Relaxed);
            }
        }
        _ => {}
    }
}

pub fn uci_loop(pool: &mut Pool) {
    let mut root = Position::startpos();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let toks: Vec<&str> = line.split_whitespace().collect();
        let Some((&cmd, args)) = toks.split_first() else { continue };
        match cmd {
            "uci" => {
                uci_out(&format!("id name {ENGINE_NAME}"));
                uci_out(&format!("id author {ENGINE_AUTHOR}"));
                uci_out("option name Hash type spin default 256 min 1 max 65536");
                uci_out("option name Threads type spin default 1 min 1 max 1024");
                uci_out("option name Move Overhead type spin default 30 min 0 max 5000");
                uci_out("option name Ponder type check default false");
                uci_out("option name UCI_Chess960 type check default false");
                uci_out("option name SyzygyPath type string default <empty>");
                uci_out("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
                uci_out("option name Syzygy50MoveRule type check default true");
                uci_out("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
                uci_out("uciok");
            }
            "isready" => {
                if !pool.running() {
                    TT.ensure();
                }
                uci_out("readyok");
            }
            "ucinewgame" => {
                pool.wait();
                TT.ensure();
                pool.clear();
            }
            "setoption" => {
                pool.wait();
                set_option(pool, args);
            }
            "position" => {
                pool.wait();
                set_position(&mut root, args);
            }
            "go" => {
                pool.wait();
                let t0 = now_ms();
                let lim = parse_go(args);
                pool.start_thinking(&root, lim, t0);
            }
            "stop" => pool.stop(),
            "ponderhit" => pool.ponderhit(),
            "quit" => {
                pool.stop();
                return;
            }
            "perft" => {
                pool.wait();
                let d = args.first().and_then(|v| v.parse().ok()).unwrap_or(1);
                let t0 = now_ms();
                let n = perft(&mut root, d);
                uci_out(&format!("nodes {n} time {}", now_ms() - t0));
            }
            "bench" => {
                pool.wait();
                let d = args.first().and_then(|v| v.parse().ok()).unwrap_or(crate::bench::DEFAULT_DEPTH);
                crate::bench::bench(pool, d);
            }
            "d" => uci_out(&root.fen()),
            "eval" => uci_out(&format!("eval {}", nnue::evaluate_full(&root))),
            _ => {}
        }
    }
    pool.stop();
}

// ---------------------------------------------------------------- time management

#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub wtime: i64,
    pub btime: i64,
    pub winc: i64,
    pub binc: i64,
    pub movetime: i64,
    pub movestogo: i32,
    pub depth: i32,
    pub infinite: bool,
    pub ponder: bool,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            wtime: -1,
            btime: -1,
            winc: 0,
            binc: 0,
            movetime: -1,
            movestogo: 0,
            depth: 0,
            infinite: false,
            ponder: false,
        }
    }
}

#[derive(Clone, Copy, Default, Debug)]
pub struct TimeManager {
    pub start_time: i64,
    pub soft_limit: i64,
    pub hard_limit: i64,
    pub timed: bool,
}

/// UCI "Move Overhead" in ms.
pub static MOVE_OVERHEAD: AtomicI64 = AtomicI64::new(30);

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Monotonic milliseconds since the first call.
pub fn now_ms() -> i64 {
    EPOCH.get_or_init(Instant::now).elapsed().as_millis() as i64
}

impl TimeManager {
    pub fn new(lim: &Limits, stm: usize, t0: i64) -> TimeManager {
        let mut tm = TimeManager { start_time: t0, ..Default::default() };
        let overhead = MOVE_OVERHEAD.load(Ordering::Relaxed);
        let time = if stm == WHITE { lim.wtime } else { lim.btime };
        let inc = if stm == WHITE { lim.winc } else { lim.binc };
        if lim.movetime > 0 {
            tm.timed = true;
            tm.hard_limit = (lim.movetime - overhead).max(1);
            tm.soft_limit = tm.hard_limit * 10; // use the full movetime
        } else if time >= 0 && !lim.infinite {
            tm.timed = true;
            let avail = (time - overhead).max(1);
            let mtg = if lim.movestogo > 0 { lim.movestogo.min(40) } else { 22 };
            let base = avail as f64 / mtg as f64 + inc as f64 * 0.8;
            let cap = if lim.movestogo == 1 { avail as f64 * 0.85 } else { avail as f64 * 0.5 };
            tm.soft_limit = ((base * 0.7).min(cap) as i64).max(1);
            tm.hard_limit = ((base * 3.0).min(cap) as i64).max(1);
        }
        tm
    }

    pub fn elapsed(&self) -> i64 {
        now_ms() - self.start_time
    }
}

// ---------------------------------------------------------------- Lazy SMP thread pool

/// All threads stop as soon as they see it.
pub static STOP: AtomicBool = AtomicBool::new(false);
/// `go ponder` in progress: time limits are not enforced.
pub static PONDERING: AtomicBool = AtomicBool::new(false);
/// Time ran out while pondering: answer at ponderhit.
pub static STOP_ON_PONDERHIT: AtomicBool = AtomicBool::new(false);

/// Every worker's counters, for node / tbhit totals in `info` lines.
static COUNTERS: Mutex<Vec<Arc<Counters>>> = Mutex::new(Vec::new());

pub fn total_nodes() -> u64 {
    COUNTERS.lock().unwrap().iter().map(|c| c.nodes.load(Relaxed)).sum()
}
pub fn total_tbhits() -> u64 {
    COUNTERS.lock().unwrap().iter().map(|c| c.tb_hits.load(Relaxed)).sum()
}

/// Stack size for search threads (deep recursion with move lists on the stack).
pub const SEARCH_STACK: usize = 64 << 20;

pub type Workers = Arc<Mutex<Vec<Box<Worker>>>>;

pub struct Pool {
    pub workers: Workers,
    handle: Option<JoinHandle<()>>,
}

impl Pool {
    pub fn new(n: usize) -> Pool {
        let mut p = Pool { workers: Arc::new(Mutex::new(Vec::new())), handle: None };
        p.set_threads(n);
        p
    }

    /// Resizes the worker pool (not during a search).
    pub fn set_threads(&mut self, n: usize) {
        let n = n.max(1);
        let mut ws = self.workers.lock().unwrap();
        if n == ws.len() {
            return;
        }
        ws.clear();
        let mut ctrs = COUNTERS.lock().unwrap();
        ctrs.clear();
        for i in 0..n {
            let c = Arc::new(Counters::default());
            ctrs.push(c.clone());
            ws.push(Box::new(Worker::new(i, c)));
        }
    }

    /// ucinewgame
    pub fn clear(&mut self) {
        TT.clear();
        for w in self.workers.lock().unwrap().iter_mut() {
            w.clear();
        }
    }

    /// Starts searching `pos` on a background thread; it prints info and bestmove when done.
    pub fn start_thinking(&mut self, pos: &Position, lim: Limits, t0: i64) {
        self.wait();
        // set the flags here, not in the search thread: a ponderhit sent right after "go ponder"
        // must not be overwritten when the thread starts
        STOP.store(false, Relaxed);
        STOP_ON_PONDERHIT.store(false, Relaxed);
        PONDERING.store(lim.ponder, Relaxed);
        let workers = self.workers.clone();
        let pos = pos.clone();
        self.handle = Some(
            std::thread::Builder::new()
                .name("search".into())
                .stack_size(SEARCH_STACK)
                .spawn(move || {
                    let mut ws = workers.lock().unwrap();
                    search_main(&mut ws, &pos, &lim, t0);
                })
                .expect("spawn search thread"),
        );
    }

    pub fn wait(&mut self) {
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }

    pub fn stop(&mut self) {
        STOP.store(true, Relaxed);
        self.wait();
    }

    pub fn running(&self) -> bool {
        self.handle.is_some()
    }

    /// Switch a ponder search to a normal timed search.
    pub fn ponderhit(&self) {
        PONDERING.store(false, Relaxed);
        if STOP_ON_PONDERHIT.load(Relaxed) {
            STOP.store(true, Relaxed);
        }
    }
}

/// Lazy SMP thread pick: vote by score and depth, prefer proven results.
fn pick_best_worker(ws: &[Box<Worker>]) -> usize {
    let mut best = 0;
    if ws.len() == 1 {
        return best;
    }
    let min_score = ws.iter().filter(|w| w.completed_depth > 0).map(|w| w.best_score).min().unwrap_or(INF);
    let votes = |m: Move| -> i64 {
        ws.iter()
            .filter(|u| u.completed_depth > 0 && u.best_move == m)
            .map(|u| (u.best_score - min_score + 14) as i64 * u.completed_depth as i64)
            .sum()
    };
    for (i, w) in ws.iter().enumerate() {
        if w.completed_depth == 0 || i == best {
            continue;
        }
        let b = &ws[best];
        if b.best_score.abs() >= TB_WIN_BOUND {
            if w.best_score > b.best_score {
                best = i;
            }
        } else if w.best_score >= TB_WIN_BOUND || votes(w.best_move) > votes(b.best_move) {
            best = i;
        }
    }
    best
}

fn search_main(ws: &mut [Box<Worker>], pos: &Position, lim: &Limits, t0: i64) {
    let time = TimeManager::new(lim, pos.stm, t0);

    let main = &mut ws[0];
    main.pos.copy_from(pos);
    let mut ml = MoveList::new();
    let mut legal = MoveList::new();
    main.pos.generate(GenType::All, &mut ml);
    for &m in ml.as_slice() {
        if main.pos.is_legal(m) {
            legal.add(m);
        }
    }
    TT.ensure();
    TT.new_search();
    let tb = syzygy::rank_root(&mut main.pos, &mut legal);

    for (i, w) in ws.iter_mut().enumerate() {
        if i != 0 {
            w.pos.copy_from(pos);
        }
        w.root_moves = legal.clone();
        w.ctr.nodes.store(0, Relaxed);
        w.ctr.tb_hits.store(0, Relaxed);
        w.time = time;
        w.tb = tb.clone();
    }

    let (first, rest) = ws.split_at_mut(1);
    std::thread::scope(|s| {
        for w in rest.iter_mut() {
            std::thread::Builder::new()
                .stack_size(SEARCH_STACK)
                .spawn_scoped(s, move || w.iterate(lim, true))
                .expect("spawn helper thread");
        }
        first[0].iterate(lim, false);
        // in infinite / ponder mode, wait for stop (or ponderhit) before answering
        while !STOP.load(Relaxed) && (lim.infinite || PONDERING.load(Relaxed)) {
            std::thread::sleep(Duration::from_millis(1));
        }
        STOP.store(true, Relaxed);
    });

    let bi = pick_best_worker(ws);
    let seldepth = ws[0].seldepth;
    let best = &ws[bi];
    if bi != 0 && best.pv_len > 0 {
        print_info(best, best.completed_depth, seldepth, best.best_score);
    }
    let bm = best.best_move;
    let mut pm = if best.pv_len >= 2 && best.pv[0] == bm { best.pv[1] } else { MOVE_NONE };
    if bm != 0 && pm == 0 {
        // ponder move from the TT when the PV got cut short
        let p = &mut ws[0].pos;
        p.make_move(bm);
        let (te, hit) = TT.probe(p.s().key);
        if hit && p.is_pseudo_legal(te.mv()) && p.is_legal(te.mv()) {
            pm = te.mv();
        }
        p.unmake_move(bm);
    }
    let mut out = format!("bestmove {}", move_to_uci(bm));
    if pm != 0 {
        out += &format!(" ponder {}", move_to_uci(pm));
    }
    uci_out(&out);
}
