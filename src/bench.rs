//! Fixed-depth node-count benchmark ("bench [depth]") and the EPD perft checker.

use std::io::Write;
use std::sync::atomic::Ordering::Relaxed;

use crate::board::Position;
use crate::movegen::perft;
use crate::moves::{GenType, MoveList};
use crate::syzygy::TbSearch;
use crate::uci::{Limits, PONDERING, Pool, STOP, TimeManager, now_ms};
use crate::tt::TT;

pub const DEFAULT_DEPTH: i32 = 12;

const FENS: [&str; 12] = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbqkb1r/pp1p1ppp/2p5/4P3/2B5/8/PPP1NnPP/RNBQK2R w KQkq - 0 6",
    "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PP3PPP/R2QKB1R w KQ - 0 8",
    "2r3k1/pp3ppp/1qn1p3/3pP3/3P4/P1Q2N2/1P3PPP/2R3K1 b - - 0 20",
    "6k1/5pp1/7p/8/8/7P/5PP1/3R2K1 w - - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    "3r1rk1/p4ppp/1p2pn2/2q5/2P5/1P2Q3/P3BPPP/3R1RK1 w - - 0 18",
    "8/8/4k3/8/2p5/2P2K2/8/8 w - - 0 1",
    "r2q1rk1/ppp2ppp/2n1bn2/2b1p3/3pP3/3P1NPP/PPP1NPB1/R1BQ1RK1 b - - 0 9",
];

fn bar(done: usize, total: usize, width: usize) -> String {
    let filled = done * width / total.max(1);
    format!("{}{}", "█".repeat(filled), "░".repeat(width - filled))
}

/// Searches every bench position to `depth` on one thread with a fresh 16 MB TT. Progress
/// goes to stderr; stdout gets the single "bench: N nodes T ms S nps" line that tools parse.
pub fn bench(pool: &mut Pool, depth: i32) {
    let lim = Limits { depth, ..Default::default() };
    TT.resize(16);
    TT.ensure();
    eprintln!("🏁 Bench: {} positions at depth {depth}", FENS.len());
    let mut total = 0u64;
    let t0 = now_ms();
    let mut best: (u64, usize) = (0, 0);
    for (i, fen) in FENS.iter().enumerate() {
        let mut pos = Position::new();
        pos.set_fen(fen);
        pool.clear();
        TT.new_search();
        PONDERING.store(false, Relaxed);
        STOP.store(false, Relaxed);
        let mut ws = pool.workers.lock().unwrap();
        let w = &mut ws[0];
        w.time = TimeManager { start_time: now_ms(), timed: false, ..Default::default() };
        w.tb = TbSearch::default();
        w.pos.copy_from(&pos);
        let mut ml = MoveList::new();
        w.root_moves.clear();
        w.pos.generate(GenType::All, &mut ml);
        for &m in ml.as_slice() {
            if w.pos.is_legal(m) {
                w.root_moves.add(m);
            }
        }
        let tp = now_ms();
        w.iterate(&lim, true);
        let n = w.nodes();
        total += n;
        if n > best.0 {
            best = (n, i);
        }
        let el = now_ms() - t0;
        let eta = if i + 1 < FENS.len() { el * (FENS.len() - i - 1) as i64 / (i + 1) as i64 } else { 0 };
        eprintln!(
            "  {} {:>2}/{} ♟️  {:>9} nodes {:>6} ms  best {:<5}  ⏳ ETA {:.1}s",
            bar(i + 1, FENS.len(), 20),
            i + 1,
            FENS.len(),
            n,
            now_ms() - tp,
            crate::uci::move_to_uci(w.best_move),
            eta as f64 / 1000.0
        );
    }
    let el = (now_ms() - t0).max(1);
    eprintln!("🐢 Busiest position: #{} with {} nodes", best.1 + 1, best.0);
    println!("bench: {total} nodes {el} ms {} nps", total as i64 * 1000 / el);
    let _ = std::io::stdout().flush();
    eprintln!("✅ Bench done: {:.2} Mnps. Happy hunting! 🦝", total as f64 / el as f64 / 1000.0);
}

/// Checks perft counts from an EPD file: each line a FEN followed by `;D<depth> <nodes>`.
/// Returns the number of failures.
pub fn perft_epd(path: &str, max_depth: i32) -> usize {
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ cannot read {path}: {e}");
            return 1;
        }
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    println!("🧮 Perft check: {} positions from {path} (depth ≤ {max_depth})", lines.len());
    let t0 = now_ms();
    let (mut fails, mut total) = (0usize, 0u64);
    for (i, line) in lines.iter().enumerate() {
        let mut parts = line.split(';');
        let fen = parts.next().unwrap_or("").trim();
        let mut pos = Position::new();
        if !pos.set_fen(fen) {
            println!("❌ bad FEN: {fen}");
            fails += 1;
            continue;
        }
        for tok in parts {
            let tok = tok.trim();
            let Some(rest) = tok.strip_prefix('D') else { continue };
            let mut it = rest.split_whitespace();
            let (Some(d), Some(exp)) =
                (it.next().and_then(|v| v.parse::<i32>().ok()), it.next().and_then(|v| v.parse::<u64>().ok()))
            else {
                continue;
            };
            if d > max_depth {
                continue;
            }
            let got = perft(&mut pos, d);
            total += got;
            if got != exp {
                fails += 1;
                println!("💥 FAIL {fen} d{d} expected {exp} got {got}");
            }
        }
        if (i + 1) % 16 == 0 || i + 1 == lines.len() {
            let el = now_ms() - t0;
            let eta = el * (lines.len() - i - 1) as i64 / (i + 1) as i64;
            println!(
                "  {} {:>5}/{}  ⏳ ETA {:.1}s",
                bar(i + 1, lines.len(), 24),
                i + 1,
                lines.len(),
                eta as f64 / 1000.0
            );
        }
    }
    let s = (now_ms() - t0).max(1) as f64 / 1000.0;
    println!(
        "{} {} positions, {fails} failures, {s:.1}s, {:.1} Mnps",
        if fails == 0 { "🎉" } else { "😬" },
        lines.len(),
        total as f64 / s / 1e6
    );
    fails
}
