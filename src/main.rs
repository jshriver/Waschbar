//! Waschbar: a UCI chess engine with NNUE evaluation (a Rust port of the original C++ Waschbar).
//!
//!   waschbar                         UCI mode
//!   waschbar bench [depth]           node-count benchmark
//!   waschbar perft-epd FILE [depth]  check perft counts from an EPD file

// Style lints that fight the engine's table-driven idiom (index loops over parallel arrays,
// the `&*(&raw const STATIC)` accessor for init-once tables, aligned doc tables).
#![allow(
    clippy::needless_range_loop,
    clippy::deref_addrof,
    clippy::doc_overindented_list_items,
    clippy::nonminimal_bool
)]

mod bench;
mod bitboard;
mod board;
mod eval;
mod movegen;
mod moves;
mod nnue;
mod search;
mod see;
mod syzygy;
mod tt;
mod uci;
mod zobrist;

use uci::{Pool, SEARCH_STACK};

fn main() {
    bitboard::init();
    zobrist::init_keys();
    if let Err(e) = nnue::init() {
        eprintln!("failed to load embedded network: {e}");
        std::process::exit(1);
    }
    tt::TT.resize(256);
    let args: Vec<String> = std::env::args().collect();

    // everything runs on a thread with a big stack (Windows' main thread has 1 MB)
    let code = std::thread::Builder::new()
        .name("main".into())
        .stack_size(SEARCH_STACK)
        .spawn(move || run(&args))
        .expect("spawn main thread")
        .join()
        .unwrap_or(1);
    std::process::exit(code);
}

fn run(args: &[String]) -> i32 {
    let mut pool = Pool::new(1);
    match args.get(1).map(String::as_str) {
        Some("bench") => {
            let d = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(bench::DEFAULT_DEPTH);
            bench::bench(&mut pool, d);
            0
        }
        Some("perft-epd") => {
            let Some(path) = args.get(2) else {
                eprintln!("usage: waschbar perft-epd <file.epd> [maxdepth]");
                return 2;
            };
            let d = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(5);
            (bench::perft_epd(path, d) > 0) as i32
        }
        _ => {
            uci::uci_loop(&mut pool);
            0
        }
    }
}
