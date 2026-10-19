<p align="center">
  <img src="waschbar.png" alt="Waschbar logo" width="320">
</p>

# Waschbar

Waschbar (Waschbär, German for raccoon) is a UCI chess engine written in Rust, with the NNUE inference code from Erebus. It pairs an NNUE evaluation with a modern alpha-beta search, and supports
multi-threading, Syzygy tablebases, Chess960 and pondering. It is a command-line program
with no board of its own: use it from a UCI chess GUI.


## Features

### Search
- Principal variation search with iterative deepening and aspiration windows
- Transposition table: 3-entry buckets, generation-based replacement, huge-page backed on
  Linux, shared lock-free between threads
- Lazy SMP multi-threading (shared transposition table), with the final move chosen by a vote
  across threads
- Pruning: null-move pruning, reverse futility pruning, razoring, futility pruning,
  late move pruning, history pruning, SEE pruning, probcut, mate distance pruning
- Reductions and extensions: late move reductions, internal iterative reductions,
  singular extensions (including double and negative extensions)
- Quiescence search with SEE and futility pruning
- Move ordering: staged move picker (TT move, good captures, killers, counter move,
  quiets, bad captures), main/butterfly, capture, pawn and continuation histories
- Static evaluation correction history (pawn structure, non-pawn material, continuation)
- Repetition detection, including upcoming repetitions (cuckoo tables)
- Time management that adapts to best-move stability, score drops and the share of nodes
  spent on the best move

### Evaluation
- NNUE: 768 inputs × 8 king buckets → 1024×2 → 16 → 32 → 1, with 8 output buckets and
  SCReLU activations (`nets/net.nnue`.
- Lazy incremental accumulators: a move only records its feature changes, and accumulators
  are brought up to date when a position is actually evaluated
- King-bucket refresh cache ("finny table") and a per-thread eval cache
- Sparse first hidden layer (skips input pairs that are zero)
- AVX2 on x86-64, NEON on ARMv8 and a scalar fallback; all give bit-identical evaluations
- The network is embedded in the binary, so no separate file is needed at runtime

### Variants and protocol
- UCI protocol
- Chess960 and Double Fischer Random Chess (X-FEN and Shredder-FEN castling rights)
- Pondering (`go ponder` / `ponderhit`)
- Syzygy endgame tablebases up to 7 pieces via
  [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy):
  - DTZ root filter (from erebus): when winning, zeroing moves first, then the shortest
    DTZ, avoiding repeats; when drawing, every drawing move; when losing, the longest
    resistance. It respects the 50-move clock (`Syzygy50MoveRule`)
  - WDL probes during search

## Requirements

- Rust 1.88 or newer (edition 2024): install with [rustup](https://rustup.rs/).
- x86-64 (fastest with BMI2 + AVX2: Intel Haswell / AMD Zen 3 or newer; older CPUs work with
  a generic build) or 64-bit ARM.
- Linux, macOS or Windows.

## Building

```sh
cargo build --release
```

This produces `target/release/waschbar` (`waschbar.exe` on Windows): an LTO build with the
network embedded, tuned for the CPU it is built on (`-C target-cpu=native`, set in
`.cargo/config.toml`). That binary may not run on an older CPU; build for another machine
by overriding the target CPU:

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release   # AVX2 + BMI2
RUSTFLAGS="-C target-cpu=x86-64-v4" cargo build --release   # AVX-512
```

| Command | Result |
|---|---|
| `cargo build --release` | Optimized build for this CPU |
| `cargo test --release` | Perft (standard and Chess960), NNUE kernels, lazy accumulators against full refreshes, reference evaluations of the net |
| `target/release/waschbar bench 13` | Runs the benchmark |
| `cargo clean` | Removes all build output |

### Versioning

The version is the `version` field in `Cargo.toml`, and appears in the engine's `id name`
(`version = "1.1.2"` → `id name Waschbar 1.1.2`).

### Releases

[`.github/workflows/release.yml`](.github/workflows/release.yml) runs when a version tag is
pushed. The tag must match `Cargo.toml` (tag `v1.1.2` for version `1.1.2`):

```sh
git tag v1.1.2
git push origin v1.1.2
```

It builds 15 binaries:

| | x86-64-v1 | x86-64-v2 | x86-64-v3 | x86-64-v4 | ARM64 |
|---|---|---|---|---|---|
| Linux (static, musl) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Windows | ✓ | ✓ | ✓ | ✓ | ✓ |
| macOS | ✓ | ✓ | ✓ | ✓ | ✓ (Apple Silicon) |

Pick the highest x86-64 level your CPU supports: v1 runs on any 64-bit x86, v2 needs SSE4.2
and POPCNT (2009+), v3 needs AVX2 and BMI2 (Intel Haswell / AMD Zen 1+; fastest on Zen 3+),
v4 needs AVX-512. Each build runs the unit tests, perft and `bench` when the runner's CPU can
execute it, and the release is only published if every tested build reports the same bench
node count.

## UCI options

| Option | Type | Default | Description |
|---|---|---|---|
| `Hash` | spin | 256 | Transposition table size in MB (1–65536) |
| `Threads` | spin | 1 | Number of search threads (1–1024) |
| `Move Overhead` | spin | 30 | Milliseconds reserved per move for GUI and network lag |
| `Ponder` | check | false | Lets the GUI use pondering |
| `UCI_Chess960` | check | false | Chess960 mode; castling is written as king-takes-rook (e.g. `e1h1`) |
| `SyzygyPath` | string | `<empty>` | Tablebase directories, separated by `:` (`;` on Windows) |
| `SyzygyProbeDepth` | spin | 1 | Minimum depth for tablebase probes during search |
| `Syzygy50MoveRule` | check | true | Treat wins and losses that the 50-move rule turns into draws as draws |
| `SyzygyProbeLimit` | spin | 7 | Maximum number of pieces to probe |

Setting `SyzygyPath` reports what was found:

```
info string Syzygy: found 510 WDL and 510 DTZ files, up to 6-piece tablebases
```

Tablebase wins and losses appear as `cp 20000` / `cp -20000` (minus the distance in plies).
Actual mates appear as `mate N`.

The transposition table is allocated (and on Linux, backed by huge pages) on the first
`isready`, `ucinewgame` or `go`. On WSL2, huge-page compaction can make that first allocation
take a while; GUIs send `isready` before searching, so it happens before the clock runs.

## Extra commands

| Command | Description |
|---|---|
| `bench [depth]` | Fixed-depth search over 12 built-in positions; prints total nodes and nodes per second (progress goes to stderr). Also works as `./waschbar bench 13` from the shell |
| `perft <depth>` | Counts the leaf nodes of the current position at that depth |
| `d` | Prints the current position as a FEN |
| `eval` | Prints the network's static evaluation of the current position |

The bench node count is a quick check that a change did not alter search behaviour.


## Source layout

| File | Contents |
|---|---|
| `main.rs` | Startup: attack tables, Zobrist keys, network, then UCI / bench / perft-epd |
| `uci.rs` | UCI command loop, options, move and score notation, time management, Lazy SMP thread pool and ponder handling |
| `search.rs` | Iterative deepening, PVS, quiescence, staged move picker, history tables, pruning |
| `nnue.rs` | NNUE inference (from erebus): net loading, SIMD kernels, lazy accumulators |
| `eval.rs` | Score constants |
| `board.rs` | Position state, FEN, make/unmake, draw and repetition detection |
| `movegen.rs` | Pseudo-legal move generation, legality checks and perft |
| `moves.rs` | Move encoding, `MoveList` |
| `bitboard.rs` | Bit helpers and attack tables: PEXT sliders with BMI2, magic bitboards otherwise |
| `see.rs` | Static exchange evaluation |
| `zobrist.rs` | Zobrist keys and cuckoo tables |
| `tt.rs` | Transposition table |
| `syzygy.rs` | Syzygy tablebase integration (shakmaty-syzygy, DTZ root filter from erebus) |
| `bench.rs` | Benchmark and the EPD perft checker |
| `nets/` | The network |

## Credits

- NNUE inference code from Erebus.
- Syzygy probing uses [shakmaty-syzygy](https://github.com/niklasf/shakmaty-syzygy)
  (GPL-3.0), based on Ronald de Man's original probing code.

## License

Waschbar is licensed under the GNU General Public License v3.0. See [LICENSE](LICENSE).
