//! Score constants.

pub const MAX_PLY: i32 = 128;
pub const MATE: i32 = 32000;
pub const MATE_BOUND: i32 = MATE - 2 * MAX_PLY;
/// Tablebase win, scored TB_WIN_SCORE - ply.
pub const TB_WIN_SCORE: i32 = MATE_BOUND - 1;
/// |score| >= this: proven mate or TB result.
pub const TB_WIN_BOUND: i32 = MATE_BOUND - MAX_PLY;
pub const INF: i32 = 32001;
pub const VALUE_NONE: i32 = 32002;
