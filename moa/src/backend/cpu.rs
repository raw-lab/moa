//! Serial and rayon expansion.

use super::compose_column;
use crate::matrix::{Column, SparseMatrix};
use rayon::prelude::*;

/// Single-threaded raw expansion. This is the definition the other backends are
/// validated against.
pub fn expand_serial(m: &SparseMatrix) -> SparseMatrix {
    let n = m.n;
    let mut scratch = vec![0.0f64; n];
    let mut touched: Vec<u32> = Vec::new();
    let mut out = SparseMatrix::new(n);
    for j in 0..n {
        out.cols[j] = compose_column(m, j, &mut scratch, &mut touched);
    }
    out
}

/// Data-parallel raw expansion: each column is composed independently, so we
/// simply `map` the compose over columns in parallel. Every rayon worker gets
/// its own dense scratch buffer via `map_init`, so there is no shared mutable
/// state and the result is deterministic and identical to [`expand_serial`].
pub fn expand_rayon(m: &SparseMatrix) -> SparseMatrix {
    let n = m.n;
    let cols: Vec<Column> = (0..n)
        .into_par_iter()
        .map_init(
            || (vec![0.0f64; n], Vec::<u32>::new()),
            |(scratch, touched), j| compose_column(m, j, scratch, touched),
        )
        .collect();
    SparseMatrix { n, cols }
}
