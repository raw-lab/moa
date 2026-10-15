//! The column-major sparse matrix MCL operates on.
//!
//! MCL runs on a *column-stochastic* matrix: every column is a probability
//! distribution over where flow leaving that node goes. We store it in
//! compressed-sparse-column (CSC) form — a `Vec` of columns, each a list of
//! `(row, value)` pairs kept sorted by row index. That layout is what the
//! reference implementation uses (`impala/matrix.c`, `impala/vector.c`) and it
//! is exactly what the expansion `M := M*M` wants: expanding column *j* is a
//! sparse linear combination of the columns named in column *j*.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// One column: `(row_index, value)` pairs, always sorted ascending by index and
/// free of duplicate indices. This mirrors `mclVector`.
pub type Column = Vec<(u32, f64)>;

/// A square, column-major sparse matrix (`mclMatrix`).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SparseMatrix {
    pub n: usize,
    pub cols: Vec<Column>,
}

impl SparseMatrix {
    /// An `n x n` all-zero matrix.
    pub fn new(n: usize) -> Self {
        SparseMatrix {
            n,
            cols: vec![Vec::new(); n],
        }
    }

    /// Build from an undirected weighted edge list. Parallel edges are summed
    /// and the graph is symmetrised (an edge `a-b` contributes to both column
    /// `a` and column `b`), matching how `mcx`/`mcl` ingest an `--abc` stream.
    /// Non-positive weights and out-of-range endpoints are dropped.
    pub fn from_edges(n: usize, edges: &[(u32, u32, f64)]) -> Self {
        let mut acc: Vec<HashMap<u32, f64>> = vec![HashMap::new(); n];
        for &(a, b, w) in edges {
            if a as usize >= n || b as usize >= n || !(w > 0.0) {
                continue;
            }
            *acc[b as usize].entry(a).or_insert(0.0) += w;
            if a != b {
                *acc[a as usize].entry(b).or_insert(0.0) += w;
            }
        }
        let mut m = SparseMatrix::new(n);
        for j in 0..n {
            let mut c: Column = acc[j].drain().collect();
            c.sort_unstable_by_key(|e| e.0);
            m.cols[j] = c;
        }
        m
    }

    /// Total number of stored entries.
    pub fn nnz(&self) -> usize {
        self.cols.iter().map(|c| c.len()).sum()
    }

    /// Value at `(row, col)`, or 0 if absent. Uses the sorted invariant for a
    /// binary search.
    #[inline]
    pub fn get(&self, row: u32, col: usize) -> f64 {
        match self.cols[col].binary_search_by_key(&row, |e| e.0) {
            Ok(p) => self.cols[col][p].1,
            Err(_) => 0.0,
        }
    }

    /// The diagonal (self-loop) value of column `j`.
    #[inline]
    pub fn self_val(&self, j: usize) -> f64 {
        self.get(j as u32, j)
    }

    /// The largest value in column `j` (0 for an empty column).
    #[inline]
    pub fn max_val(&self, j: usize) -> f64 {
        self.cols[j].iter().map(|e| e.1).fold(0.0, f64::max)
    }

    /// `mclxAdjustLoops` with the `max` rule: give every node a self-loop whose
    /// weight equals the largest off-diagonal weight in its column (1.0 for an
    /// isolated node). This is what keeps flow from annihilating a node during
    /// the very first expansion — without it, a degree-1 leaf can lose all of
    /// its mass. The reference sets loops before making the matrix stochastic.
    pub fn add_self_loops_max(&mut self) {
        for j in 0..self.n {
            let jj = j as u32;
            let mut best = 0.0f64;
            let mut has = false;
            for &(i, v) in self.cols[j].iter() {
                if i == jj {
                    has = true;
                } else if v > best {
                    best = v;
                }
            }
            let w = if best > 0.0 { best } else { 1.0 };
            if has {
                for e in self.cols[j].iter_mut() {
                    if e.0 == jj {
                        e.1 = w;
                    }
                }
            } else {
                // Insert in sorted position (usually the tail is fine, but keep
                // the invariant regardless of where the diagonal falls).
                let pos = self.cols[j]
                    .binary_search_by_key(&jj, |e| e.0)
                    .unwrap_or_else(|p| p);
                self.cols[j].insert(pos, (jj, w));
            }
        }
    }

    /// Scale each column to sum 1 (`mclxMakeStochastic`). An all-zero column is
    /// left as-is (its node is isolated and will become a singleton).
    pub fn make_stochastic(&mut self) {
        for c in self.cols.iter_mut() {
            let s: f64 = c.iter().map(|e| e.1).sum();
            if s > 0.0 {
                let inv = 1.0 / s;
                for e in c.iter_mut() {
                    e.1 *= inv;
                }
            }
        }
    }
}
