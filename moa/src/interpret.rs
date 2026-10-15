//! Reading clusters off the fixed point — a port of `mcl/interpret.c`.
//!
//! Two stages, exactly as the reference:
//!
//! 1. **`mclDag`** prunes the converged (stochastic) matrix to a directed
//!    acyclic-ish flow graph. For each column it keeps only entries at or above
//!    a per-column bar
//!    ```text
//!      bar = selfval < maxval ? 0.999*selfval + 0.001*maxval
//!                             : selfval / (1 + 0.01)
//!    ```
//!    (`w_selfval = 0.999`, `w_maxval = 0.001`, `delta = 0.01`). An attractor's
//!    own self-loop survives; a satellite keeps (essentially) only its edge to
//!    the attractor it flows into.
//!
//! 2. **`mclInterpret`** marks **attractors** (nodes whose DAG column retains
//!    their own self-loop), groups connected attractors into **attractor
//!    systems** (the cluster cores), and attaches every remaining node to the
//!    system its flow reaches. Reachability is transitive, so a union-find over
//!    DAG edges reproduces the reference's closure-based assignment.
//!
//! Recall the storage convention: `M[i][j]` is flow from `j` to `i`, so the
//! out-neighbours of node `j` are the row indices listed in column `j`.

use crate::matrix::SparseMatrix;
use std::collections::HashMap;

const W_SELFVAL: f64 = 0.999;
const W_MAXVAL: f64 = 0.001;
const DELTA: f64 = 0.01;

/// `mclDag`: prune the converged matrix to its flow DAG.
fn dag(m: &SparseMatrix) -> SparseMatrix {
    let n = m.n;
    let mut d = SparseMatrix::new(n);
    for j in 0..n {
        let selfval = m.self_val(j);
        let maxval = m.max_val(j);
        let bar = if selfval < maxval {
            W_SELFVAL * selfval + W_MAXVAL * maxval
        } else {
            selfval / (1.0 + DELTA)
        };
        d.cols[j] = m.cols[j].iter().copied().filter(|e| e.1 >= bar).collect();
    }
    d
}

/// `mclInterpret`: attractors → attractor systems → closure assignment.
pub fn interpret(m: &SparseMatrix) -> Vec<Vec<u32>> {
    let n = m.n;
    if n == 0 {
        return Vec::new();
    }
    // Work on a stochastic copy so the bar comparison matches the reference,
    // which interprets a column-stochastic iterand.
    let mut ms = m.clone();
    ms.make_stochastic();
    let d = dag(&ms);

    // Attractors: DAG column j still contains its own diagonal.
    let is_attractor: Vec<bool> = (0..n).map(|j| d.self_val(j) > 0.0).collect();

    let mut uf = UnionFind::new(n);

    // Attractor systems: connected attractors (via DAG edges among attractors).
    for j in 0..n {
        if !is_attractor[j] {
            continue;
        }
        for &(i, v) in d.cols[j].iter() {
            if v > 0.0 && is_attractor[i as usize] {
                uf.union(j, i as usize);
            }
        }
    }
    // Every other node joins the system its flow reaches. Union with each
    // out-neighbour; transitivity of union-find realises the full closure.
    for j in 0..n {
        if is_attractor[j] {
            continue;
        }
        for &(i, v) in d.cols[j].iter() {
            if v > 0.0 {
                uf.union(j, i as usize);
            }
        }
    }

    let mut groups: HashMap<usize, Vec<u32>> = HashMap::new();
    for v in 0..n {
        groups.entry(uf.find(v)).or_default().push(v as u32);
    }
    let mut out: Vec<Vec<u32>> = groups.into_values().collect();
    for g in out.iter_mut() {
        g.sort_unstable();
    }
    out.sort_unstable_by_key(|g| g[0]);
    out
}

/// Union-find with path compression (`clmUGraphComponents` in spirit).
pub struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
        }
    }
    pub fn find(&mut self, x: usize) -> usize {
        let mut r = x;
        while self.parent[r] != r {
            r = self.parent[r];
        }
        let mut c = x;
        while self.parent[c] != r {
            let nx = self.parent[c];
            self.parent[c] = r;
            c = nx;
        }
        r
    }
    pub fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}
