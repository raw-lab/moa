//! Reading clusters off the MCL limit — `mcl/interpret.c` + `clew/cat.c`.
//!
//! 1. **`mclDag`.** For each column keep entries `>= bar`, with
//!    `bar = selfval < maxval ? 0.999·selfval + 0.001·maxval : selfval/1.01`.
//! 2. **`mclInterpret`.** Attractors are nodes whose DAG column keeps their own
//!    loop. Connected attractors (undirected) form *attractor systems*, ordered
//!    as `clmUGraphComponents` leaves them: largest first, ties broken by
//!    lexicographic member order (`mclvSizeRevCmp`). Every other node is
//!    assigned to **every** system its flow closure reaches.
//! 3. **`clmEnstrict` with `-overlap cut`** (mcl's default) resolves overlap by
//!    keeping each node only in its first (lowest-index) system, and gathers
//!    nodes that reach no attractor into one extra cluster.
//!
//! The union-find shortcut used by moa 1.x is *not* equivalent: one node whose
//! closure touches two systems merged them into a single cluster.

use crate::graph::{Csc, UnionFind};
use crate::util::Marker;

/// Interpretation of one component (local node ids).
pub struct Interpretation {
    /// Clusters, in mcl's system order (largest first, then lexicographic).
    pub clusters: Vec<Vec<u32>>,
    /// Nodes whose closure reached no attractor (mcl's "missing" nodes).
    pub truants: Vec<u32>,
    /// Instances of overlap that `cut` resolved.
    pub overlap: usize,
    pub attractors: usize,
}

const W_SELFVAL: f64 = 0.999;
const W_MAXVAL: f64 = 0.001;
const DELTA: f64 = 0.01;

/// `mclDag` on a (stochastic) limit matrix.
pub fn dag(m: &Csc) -> Csc {
    let mut ptr = Vec::with_capacity(m.n + 1);
    ptr.push(0);
    let mut idx = Vec::with_capacity(m.nnz());
    let mut val = Vec::with_capacity(m.nnz());
    for j in 0..m.n {
        let (ix, vx) = m.col(j);
        let selfval = match ix.binary_search(&(j as u32)) {
            Ok(p) => vx[p] as f64,
            Err(_) => 0.0,
        };
        let maxval = vx.iter().fold(0.0f64, |a, &v| a.max(v as f64));
        let bar = if selfval < maxval { W_SELFVAL * selfval + W_MAXVAL * maxval } else { selfval / (1.0 + DELTA) };
        for (&i, &v) in ix.iter().zip(vx) {
            if v as f64 >= bar {
                idx.push(i);
                val.push(v);
            }
        }
        ptr.push(idx.len());
    }
    Csc { n: m.n, ptr, idx, val }
}

/// Full interpretation of one component's limit matrix.
#[allow(clippy::needless_range_loop)] // loops run over node ids
pub fn interpret(m: &Csc) -> Interpretation {
    let n = m.n;
    let d = dag(m);
    let is_attr: Vec<bool> = (0..n).map(|j| d.col(j).0.binary_search(&(j as u32)).is_ok()).collect();

    // Attractor systems: undirected components of the attractor subgraph.
    let mut uf = UnionFind::new(n);
    for j in 0..n {
        if is_attr[j] {
            for &i in d.col(j).0 {
                if is_attr[i as usize] {
                    uf.union(i as usize, j);
                }
            }
        }
    }
    let mut root_sys = vec![u32::MAX; n];
    let mut systems: Vec<Vec<u32>> = Vec::new();
    for j in 0..n {
        if is_attr[j] {
            let r = uf.find(j);
            if root_sys[r] == u32::MAX {
                root_sys[r] = systems.len() as u32;
                systems.push(Vec::new());
            }
            systems[root_sys[r] as usize].push(j as u32); // ascending by construction
        }
    }
    // clmUGraphComponents → mclxColumnsRealign(mclvSizeRevCmp).
    systems.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
    let mut sys_of = vec![u32::MAX; n];
    for (s, mem) in systems.iter().enumerate() {
        for &v in mem {
            sys_of[v as usize] = s as u32;
        }
    }
    let attractors = systems.iter().map(|s| s.len()).sum();

    // Non-attractors: flow closure over DAG out-edges; collect the systems hit.
    let mut assign = sys_of.clone();
    let mut overlap = 0usize;
    let mut truants = Vec::new();
    let mut mark = Marker::new(n);
    let mut stack: Vec<u32> = Vec::new();
    for j in 0..n {
        if is_attr[j] {
            continue;
        }
        mark.clear();
        stack.clear();
        for &i in d.col(j).0 {
            if mark.first(i) {
                stack.push(i);
            }
        }
        let mut best = u32::MAX;
        let mut hits = 0usize;
        let mut seen_sys = Vec::new();
        while let Some(v) = stack.pop() {
            let s = sys_of[v as usize];
            if s != u32::MAX && !seen_sys.contains(&s) {
                seen_sys.push(s);
                hits += 1;
                best = best.min(s);
            }
            for &i in d.col(v as usize).0 {
                if mark.first(i) {
                    stack.push(i);
                }
            }
        }
        if hits == 0 {
            truants.push(j as u32);
        } else {
            overlap += hits - 1;
            assign[j] = best;
        }
    }

    let mut clusters: Vec<Vec<u32>> = vec![Vec::new(); systems.len()];
    for v in 0..n {
        let s = assign[v];
        if s != u32::MAX {
            clusters[s as usize].push(v as u32);
        }
    }
    clusters.retain(|c| !c.is_empty());
    Interpretation { clusters, truants, overlap, attractors }
}
