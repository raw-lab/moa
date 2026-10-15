//! Markov clustering — a faithful port of Stijn van Dongen's reference `mcl`
//! (`src/mcl/{expand,inflate,proc,interpret}.c`, `src/impala/vector.c`).
//!
//! The algorithm alternates two operators on a column-stochastic matrix until a
//! near-idempotent fixed point:
//!
//! * **expansion** `M := M*M` — spreads flow along length-two paths (delegated
//!   to a [`crate::backend::Backend`]);
//! * **inflation** `M[i][j] := M[i][j]^r`, renormalised per column — sharpens
//!   strong flow and starves weak flow. `r` (`-I`) is the granularity knob.
//!
//! Between the two, the reference keeps the matrix sparse with a
//! **prune / select / recover** scheme (all of `mclExpandVector`), and measures
//! progress with *chaos*:
//!
//! ```text
//!   center    = sum(v_i^2)                 (expand.c: vecMeasure)
//!   maxval    = max(v_i)
//!   chaos_col = (maxval - center) * nnz    (expand.c:503, nnz post-prune)
//!   chaos     = max over columns
//! ```
//!
//! Iteration stops once `chaos <= chaos_limit` (reference default `1e-4`). At a
//! fixed point every column is concentrated on a single attractor, so
//! `maxval == center` and chaos is exactly 0. Clusters are then read off with
//! [`crate::interpret`], the port of `interpret.c`.

use crate::backend::Backend;
use crate::interpret;
use crate::matrix::{Column, SparseMatrix};

/// MCL tunables, named as in the reference. The defaults are `dir_scheme` row 6
/// of `procinit.c` — `{prune 10000, select 1100, recover 1400, pct 90}` — the
/// scheme the `mcl` binary uses out of the box.
#[derive(Clone, Copy, Debug)]
pub struct MclParams {
    /// Inflation `-I`: higher means finer clusters. Reference default 2.0.
    pub inflation: f64,
    /// `-P`: drop entries below `0.99999 / prune` after expansion.
    pub prune: usize,
    /// `-S`: keep at most this many entries per column.
    pub select: usize,
    /// `-R`: if pruning shed too much mass, recover up to this many entries.
    pub recover: usize,
    /// `-pct` (as a fraction): a pruned column keeping less than this share of
    /// its mass triggers recovery.
    pub pct: f64,
    /// Stop when chaos falls to or below this (`chaosLimit`, default `1e-4`).
    pub chaos_limit: f64,
    /// Hard cap on iterations (`mainLoopLength`). Convergence is normally well
    /// under 100.
    pub max_iter: usize,
}

impl Default for MclParams {
    fn default() -> Self {
        MclParams {
            inflation: 2.0,
            prune: 10000,
            select: 1100,
            recover: 1400,
            pct: 0.90,
            chaos_limit: 1e-4,
            max_iter: 10000,
        }
    }
}

impl MclParams {
    /// The absolute prune cut `precision = 0.99999 / prune` (`procinit.c:673`).
    #[inline]
    fn precision(&self) -> f64 {
        if self.prune > 0 {
            0.99999 / self.prune as f64
        } else {
            0.0
        }
    }
}

/// Result of a run: the flat clustering plus convergence diagnostics.
pub struct MclResult {
    /// Clusters as sorted node-index lists, largest first is *not* enforced
    /// here (the CLI orders on output); order is by smallest member.
    pub clusters: Vec<Vec<u32>>,
    /// Iterations performed before the chaos criterion (or `max_iter`) was met.
    pub iterations: usize,
    /// Chaos at the final iteration (→ 0 at a clean fixed point).
    pub final_chaos: f64,
}

/// `vecMeasure` (expand.c): `maxval = max(v)`, `center = sum(v^2)` over the
/// full expanded column, before any pruning.
#[inline]
fn measure(col: &Column) -> (f64, f64) {
    let mut maxval = 0.0f64;
    let mut center = 0.0f64;
    for &(_, v) in col.iter() {
        center += v * v;
        if v > maxval {
            maxval = v;
        }
    }
    (maxval, center)
}

/// Keep the `k` largest entries of `col` by value, restoring index-sorted order.
fn keep_top_k(col: &mut Column, k: usize) {
    if col.len() <= k {
        return;
    }
    col.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    col.truncate(k);
    col.sort_unstable_by_key(|e| e.0);
}

/// Sum of a column's values.
#[inline]
fn mass(col: &Column) -> f64 {
    col.iter().map(|e| e.1).sum()
}

/// prune / select / recover, faithfully following `mclExpandVector`
/// (expand.c:349-503). `expanded` is the raw (stochastic) `M*M` column. Returns
/// the rescaled, stochastic result column together with its final entry count.
fn prune_select_recover(mut expanded: Column, j: usize, p: &MclParams) -> Column {
    // Buffer of the full expanded column, needed if recovery kicks in.
    let before: Column = if p.recover > 0 { expanded.clone() } else { Vec::new() };

    // --- prune: absolute precision cut -----------------------------------
    let cut = p.precision();
    if cut > 0.0 {
        expanded.retain(|e| e.1 >= cut);
    }
    let mass_prune = mass(&expanded);
    let mut mass_final = mass_prune;

    // --- recover, else select (with a nested recover) --------------------
    if p.recover > 0 && expanded.len() < p.recover && mass_prune < p.pct {
        // Too much mass lost by pruning: recover the largest dropped entries.
        expanded = before.clone();
        keep_top_k(&mut expanded, p.recover);
        mass_final = mass(&expanded);
    } else if p.select > 0 && expanded.len() > p.select {
        keep_top_k(&mut expanded, p.select);
        let mass_select = mass(&expanded);
        mass_final = mass_select;
        if p.recover > 0 && expanded.len() < p.recover && mass_select < p.pct {
            expanded = before.clone();
            keep_top_k(&mut expanded, p.recover);
            mass_final = mass(&expanded);
        }
    }

    // --- emergency loop: never let a node vanish (expand.c:394) ----------
    if expanded.is_empty() {
        expanded.push((j as u32, 1.0));
        mass_final = 1.0;
    }

    // --- rescale to stochastic (mclvScale) -------------------------------
    if mass_final > 0.0 && mass_final != 1.0 {
        let inv = 1.0 / mass_final;
        for e in expanded.iter_mut() {
            e.1 *= inv;
        }
    }
    expanded
}

/// `mclvInflate` (vector.c:1308): raise each entry to `power`, renormalise by
/// the power-sum. An all-nonpositive column is emptied (the caller's emergency
/// loop already prevents that upstream).
fn inflate_column(col: &mut Column, power: f64) {
    let mut powsum = 0.0;
    for e in col.iter_mut() {
        e.1 = e.1.powf(power);
        powsum += e.1;
    }
    if powsum <= 0.0 {
        col.clear();
        return;
    }
    let inv = 1.0 / powsum;
    for e in col.iter_mut() {
        e.1 *= inv;
    }
}

/// Run MCL to convergence on an undirected weighted graph, expanding with the
/// given backend. The clustering is independent of the backend.
pub fn run(mut m: SparseMatrix, p: &MclParams, backend: &Backend) -> MclResult {
    // Setup: self-loops so nothing vanishes, then column-stochastic.
    m.add_self_loops_max();
    m.make_stochastic();

    let mut iterations = 0usize;
    let mut chaos = f64::INFINITY;

    for it in 0..p.max_iter {
        iterations = it + 1;

        // Expansion M*M (the backend's job).
        let raw = backend.raw_expand(&m);

        // Post-process each expanded column: measure -> prune/select/recover ->
        // rescale, accumulating the chaos (max over columns).
        let n = raw.n;
        let mut next = SparseMatrix::new(n);
        chaos = 0.0;
        for (j, col) in raw.cols.into_iter().enumerate() {
            let (maxval, center) = measure(&col);
            let pruned = prune_select_recover(col, j, p);
            let c = (maxval - center) * pruned.len() as f64;
            if c > chaos {
                chaos = c;
            }
            next.cols[j] = pruned;
        }

        // Inflation, in place, per column.
        for col in next.cols.iter_mut() {
            inflate_column(col, p.inflation);
        }

        m = next;

        if chaos <= p.chaos_limit {
            break;
        }
    }

    let clusters = interpret::interpret(&m);
    MclResult {
        clusters,
        iterations,
        final_chaos: chaos,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{Backend, Engine};

    fn serial() -> Backend {
        Backend::start(Engine::Serial).unwrap()
    }

    #[test]
    fn clique_is_one_cluster() {
        let e = [(0, 1, 1.0), (0, 2, 1.0), (1, 2, 1.0)];
        let r = run(SparseMatrix::from_edges(3, &e), &MclParams::default(), &serial());
        assert_eq!(r.clusters.len(), 1);
        assert!(r.final_chaos <= MclParams::default().chaos_limit);
    }

    #[test]
    fn two_cliques_weak_bridge_split() {
        let mut e = vec![];
        for (a, b) in [(0, 1), (0, 2), (1, 2)] {
            e.push((a, b, 100.0));
        }
        for (a, b) in [(3, 4), (3, 5), (4, 5)] {
            e.push((a, b, 100.0));
        }
        e.push((2, 3, 0.1));
        let r = run(SparseMatrix::from_edges(6, &e), &MclParams::default(), &serial());
        assert_eq!(r.clusters.len(), 2, "{:?}", r.clusters);
    }

    #[test]
    fn disconnected_nodes_are_singletons() {
        let e: [(u32, u32, f64); 0] = [];
        let r = run(SparseMatrix::from_edges(4, &e), &MclParams::default(), &serial());
        assert_eq!(r.clusters.len(), 4);
    }
}
