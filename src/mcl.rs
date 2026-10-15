//! The MCL process: parameters, per-component iteration, and the global driver.
//!
//! ### Why per-component, and why it is still exact
//!
//! Expansion, pruning and inflation are column-local and flow never crosses
//! between connected components, so each component evolves exactly as it would
//! inside the whole-graph matrix. The one global coupling in mcl is the stop
//! rule: iterate until `max over all columns of chaos < 1e-4` (`proc.c:463`).
//! A component that settles early keeps being iterated in the whole-graph run.
//!
//! moa therefore runs every component to its *own* convergence in parallel,
//! takes `T = max tᶜ`, and advances any component with `tᶜ < T` to exactly
//! `T` iterations (re-checking the global criterion at `T`, `T+1`, …). A
//! component whose iterand stops changing bit-for-bit is *frozen* — further
//! iterations are provably no-ops — so this reconciliation is nearly free while
//! the final matrices are exactly those of a single whole-graph run.

use crate::engine::Engine;
use crate::graph::{Csc, Graph};
use crate::interpret;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;

/// MCL parameters, named and defaulted as in `mcl 22-282` (`mcl -z`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MclParams {
    /// `-I` main inflation (default 2.0). Higher means finer clusters.
    pub inflation: f64,
    /// `-P` prune number: entries below `0.99999 / P` are pruned (0 = off).
    pub prune: usize,
    /// `-S` select number: keep at most this many entries per column (0 = off).
    pub select: usize,
    /// `-R` recover number (0 = off).
    pub recover: usize,
    /// `-pct` recovery percentage as a fraction (default 0.90).
    pub pct: f64,
    /// Stop when chaos drops *below* this (`chaosLimit`, 1e-4).
    pub chaos_limit: f64,
    /// `-L` main loop length (default 10000).
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
    /// `mxp->precision = num_prune ? 0.99999 / num_prune : 0.0` (procinit.c:673).
    #[inline]
    pub fn precision(&self) -> f64 {
        if self.prune > 0 {
            0.99999 / self.prune as f64
        } else {
            0.0
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        if !(self.inflation.is_finite() && self.inflation > 0.0) {
            return Err(format!("inflation must be a positive number (got {})", self.inflation));
        }
        if !(0.0..=1.0).contains(&self.pct) {
            return Err(format!("pct must be within 0..100 (got {})", self.pct * 100.0));
        }
        if self.max_iter == 0 {
            return Err("max iterations must be at least 1".into());
        }
        Ok(())
    }
}

/// The iteration state of one component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompRun {
    /// The current iterand. Shared (`Arc`) so HydraMPP block tasks can read it
    /// without copying.
    pub m: Arc<Csc>,
    /// Iterations performed.
    pub t: usize,
    /// Chaos measured during iteration `t` (∞ before the first).
    pub chaos: f64,
    /// The last iteration left the matrix bit-identical: a fixed point.
    pub frozen: bool,
}

impl CompRun {
    /// Start state from a component adjacency (loops + stochastic applied).
    pub fn start(adj: Csc) -> CompRun {
        let m = adj.into_start_matrix();
        // A singleton is [1]; a pair is [[½,½],[½,½]] once loops are set to the
        // max weight. Both are already fixed points with zero chaos — exactly
        // what the first mcl iteration would compute — so skip the work.
        if m.n <= 2 {
            return CompRun { m: Arc::new(m), t: 1, chaos: 0.0, frozen: true };
        }
        CompRun { m: Arc::new(m), t: 0, chaos: f64::INFINITY, frozen: false }
    }
}

/// How one matrix step is executed (chosen per component by the engine).
pub trait StepExec: Sync {
    fn step(&self, m: &Arc<Csc>, p: &MclParams) -> (Csc, f64);
}

/// Advance `st` either to its own convergence (`target = None`) or to exactly
/// `target` iterations, honouring `max_iter`.
pub fn advance(st: &mut CompRun, p: &MclParams, target: Option<usize>, exec: &dyn StepExec) {
    loop {
        match target {
            Some(t) if st.t >= t => break,
            None if st.t > 0 && st.chaos < p.chaos_limit => break,
            _ => {}
        }
        if st.t >= p.max_iter {
            break;
        }
        if st.frozen {
            // Every further iteration reproduces this matrix and this chaos.
            st.t = match target {
                Some(t) => t.min(p.max_iter),
                None => p.max_iter, // frozen above the limit: never converges
            };
            break;
        }
        let t0 = if trace() { Some(std::time::Instant::now()) } else { None };
        let (next, chaos) = exec.step(&st.m, p);
        if let Some(t0) = t0 {
            if st.m.n >= 1000 {
                eprintln!(
                    "  [moa] it {:>3}  n {:>8}  nnz {:>11} → {:>11}  chaos {:>9.3e}  {:.3}s",
                    st.t + 1,
                    st.m.n,
                    st.m.nnz(),
                    next.nnz(),
                    chaos,
                    t0.elapsed().as_secs_f64()
                );
            }
        }
        st.t += 1;
        st.chaos = chaos;
        if next == *st.m {
            st.frozen = true;
        }
        st.m = Arc::new(next);
    }
}

static TRACE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// Print one line per iteration of every component with ≥ 1000 nodes (stderr).
pub fn set_trace(on: bool) {
    TRACE.store(on, std::sync::atomic::Ordering::Relaxed);
}
#[inline]
fn trace() -> bool {
    TRACE.load(std::sync::atomic::Ordering::Relaxed)
}

/// Full clustering of a graph, plus diagnostics.
#[derive(Clone, Debug)]
pub struct MclOutput {
    /// Clusters (global node ids, ascending), in mcl's output order.
    pub clusters: Vec<Vec<u32>>,
    pub iterations: usize,
    pub chaos: f64,
    pub components: usize,
    pub largest_component: usize,
    pub attractors: usize,
    pub overlap: usize,
    pub truants: usize,
    pub nnz_start: usize,
    pub converged: bool,
    pub secs_iterate: f64,
    pub secs_interpret: f64,
}

/// Run MCL on `g` with `engine`. The result is engine-independent.
pub fn run(g: Graph, p: &MclParams, engine: &Engine) -> MclOutput {
    let ts = Instant::now();
    let comps = g.components();
    drop(g);
    if trace() {
        eprintln!("  [moa] components {:.3}s", ts.elapsed().as_secs_f64());
    }
    let largest_component = comps.first().map(|c| c.nodes.len()).unwrap_or(0);
    let nodes: Vec<Vec<u32>> = comps.iter().map(|c| c.nodes.clone()).collect();
    let mut runs: Vec<CompRun> = comps.into_par_iter().map(|c| CompRun::start(c.adj)).collect();
    let nnz_start = runs.iter().map(|r| r.m.nnz()).sum();

    let t0 = Instant::now();
    // Phase 1: every component to its own convergence.
    let all: Vec<usize> = (0..runs.len()).collect();
    engine.advance(&mut runs, &all, p, None);
    // Phase 2: reconcile to the global stop rule.
    let mut t_glob = runs.iter().map(|r| r.t).max().unwrap_or(0);
    loop {
        let lagging: Vec<usize> = (0..runs.len()).filter(|&i| runs[i].t < t_glob).collect();
        if !lagging.is_empty() {
            engine.advance(&mut runs, &lagging, p, Some(t_glob));
        }
        let ok = runs.iter().all(|r| r.chaos < p.chaos_limit);
        if ok || t_glob >= p.max_iter {
            break;
        }
        t_glob += 1;
    }
    let chaos = runs.iter().map(|r| r.chaos).fold(0.0f64, f64::max);
    let secs_iterate = t0.elapsed().as_secs_f64();

    // Interpret each limit, map back to global ids.
    let t1 = Instant::now();
    let parts: Vec<interpret::Interpretation> = runs.par_iter().map(|r| interpret::interpret(&r.m)).collect();
    let mut clusters = Vec::new();
    let mut truants = Vec::new();
    let (mut attractors, mut overlap) = (0, 0);
    for (part, ids) in parts.into_iter().zip(&nodes) {
        attractors += part.attractors;
        overlap += part.overlap;
        for c in part.clusters {
            clusters.push(c.into_iter().map(|l| ids[l as usize]).collect::<Vec<u32>>());
        }
        truants.extend(part.truants.into_iter().map(|l| ids[l as usize]));
    }
    let n_truants = truants.len();
    if !truants.is_empty() {
        truants.sort_unstable();
        clusters.push(truants); // clmEnstrict gathers missing nodes into one cluster
    }
    order_clusters(&mut clusters);
    MclOutput {
        clusters,
        iterations: t_glob,
        chaos,
        components: nodes.len(),
        largest_component,
        attractors,
        overlap,
        truants: n_truants,
        nnz_start,
        converged: chaos < p.chaos_limit,
        secs_iterate,
        secs_interpret: t1.elapsed().as_secs_f64(),
    }
}

/// mcl's output order: largest cluster first, ties by lexicographic member ids.
pub fn order_clusters(c: &mut [Vec<u32>]) {
    c.par_sort_unstable_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
}

/// Convenience for library users: run on a raw edge list with the rayon engine.
pub fn cluster_edges(n: usize, edges: &[(u32, u32, f64)], p: &MclParams) -> MclOutput {
    run(Graph::from_edges(n, edges), p, &Engine::Rayon)
}
