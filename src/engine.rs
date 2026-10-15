//! Execution engines. All three produce bit-identical clusterings: the kernel is
//! deterministic per column, and only *where* columns and components run changes.
//!
//! * **serial** — one thread; the reference for equivalence checks.
//! * **rayon** — work-stealing over components (largest first, one task each)
//!   *and*, inside any component of ≥ [`PAR_COLS`] columns, over nnz-balanced
//!   column blocks. Nested parallelism is free with rayon, so one giant
//!   component and a million tiny ones both saturate the machine.
//! * **hydra** — the same two levels on HydraMPP: small and medium components
//!   are packed into batch tasks (amortising per-task overhead), large ones are
//!   driven from the host one iteration at a time with column-block tasks over
//!   a shared `Arc` matrix (zero-copy on a local runtime, shipped over TCP to
//!   `--hydra-client` / `--hydra-slurm` worker nodes otherwise).

use crate::graph::Csc;
use crate::kernel::{self, Block, Scratch};
use crate::mcl::{advance, CompRun, MclParams, StepExec};
use hydra_mpp::prelude::*;
use rayon::prelude::*;
use std::cell::RefCell;
use std::sync::Arc;

/// Components with at least this many columns get column-parallel steps…
pub const PAR_COLS: usize = 2048;
/// …and so does any iterand holding at least this many entries, whatever its
/// size: a dense 1,500-node family is as much work as a sparse 100k-node one.
pub const PAR_NNZ: usize = 60_000;

#[inline]
fn wants_parallel(m: &Csc) -> bool {
    m.n >= 64 && (m.n >= PAR_COLS || m.nnz() >= PAR_NNZ)
}

thread_local! {
    static SCRATCH: RefCell<Scratch> = RefCell::new(Scratch::new(0));
}

#[inline]
fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

/// Split `0..n` into ~`parts` ranges of roughly equal nnz (work proxy).
pub fn split_cols(m: &Csc, parts: usize) -> Vec<(usize, usize)> {
    let n = m.n;
    if n == 0 {
        return vec![];
    }
    let parts = parts.clamp(1, n);
    // Weight = column nnz + 1, so empty columns still split evenly.
    let total = m.nnz() + n;
    let per = total.div_ceil(parts);
    let mut out = Vec::with_capacity(parts);
    let (mut lo, mut acc) = (0usize, 0usize);
    for j in 0..n {
        acc += m.ptr[j + 1] - m.ptr[j] + 1;
        if acc >= per {
            out.push((lo, j + 1));
            lo = j + 1;
            acc = 0;
        }
    }
    if lo < n {
        out.push((lo, n));
    }
    out
}

/// Serial step: one block.
pub struct SerialStep;
impl StepExec for SerialStep {
    fn step(&self, m: &Arc<Csc>, p: &MclParams) -> (Csc, f64) {
        let b = with_scratch(|s| kernel::step_block(m, 0, m.n, p, s));
        kernel::stitch(m.n, vec![b])
    }
}

/// Rayon step: nnz-balanced column blocks in parallel.
pub struct RayonStep;
impl StepExec for RayonStep {
    fn step(&self, m: &Arc<Csc>, p: &MclParams) -> (Csc, f64) {
        let m: &Csc = m;
        let ranges = split_cols(m, rayon::current_num_threads() * 8);
        let blocks: Vec<Block> =
            ranges.into_par_iter().map(|(lo, hi)| with_scratch(|s| kernel::step_block(m, lo, hi, p, s))).collect();
        kernel::stitch(m.n, blocks)
    }
}

/// Rayon step chosen per iteration from the current iterand's size.
pub struct AdaptiveStep;
impl StepExec for AdaptiveStep {
    fn step(&self, m: &Arc<Csc>, p: &MclParams) -> (Csc, f64) {
        if wants_parallel(m) {
            RayonStep.step(m, p)
        } else {
            SerialStep.step(m, p)
        }
    }
}

/// An execution engine.
pub enum Engine {
    Serial,
    Rayon,
    Hydra(HydraEngine),
}

impl Engine {
    pub fn name(&self) -> &'static str {
        match self {
            Engine::Serial => "serial",
            Engine::Rayon => "rayon",
            Engine::Hydra(_) => "hydra",
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Engine::Serial => "serial · 1 thread".into(),
            Engine::Rayon => format!("rayon · {} threads · components ∥ columns", rayon::current_num_threads()),
            Engine::Hydra(h) => h.describe(),
        }
    }

    /// Advance the listed components (see [`advance`]).
    pub fn advance(&self, runs: &mut [CompRun], which: &[usize], p: &MclParams, target: Option<usize>) {
        match self {
            Engine::Serial => {
                for &i in which {
                    advance(&mut runs[i], p, target, &SerialStep);
                }
            }
            Engine::Rayon => {
                // Pull the selected runs out so rayon can own them mutably.
                let mut sel: Vec<(usize, CompRun)> =
                    which.iter().map(|&i| (i, std::mem::replace(&mut runs[i], placeholder()))).collect();
                sel.par_iter_mut().with_max_len(1).for_each(|(_, r)| advance(r, p, target, &AdaptiveStep));
                for (i, r) in sel {
                    runs[i] = r;
                }
            }
            Engine::Hydra(h) => h.advance(runs, which, p, target),
        }
    }
}

fn placeholder() -> CompRun {
    CompRun { m: Arc::new(Csc::empty(0)), t: 0, chaos: 0.0, frozen: true }
}

// ----------------------------------------------------------------------------
// HydraMPP
// ----------------------------------------------------------------------------

/// Task: advance a batch of (small) components serially.
type BatchIn = (MclParams, Option<usize>, Vec<CompRun>);
/// Task: step one column block of a shared matrix.
type BlockIn = (MclParams, Arc<Csc>, usize, usize);

fn task_batch((p, target, mut runs): BatchIn) -> Vec<CompRun> {
    for r in runs.iter_mut() {
        advance(r, &p, target, &SerialStep);
    }
    runs
}

fn task_block((p, m, lo, hi): BlockIn) -> Block {
    with_scratch(|s| kernel::step_block(&m, lo, hi, &p, s))
}

/// Register moa's task functions on a HydraMPP handle (host *and* workers must
/// register the same names).
pub fn register_tasks(h: &Hydra) {
    h.register("moa.batch", task_batch);
    h.register("moa.block", task_block);
}

pub struct HydraEngine {
    hydra: Hydra,
    pub cpus: usize,
    pub gpus: usize,
    pub nodes: usize,
    pub hostnames: Vec<String>,
    /// Components with at least this many columns are host-driven block jobs.
    pub big_cols: usize,
}

impl HydraEngine {
    /// Start HydraMPP with `cfg` (local by default; `--hydra-host`,
    /// `--hydra-slurm` and `--hydra-cpus` arrive through `Config::parse`).
    pub fn start(cfg: Config) -> Result<HydraEngine, String> {
        let hydra = Hydra::new();
        register_tasks(&hydra);
        hydra.init(cfg).map_err(|e| format!("HydraMPP init failed: {e}"))?;
        let nodes = hydra.nodes().map_err(|e| format!("HydraMPP node query failed: {e}"))?;
        let cpus = nodes.iter().map(|n| n.total_cpus).sum::<usize>().max(1);
        let gpus = nodes.iter().map(|n| n.total_gpus).sum();
        Ok(HydraEngine {
            hostnames: nodes.iter().map(|n| n.hostname.clone()).collect(),
            nodes: nodes.len(),
            cpus,
            gpus,
            hydra,
            big_cols: PAR_COLS,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "hydra · {} node{} · {} CPU slots{}",
            self.nodes,
            if self.nodes == 1 { "" } else { "s" },
            self.cpus,
            if self.gpus > 0 { format!(" · {} GPUs", self.gpus) } else { String::new() }
        )
    }

    fn advance(&self, runs: &mut [CompRun], which: &[usize], p: &MclParams, target: Option<usize>) {
        let (big, small): (Vec<usize>, Vec<usize>) = which
            .iter()
            .partition(|&&i| runs[i].m.n >= self.big_cols || (runs[i].m.n >= 64 && runs[i].m.nnz() >= PAR_NNZ));

        // Small components: pack into batches by nnz, ~4 batches per CPU slot.
        if !small.is_empty() {
            let mut order = small.clone();
            order.sort_by_key(|&i| std::cmp::Reverse(runs[i].m.nnz()));
            let total: usize = order.iter().map(|&i| runs[i].m.nnz() + runs[i].m.n).sum();
            // ~4 batches per CPU slot, but no batch lighter than ~64k entries
            // (per-task overhead is ~100 µs; a batch should dwarf it).
            let n_batches = (self.cpus * 4).min(total / 65_536 + 1).min(order.len()).max(1);
            // Greedy fill: largest first into the lightest bin keeps bins even.
            let mut bins: Vec<(usize, Vec<usize>)> = (0..n_batches).map(|_| (0, Vec::new())).collect();
            for &i in &order {
                let w = runs[i].m.nnz() + runs[i].m.n;
                let b = bins.iter_mut().min_by_key(|b| b.0).unwrap();
                b.0 += w;
                b.1.push(i);
            }
            let bins: Vec<Vec<usize>> = bins.into_iter().map(|b| b.1).filter(|b| !b.is_empty()).collect();
            let jobs: Vec<BatchIn> = bins
                .iter()
                .map(|ids| (*p, target, ids.iter().map(|&i| std::mem::replace(&mut runs[i], placeholder())).collect()))
                .collect();
            let out: Vec<Vec<CompRun>> = self.hydra.map_owned("moa.batch", jobs).expect("HydraMPP batch tasks failed");
            for (ids, rs) in bins.iter().zip(out) {
                for (&i, r) in ids.iter().zip(rs) {
                    runs[i] = r;
                }
            }
        }

        // Big components: host-driven, one block fan-out per iteration.
        for i in big {
            let step = HydraStep { h: self };
            advance(&mut runs[i], p, target, &step);
        }
    }
}

struct HydraStep<'a> {
    h: &'a HydraEngine,
}

// Hydra is Send + Sync (documented), so the step executor may be shared.
impl StepExec for HydraStep<'_> {
    fn step(&self, m: &Arc<Csc>, p: &MclParams) -> (Csc, f64) {
        let shared = Arc::clone(m); // no copy: tasks share the iterand
        let jobs: Vec<BlockIn> =
            split_cols(m, self.h.cpus * 4).into_iter().map(|(lo, hi)| (*p, Arc::clone(&shared), lo, hi)).collect();
        let blocks: Vec<Block> = self.h.hydra.map_owned("moa.block", jobs).expect("HydraMPP block tasks failed");
        kernel::stitch(m.n, blocks)
    }
}

impl Drop for HydraEngine {
    fn drop(&mut self) {
        self.hydra.shutdown();
    }
}

/// Become a HydraMPP worker node: register moa's tasks and serve the host until
/// it disconnects. Called when `--hydra-client <host>` is on the command line
/// (by hand, or by a `--hydra-slurm` head relaunching moa on each node).
pub fn serve_worker(cfg: Config) -> ! {
    let hydra = Hydra::new();
    register_tasks(&hydra);
    match hydra.init(cfg) {
        Ok(()) => std::process::exit(0),
        Err(e) => {
            eprintln!("moa worker: HydraMPP client failed: {e}");
            std::process::exit(1)
        }
    }
}
