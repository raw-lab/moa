//! HydraMPP expansion backend.
//!
//! Expansion is embarrassingly parallel across output columns, which is exactly
//! the shape HydraMPP likes: we split the columns into blocks, register a task
//! that expands one block, and fan the blocks out with [`Hydra::map_owned`].
//! On a laptop this runs on the local worker pool; point the same binary at a
//! host with `--hydra-client <ip>` and the identical task graph spreads across
//! the cluster — HydraMPP dispatches by function name, so every node runs the
//! same registered `moa.expand_block`.
//!
//! Because expanding *any* column needs the whole matrix (a column of `M*M` is
//! a linear combination of columns of `M`), each block job carries a copy of
//! the matrix. That broadcast is unavoidable for distributed MCL — on a real
//! cluster the operand has to reach every node — and on a local runtime
//! `map_owned` moves the copy in-memory with no serialisation round-trip.

use super::compose_column;
use crate::matrix::{Column, SparseMatrix};
use hydra_mpp_core::prelude::*;
use serde::{Deserialize, Serialize};

/// The unit of work shipped to a HydraMPP worker: expand columns `[lo, hi)` of
/// `mat`.
#[derive(Serialize, Deserialize)]
struct ExpandJob {
    mat: SparseMatrix,
    lo: usize,
    hi: usize,
}

/// Expand one block. Free function (not a closure) so it can be registered by
/// name on every node.
fn expand_block(job: ExpandJob) -> Vec<Column> {
    let ExpandJob { mat, lo, hi } = job;
    let mut scratch = vec![0.0f64; mat.n];
    let mut touched: Vec<u32> = Vec::new();
    (lo..hi)
        .map(|j| compose_column(&mat, j, &mut scratch, &mut touched))
        .collect()
}

pub struct HydraBackend {
    hydra: Hydra,
    /// How many column blocks to cut the matrix into per expansion — one per
    /// local CPU slot, so the worker pool stays saturated without over-cloning.
    blocks: usize,
    total_cpus: usize,
    total_gpus: usize,
    hostname: String,
}

impl HydraBackend {
    pub fn start() -> Result<HydraBackend, String> {
        let hydra = Hydra::new();
        hydra.register("moa.expand_block", expand_block);
        hydra
            .init(Config::local().quiet(true))
            .map_err(|e| format!("HydraMPP init failed: {e}"))?;
        let nodes = hydra
            .nodes()
            .map_err(|e| format!("HydraMPP node query failed: {e}"))?;
        let (total_cpus, total_gpus, hostname) = nodes
            .first()
            .map(|n| (n.total_cpus, n.total_gpus, n.hostname.clone()))
            .unwrap_or((1, 0, "local".into()));
        Ok(HydraBackend {
            hydra,
            blocks: total_cpus.max(1),
            total_cpus,
            total_gpus,
            hostname,
        })
    }

    pub fn describe(&self) -> String {
        format!(
            "HydraMPP local ({}: {} CPUs, {} GPUs, {} blocks/expansion)",
            self.hostname, self.total_cpus, self.total_gpus, self.blocks
        )
    }

    pub fn expand(&self, m: &SparseMatrix) -> SparseMatrix {
        let n = m.n;
        if n == 0 {
            return SparseMatrix::new(0);
        }
        let blocks = self.blocks.min(n).max(1);
        // Even column ranges across `blocks` jobs.
        let base = n / blocks;
        let rem = n % blocks;
        let mut jobs: Vec<ExpandJob> = Vec::with_capacity(blocks);
        let mut lo = 0usize;
        for b in 0..blocks {
            let len = base + if b < rem { 1 } else { 0 };
            let hi = lo + len;
            if hi > lo {
                jobs.push(ExpandJob {
                    mat: m.clone(),
                    lo,
                    hi,
                });
            }
            lo = hi;
        }
        // I/O/compute-bound blocks reserve one CPU slot each (cpus_each = 1 via
        // map_owned's default); results come back in submission (column) order.
        let results: Vec<Vec<Column>> = self
            .hydra
            .map_owned("moa.expand_block", jobs)
            .expect("HydraMPP expansion dispatch failed");
        let mut cols: Vec<Column> = Vec::with_capacity(n);
        for block in results {
            cols.extend(block);
        }
        debug_assert_eq!(cols.len(), n);
        SparseMatrix { n, cols }
    }
}

impl Drop for HydraBackend {
    fn drop(&mut self) {
        self.hydra.shutdown();
    }
}
