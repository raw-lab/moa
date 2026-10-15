//! Expansion engines.
//!
//! The MCL inner loop spends essentially all of its time in **expansion** —
//! the matrix square `M := M*M`. Everything else (inflation, pruning, the
//! chaos measure) is linear in the number of stored entries and trivially
//! cheap by comparison. So expansion is the one operation Moa parallelises,
//! and it offers four interchangeable backends that all compute the *same*
//! raw product:
//!
//! * [`Engine::Serial`] — one thread; the reference against which the others
//!   are checked.
//! * [`Engine::Rayon`] — data-parallel across columns on the local cores.
//! * [`Engine::Hydra`] — column blocks farmed out through **HydraMPP**, the
//!   same task engine that scales from a laptop to a whole SLURM allocation.
//! * [`Engine::Gpu`] — a dense `M*M` compute shader run through **wgpu** on any
//!   Vulkan / Metal / DX12 / GL adapter (software `lavapipe` included).
//!
//! Expansion is a pure function of the input matrix, so swapping the engine can
//! never change the clustering (up to GPU single-precision rounding). That is
//! what lets `moa doctor` cross-check every backend against `Serial` on a live
//! graph.

use crate::matrix::{Column, SparseMatrix};

mod cpu;
mod hydra;
#[cfg(feature = "gpu")]
mod gpu;

/// Which expansion engine to use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    /// Single-threaded. Always available; the correctness reference.
    Serial,
    /// Data-parallel over columns with rayon.
    Rayon,
    /// Distributed column blocks via HydraMPP.
    Hydra,
    /// Dense `M*M` on the GPU via wgpu.
    Gpu,
}

impl Engine {
    pub fn as_str(self) -> &'static str {
        match self {
            Engine::Serial => "serial",
            Engine::Rayon => "rayon",
            Engine::Hydra => "hydra",
            Engine::Gpu => "gpu",
        }
    }

    pub fn parse(s: &str) -> Option<Engine> {
        match s.to_ascii_lowercase().as_str() {
            "serial" | "seq" => Some(Engine::Serial),
            "rayon" | "cpu" | "auto" => Some(Engine::Rayon),
            "hydra" | "hydrampp" | "mpp" => Some(Engine::Hydra),
            "gpu" | "wgpu" => Some(Engine::Gpu),
            _ => None,
        }
    }
}

/// A live expansion backend, holding whatever runtime state the engine needs
/// (a HydraMPP runtime, a wgpu device, …). Build one with [`Backend::start`],
/// hand it to [`crate::mcl::run`], and drop it when done.
pub enum Backend {
    Serial,
    Rayon,
    Hydra(hydra::HydraBackend),
    #[cfg(feature = "gpu")]
    Gpu(gpu::GpuBackend),
}

impl Backend {
    /// Start the requested engine, returning an error string if it is
    /// unavailable in this environment (no GPU adapter, HydraMPP init failure,
    /// GPU support compiled out, …).
    pub fn start(engine: Engine) -> Result<Backend, String> {
        match engine {
            Engine::Serial => Ok(Backend::Serial),
            Engine::Rayon => Ok(Backend::Rayon),
            Engine::Hydra => Ok(Backend::Hydra(hydra::HydraBackend::start()?)),
            #[cfg(feature = "gpu")]
            Engine::Gpu => Ok(Backend::Gpu(gpu::GpuBackend::start()?)),
            #[cfg(not(feature = "gpu"))]
            Engine::Gpu => Err("GPU support was not compiled in (build without \
                                `--no-default-features`, or with `--features gpu`)"
                .into()),
        }
    }

    /// A human-readable one-liner about the running backend (thread count,
    /// device name, …), for banners and `doctor`.
    pub fn describe(&self) -> String {
        match self {
            Backend::Serial => "serial (1 thread)".to_string(),
            Backend::Rayon => format!("rayon ({} threads)", rayon::current_num_threads()),
            Backend::Hydra(h) => h.describe(),
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.describe(),
        }
    }

    pub fn engine(&self) -> Engine {
        match self {
            Backend::Serial => Engine::Serial,
            Backend::Rayon => Engine::Rayon,
            Backend::Hydra(_) => Engine::Hydra,
            #[cfg(feature = "gpu")]
            Backend::Gpu(_) => Engine::Gpu,
        }
    }

    /// Compute the raw expansion `M*M` (no pruning) using this backend. The
    /// result is column-major, each column sorted by row index — identical
    /// across backends, save for GPU single-precision rounding.
    pub fn raw_expand(&self, m: &SparseMatrix) -> SparseMatrix {
        match self {
            Backend::Serial => cpu::expand_serial(m),
            Backend::Rayon => cpu::expand_rayon(m),
            Backend::Hydra(h) => h.expand(m),
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.expand(m),
        }
    }
}

/// The one hot kernel, shared by every CPU-side backend: expand a single
/// column, i.e. `out[:, j] = sum_k M[k][j] * M[:, k]`. This is the sparse
/// column compose of `impala/compose.c` / `expand.c:mclxVectorCompose`.
///
/// Accumulation is done in a dense scratch buffer of length `n`, which is O(n)
/// memory but touches each nonzero exactly once and avoids per-insert hashing;
/// `touched` records which rows were written so the scratch can be read out and
/// reset without scanning all `n` slots.
#[inline]
pub(crate) fn compose_column(
    m: &SparseMatrix,
    j: usize,
    scratch: &mut [f64],
    touched: &mut Vec<u32>,
) -> Column {
    touched.clear();
    for &(k, wkj) in m.cols[j].iter() {
        if wkj == 0.0 {
            continue;
        }
        for &(i, wik) in m.cols[k as usize].iter() {
            let slot = &mut scratch[i as usize];
            if *slot == 0.0 {
                touched.push(i);
            }
            *slot += wik * wkj;
        }
    }
    touched.sort_unstable();
    let mut out: Column = Vec::with_capacity(touched.len());
    for &i in touched.iter() {
        let v = scratch[i as usize];
        scratch[i as usize] = 0.0; // reset for the next column
        if v != 0.0 {
            out.push((i, v));
        }
    }
    out
}
