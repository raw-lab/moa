//! # moa — reference-faithful, parallel Markov clustering
//!
//! A from-scratch Rust implementation of Stijn van Dongen's **MCL** that
//! reproduces `mcl 22-282` partition-for-partition, plus an alignment-free
//! **MinHash** front-end that turns sequences into the graph MCL consumes.
//!
//! | module        | what it does                                                  |
//! |---------------|---------------------------------------------------------------|
//! | [`graph`]     | `--abc` parsing, max-merge symmetrisation, components, CSC    |
//! | [`kernel`]    | the fused compose → prune/select/recover → inflate column step |
//! | [`mcl`]       | parameters, per-component iteration, global stop reconciling  |
//! | [`interpret`] | `mclDag` + `mclInterpret` + `-overlap cut`                    |
//! | [`engine`]    | serial · rayon · HydraMPP execution                           |
//! | [`minhash`]   | bottom-s sketches, linear candidate index, FASTA              |
//!
//! ```
//! use moa_core::{cluster_abc, MclParams, Engine};
//! let (labels, out) = cluster_abc(moa_core::FRIENDS_ABC.as_bytes(), &MclParams::default(), &Engine::Rayon).unwrap();
//! assert_eq!(out.clusters.len(), 3);
//! # let _ = labels;
//! ```

pub mod art;
pub mod engine;
pub mod graph;
pub mod interpret;
pub mod kernel;
pub mod mcl;
pub mod minhash;
pub mod regress;
pub mod synth;
pub mod util;

pub use engine::Engine;
pub use graph::{Csc, Graph};
pub use mcl::{MclOutput, MclParams};
pub use minhash::{GraphParams, SketchParams};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The reference implementation moa is validated against.
pub const REFERENCE: &str = "mcl 22-282";

/// The canonical MCL "friends" example (van Dongen). `-I 2.0` → 3 groups,
/// `-I 1.4` → 2 groups. Embedded so `doctor` and the tests need no files.
pub const FRIENDS_ABC: &str = "\
ann bob
ann fred
ann john
ann gilian
bob charlie
bob esme
charlie esme
charlie dan
dan harriet
dan kelly
dan indy
esme gilian
esme harriet
fred john
gilian john
harriet kelly
harriet indy
indy kelly
indy lucy
kelly lucy
";

/// Parse an `--abc` stream and cluster it. Returns node labels (by id) and
/// the result; cluster members are ids into `labels`.
pub fn cluster_abc(text: &[u8], p: &MclParams, engine: &Engine) -> Result<(Vec<String>, MclOutput), String> {
    p.validate()?;
    let lab = graph::parse_abc(text)?;
    let g = Graph::from_edges(lab.labels.len(), &lab.edges);
    Ok((lab.labels, mcl::run(g, p, engine)))
}

/// Everything `moa cluster` produces.
pub struct SequenceClustering {
    pub ids: Vec<String>,
    pub edges: Vec<(u32, u32, f64)>,
    pub out: MclOutput,
    pub secs_sketch: f64,
    pub secs_graph: f64,
}

/// MinHash graph + MCL over FASTA text.
pub fn cluster_fasta(
    fasta: &[u8],
    gp: &GraphParams,
    p: &MclParams,
    engine: &Engine,
) -> Result<SequenceClustering, String> {
    p.validate()?;
    let f = minhash::read_fasta(fasta)?;
    let t0 = std::time::Instant::now();
    let sketches = minhash::sketch_many(&f.seqs, gp.sketch);
    let lengths: Vec<u32> = f.seqs.iter().map(|s| s.len() as u32).collect();
    let secs_sketch = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let edges = minhash::build_graph(&sketches, &lengths, gp);
    let secs_graph = t1.elapsed().as_secs_f64();
    let g = Graph::from_edges(f.ids.len(), &edges);
    let out = mcl::run(g, p, engine);
    Ok(SequenceClustering { ids: f.ids, edges, out, secs_sketch, secs_graph })
}
