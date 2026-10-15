//! # Moa
//!
//! A faithful, 100% Rust port of Markov clustering (MCL) with a MinHash
//! ortholog-graph front-end and four interchangeable expansion engines
//! (serial, rayon, HydraMPP, GPU).
//!
//! * [`mcl`] — the expansion/inflation/prune loop and the chaos criterion.
//! * [`interpret`] — turning the fixed point into clusters.
//! * [`matrix`] — the CSC sparse matrix MCL runs on.
//! * [`backend`] — the four expansion engines.
//! * [`minhash`] — alignment-free similarity graphs from sequences.
//! * [`art`] — the mascot and banners.

pub mod art;
pub mod backend;
pub mod interpret;
pub mod matrix;
pub mod mcl;
pub mod minhash;

pub use backend::{Backend, Engine};
pub use matrix::SparseMatrix;
pub use mcl::{MclParams, MclResult};

/// The canonical MCL "friends" example (19 undirected, unweighted edges over
/// 12 people). At `-I 2.0` it splits into three friend groups; at `-I 1.4` it
/// coarsens to two. Embedded so `doctor` and the test-suite need no data files.
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

/// A graph parsed from a labelled edge stream.
pub struct LabeledGraph {
    /// Node labels, indexed by node id (first-seen order).
    pub labels: Vec<String>,
    /// Undirected weighted edges over node ids.
    pub edges: Vec<(u32, u32, f64)>,
}

/// Parse an `--abc`-style edge stream: one edge per line as
/// `labelA labelB [weight]`, whitespace- or tab-separated. A missing weight is
/// 1.0. Labels are interned in first-seen order. Blank lines and `#` comments
/// are ignored.
pub fn parse_labeled_graph(text: &str) -> LabeledGraph {
    use std::collections::HashMap;
    let mut index: HashMap<String, u32> = HashMap::new();
    let mut labels: Vec<String> = Vec::new();
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();

    let intern = |s: &str, labels: &mut Vec<String>, index: &mut HashMap<String, u32>| -> u32 {
        if let Some(&id) = index.get(s) {
            id
        } else {
            let id = labels.len() as u32;
            labels.push(s.to_string());
            index.insert(s.to_string(), id);
            id
        }
    };

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let (a, b) = match (it.next(), it.next()) {
            (Some(a), Some(b)) => (a, b),
            _ => continue, // a lone token is not an edge
        };
        let w = it.next().and_then(|x| x.parse::<f64>().ok()).unwrap_or(1.0);
        let ai = intern(a, &mut labels, &mut index);
        let bi = intern(b, &mut labels, &mut index);
        edges.push((ai, bi, w));
    }
    LabeledGraph { labels, edges }
}

/// Cluster a labelled graph end-to-end; returns clusters as label lists plus
/// diagnostics.
pub fn cluster_labeled_graph(
    text: &str,
    params: &MclParams,
    backend: &Backend,
) -> (Vec<Vec<String>>, GraphStats) {
    let g = parse_labeled_graph(text);
    let n = g.labels.len();
    let m = SparseMatrix::from_edges(n, &g.edges);
    let nnz = m.nnz();
    let res = mcl::run(m, params, backend);
    let clusters = res
        .clusters
        .iter()
        .map(|c| c.iter().map(|&i| g.labels[i as usize].clone()).collect())
        .collect();
    let stats = GraphStats {
        n_nodes: n,
        n_edges: g.edges.len(),
        nnz,
        iterations: res.iterations,
        final_chaos: res.final_chaos,
    };
    (clusters, stats)
}

/// Diagnostics returned alongside a clustering.
pub struct GraphStats {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub nnz: usize,
    pub iterations: usize,
    pub final_chaos: f64,
}

/// Result of clustering sequences.
pub struct SequenceClustering {
    /// Clusters as sequence-id lists.
    pub clusters: Vec<Vec<String>>,
    pub n_seqs: usize,
    pub n_edges: usize,
    pub iterations: usize,
    pub final_chaos: f64,
}

/// Cluster protein/nucleotide sequences (FASTA text) with MinHash + MCL.
pub fn cluster_sequences(
    fasta: &str,
    gp: &minhash::GraphParams,
    mp: &MclParams,
    backend: &Backend,
) -> SequenceClustering {
    let (ids, seqs) = minhash::read_fasta(fasta);
    let n = ids.len();
    let lengths: Vec<u32> = seqs.iter().map(|s| s.len() as u32).collect();
    let sketches = minhash::sketch_many(&seqs, gp.sketch);
    let edges = minhash::build_graph(&sketches, &lengths, gp);
    let m = SparseMatrix::from_edges(n, &edges);
    let res = mcl::run(m, mp, backend);
    let clusters = res
        .clusters
        .iter()
        .map(|c| c.iter().map(|&i| ids[i as usize].clone()).collect())
        .collect();
    SequenceClustering {
        clusters,
        n_seqs: n,
        n_edges: edges.len(),
        iterations: res.iterations,
        final_chaos: res.final_chaos,
    }
}
