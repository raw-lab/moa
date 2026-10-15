//! End-to-end tests. Expected partitions marked "mcl" were recorded from the
//! reference `mcl 22-282` binary.

use moa_core::engine::{Engine, HydraEngine};
use moa_core::graph::{parse_abc, Csc};
use moa_core::mcl::{self, CompRun, MclParams};
use moa_core::{cluster_abc, interpret, Graph, FRIENDS_ABC};
use std::collections::BTreeSet;

type Partition = BTreeSet<BTreeSet<String>>;

fn part(labels: &[String], clusters: &[Vec<u32>]) -> Partition {
    clusters.iter().map(|c| c.iter().map(|&m| labels[m as usize].clone()).collect()).collect()
}
fn set(xs: &[&str]) -> BTreeSet<String> {
    xs.iter().map(|s| s.to_string()).collect()
}
fn render(labels: &[String], clusters: &[Vec<u32>]) -> String {
    clusters
        .iter()
        .map(|c| c.iter().map(|&m| labels[m as usize].as_str()).collect::<Vec<_>>().join("\t") + "\n")
        .collect()
}

// --- the friends proof ------------------------------------------------------------

#[test]
fn friends_three_named_groups_at_i2() {
    let (lab, out) = cluster_abc(FRIENDS_ABC.as_bytes(), &MclParams::default(), &Engine::Serial).unwrap();
    let want: Partition = [
        set(&["ann", "fred", "john", "gilian"]),
        set(&["bob", "charlie", "esme"]),
        set(&["dan", "harriet", "indy", "kelly", "lucy"]),
    ]
    .into_iter()
    .collect();
    assert_eq!(part(&lab, &out.clusters), want);
    assert!(out.converged && out.chaos < 1e-4);
}

#[test]
fn friends_output_is_byte_identical_to_mcl() {
    // `mcl friends --abc -o -` (mcl 22-282): order of clusters and members.
    let (lab, out) = cluster_abc(FRIENDS_ABC.as_bytes(), &MclParams::default(), &Engine::Rayon).unwrap();
    assert_eq!(
        render(&lab, &out.clusters),
        "dan\tharriet\tkelly\tindy\tlucy\nann\tfred\tjohn\tgilian\nbob\tcharlie\tesme\n"
    );
}

#[test]
fn friends_coarsens_to_two_at_i14() {
    let p = MclParams { inflation: 1.4, ..Default::default() };
    let (_, out) = cluster_abc(FRIENDS_ABC.as_bytes(), &p, &Engine::Serial).unwrap();
    assert_eq!(out.clusters.len(), 2);
}

#[test]
fn inflation_never_coarsens_friends() {
    let mut last = 0;
    for i in [1.2, 1.4, 2.0, 3.0, 5.0] {
        let p = MclParams { inflation: i, ..Default::default() };
        let n = cluster_abc(FRIENDS_ABC.as_bytes(), &p, &Engine::Serial).unwrap().1.clusters.len();
        assert!(n >= last, "-I {i}: {n} < {last}");
        last = n;
    }
}

// --- the regression corpus (every bug class that 1.x got wrong) --------------------

#[test]
fn regression_corpus_matches_mcl() {
    for c in moa_core::regress::CASES {
        let p = MclParams {
            inflation: c.inflation,
            prune: c.prune,
            select: c.select,
            recover: c.recover,
            ..Default::default()
        };
        let (lab, out) = cluster_abc(c.abc.as_bytes(), &p, &Engine::Serial).unwrap();
        let want: Partition = c.expected.iter().map(|cl| cl.iter().map(|s| s.to_string()).collect()).collect();
        assert_eq!(part(&lab, &out.clusters), want, "case: {}", c.name);
    }
}

// --- engines ------------------------------------------------------------------------

#[test]
fn engines_are_bit_identical() {
    // One component above PAR_COLS (column-parallel + Hydra block path) plus
    // hundreds of small ones (component-parallel + Hydra batch path).
    let (n, e) = moa_core::synth::synthetic(42, 2600, 300);
    let g = Graph::from_edges(n, &e);
    let p = MclParams::default();
    let a = mcl::run(g.clone(), &p, &Engine::Serial);
    let b = mcl::run(g.clone(), &p, &Engine::Rayon);
    let h = Engine::Hydra(HydraEngine::start(hydra_mpp::Config::local().quiet(true)).unwrap());
    let c = mcl::run(g, &p, &h);
    assert_eq!(a.clusters, b.clusters);
    assert_eq!(a.clusters, c.clusters);
    assert_eq!((a.iterations, a.chaos.to_bits()), (b.iterations, b.chaos.to_bits()));
    assert_eq!((a.iterations, a.chaos.to_bits()), (c.iterations, c.chaos.to_bits()));
}

#[test]
fn per_component_equals_whole_graph() {
    // Iterate the WHOLE matrix as one unit (mcl's way) and compare with moa's
    // per-component run + global-stop reconciliation.
    let (n, e) = moa_core::synth::synthetic(7, 900, 400);
    let g = Graph::from_edges(n, &e);
    let p = MclParams::default();
    let mut whole = CompRun::start(g.adj.clone());
    mcl::advance(&mut whole, &p, None, &moa_core::engine::SerialStep);
    let mut wc = interpret::interpret(&whole.m).clusters;
    mcl::order_clusters(&mut wc);
    let per = mcl::run(g, &p, &Engine::Rayon);
    assert_eq!(per.iterations, whole.t, "global iteration count");
    assert_eq!(per.clusters, wc);
}

// --- input conventions (mcl --abc) ---------------------------------------------------

#[test]
fn duplicates_and_reverse_edges_merge_by_max() {
    let lab = parse_abc(b"a b 3\nb a 5\na b 1\nb c 2\n").unwrap();
    let g = Graph::from_edges(lab.labels.len(), &lab.edges);
    assert_eq!(g.adj.get(1, 0), 5.0); // (b,a)
    assert_eq!(g.adj.get(0, 1), 5.0); // (a,b)
    assert_eq!(g.adj.get(2, 1), 2.0);
    assert_eq!(g.n_edges, 2);
}

#[test]
fn input_loops_are_reset_to_column_max() {
    let lab = parse_abc(b"a a 100\na b 2\nb c 4\n").unwrap();
    let g = Graph::from_edges(lab.labels.len(), &lab.edges);
    let s = g.adj.into_start_matrix();
    // column a: loop := max off-diagonal = 2 → [a:2, b:2]/4
    assert_eq!(s.get(0, 0), 0.5);
    assert_eq!(s.get(1, 0), 0.5);
}

#[test]
fn bad_weights_are_rejected_or_ignored() {
    assert!(parse_abc(b"a b 1\nc d 1.5x\n").unwrap_err().contains("line 2"));
    assert!(parse_abc(b"lonely\n").unwrap_err().contains("line 1"));
    let lab = parse_abc(b"# comment\n\na b -1\nc d 0\ne f nan\ng h inf\ni j 2\n").unwrap();
    let g = Graph::from_edges(lab.labels.len(), &lab.edges);
    assert_eq!(g.n_edges, 1, "only i-j survives");
}

#[test]
fn whitespace_and_default_weight() {
    let lab = parse_abc(b"x\t y\nx   z   2.5\r\n").unwrap();
    assert_eq!(lab.labels, vec!["x", "y", "z"]);
    assert_eq!(lab.edges[0].2, 1.0);
    assert_eq!(lab.edges[1].2, 2.5);
}

// --- interpretation --------------------------------------------------------------------

fn csc(n: usize, cols: &[&[(u32, f32)]]) -> Csc {
    let mut ptr = vec![0];
    let (mut idx, mut val) = (vec![], vec![]);
    for c in cols {
        for &(i, v) in *c {
            idx.push(i);
            val.push(v);
        }
        ptr.push(idx.len());
    }
    Csc { n, ptr, idx, val }
}

#[test]
fn overlap_is_cut_to_the_larger_system_not_merged() {
    // Attractors {0,1} (system A, size 2) and {3} (system B). Node 2 flows
    // evenly into both. mcl: node 2 → A (first = largest system); A, B stay apart.
    let m = csc(4, &[&[(0, 0.5), (1, 0.5)], &[(0, 0.5), (1, 0.5)], &[(1, 0.5), (3, 0.5)], &[(3, 1.0)]]);
    let r = interpret::interpret(&m);
    assert_eq!(r.clusters, vec![vec![0, 1, 2], vec![3]]);
    assert_eq!(r.overlap, 1);
}

#[test]
fn nodes_reaching_no_attractor_are_gathered() {
    // 2 → 3 → 2 (cycle of non-attractors), 0 and 1 are attractors.
    let m = csc(4, &[&[(0, 1.0)], &[(1, 1.0)], &[(3, 1.0)], &[(2, 1.0)]]);
    let r = interpret::interpret(&m);
    assert_eq!(r.truants, vec![2, 3]);
}

// --- MinHash / FASTA ---------------------------------------------------------------

#[test]
fn fasta_reader_normalises_input() {
    let f = moa_core::minhash::read_fasta(b"junk before\n>p1 desc\nmkta*\nYI-AK\r\n>\nWW\n").unwrap();
    assert_eq!(f.ids, vec!["p1", "seq2"]);
    assert_eq!(f.seqs[0], b"MKTAYIAK");
    assert_eq!(f.seqs[1], b"WW");
    assert!(moa_core::minhash::read_fasta(b"no headers").is_err());
}

#[test]
fn rolling_sketch_equals_naive_and_k0_is_safe() {
    use moa_core::minhash::{sketch, SketchParams};
    let seq: Vec<u8> =
        (0..5000u32).map(|i| b"ACDEFGHIKLMNPQRSTVWY"[(moa_core::util::mix64(i as u64) % 20) as usize]).collect();
    for k in [1, 5, 7, 13, 21] {
        let p = SketchParams { k, s: 64 };
        let mut naive: Vec<u64> = seq
            .windows(k)
            .map(|w| moa_core::util::mix64(w.iter().fold(0u64, |a, &b| a.wrapping_mul(31).wrapping_add(b as u64))))
            .collect();
        naive.sort_unstable();
        naive.dedup();
        naive.truncate(64);
        assert_eq!(sketch(&seq, p), naive, "k={k}");
    }
    assert!(sketch(b"MKT", SketchParams { k: 0, s: 8 }).is_empty());
}

#[test]
fn sequence_families_are_recovered() {
    let demo = std::fs::read("examples/demo.faa").unwrap();
    let gp = moa_core::GraphParams { min_id: 0.90, ..Default::default() };
    let r = moa_core::cluster_fasta(&demo, &gp, &MclParams::default(), &Engine::Rayon).unwrap();
    let p = part(&r.ids, &r.out.clusters);
    assert_eq!(p.len(), 4);
    assert!(p.contains(&set(&["orfA_1", "orfA_2", "orfA_3"])));
    assert!(p.contains(&set(&["lonely_1"])));
}

// --- utilities ------------------------------------------------------------------------

#[test]
fn label_hash_spreads_low_bits() {
    // Regression: shared-prefix labels must not collide in hashbrown's low bits.
    use std::hash::BuildHasher;
    let b = moa_core::util::FxBuild::default();
    // A perfect hash fills ~2589 of 4096 buckets with 4096 keys (1 − 1/e).
    for prefix in ["gene_", "NZ_CP012345.1_", "p"] {
        let lows: BTreeSet<u64> = (0..4096).map(|i| b.hash_one(format!("{prefix}{i}").as_bytes()) & 0xfff).collect();
        assert!(lows.len() > 2400, "{prefix}: only {} of 4096 low-bit buckets used", lows.len());
    }
}

#[test]
fn large_family_is_one_cluster_and_families_stay_apart() {
    // Regression for 1.1.0: a ring-only candidate graph kept a big family
    // connected but MCL fragmented it (1,500 members → 1,358 clusters).
    let fa = moa_core::synth::family(700, 300, 11);
    let fb = moa_core::synth::family(700, 300, 12);
    let mut fasta = String::new();
    for (i, s) in fa.iter().enumerate() {
        fasta += &format!(">A_{i}\n{}\n", String::from_utf8_lossy(s));
    }
    for (i, s) in fb.iter().enumerate() {
        fasta += &format!(">B_{i}\n{}\n", String::from_utf8_lossy(s));
    }
    let r = moa_core::cluster_fasta(
        fasta.as_bytes(),
        &moa_core::GraphParams::default(),
        &MclParams::default(),
        &Engine::Rayon,
    )
    .unwrap();
    assert_eq!(r.out.clusters.len(), 2, "two families expected, got {}", r.out.clusters.len());
    for c in &r.out.clusters {
        let fam: BTreeSet<&str> = c.iter().map(|&m| &r.ids[m as usize][..1]).collect();
        assert_eq!(fam.len(), 1, "families merged");
        assert_eq!(c.len(), 700);
    }
}
