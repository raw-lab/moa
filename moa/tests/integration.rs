//! End-to-end tests.
//!
//! The centrepiece is the reference "friends" proof: at inflation 2.0 the graph
//! must split into exactly the three named friend groups, and the chaos measure
//! must fall to zero. The rest pin down granularity behaviour, the algebraic
//! edge cases, backend equivalence, and the MinHash front-end.
//!
//! Only the `all_engines_agree` test touches the HydraMPP / GPU runtimes, and it
//! does so sequentially, so the suite is safe to run in parallel.

use moa_core::backend::{Backend, Engine};
use moa_core::minhash::{GraphParams, SketchParams};
use moa_core::{cluster_labeled_graph, cluster_sequences, MclParams, SparseMatrix};
use std::collections::BTreeSet;

fn serial() -> Backend {
    Backend::start(Engine::Serial).unwrap()
}

/// Cluster the embedded friends graph at a given inflation with the serial
/// engine, returning a canonical set-of-sets.
fn friends(inflation: f64) -> BTreeSet<BTreeSet<String>> {
    let p = MclParams {
        inflation,
        ..MclParams::default()
    };
    let (clusters, _) = cluster_labeled_graph(moa_core::FRIENDS_ABC, &p, &serial());
    clusters
        .into_iter()
        .map(|c| c.into_iter().collect())
        .collect()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| s.to_string()).collect()
}

#[test]
fn friends_splits_into_three_named_clusters() {
    // The canonical MCL result at -I 2.0 (van Dongen's manual).
    let got = friends(2.0);
    let expected: BTreeSet<BTreeSet<String>> = [
        set(&["ann", "fred", "john", "gilian"]),
        set(&["bob", "charlie", "esme"]),
        set(&["dan", "harriet", "indy", "kelly", "lucy"]),
    ]
    .into_iter()
    .collect();
    assert_eq!(got, expected, "friends -I 2.0 must give the three named groups");
}

#[test]
fn friends_coarsens_to_two_at_low_inflation() {
    let got = friends(1.4);
    assert_eq!(got.len(), 2, "friends -I 1.4 should coarsen to two groups");
}

#[test]
fn friends_chaos_converges_to_zero() {
    let (_c, stats) = cluster_labeled_graph(moa_core::FRIENDS_ABC, &MclParams::default(), &serial());
    assert!(
        stats.final_chaos <= MclParams::default().chaos_limit,
        "chaos {} must reach the limit",
        stats.final_chaos
    );
    // A clean fixed point has essentially zero chaos.
    assert!(stats.final_chaos < 1e-3, "chaos should be ~0, got {}", stats.final_chaos);
    assert!(stats.iterations >= 1 && stats.iterations < 100);
}

#[test]
fn higher_inflation_never_yields_fewer_clusters() {
    let mut last = 0usize;
    for &i in &[1.2f64, 1.4, 2.0, 3.0, 5.0] {
        let n = friends(i).len();
        assert!(
            n >= last,
            "cluster count should be monotone in inflation: {i} gave {n} after {last}"
        );
        last = n;
    }
    assert!(last >= 5, "at -I 5.0 the friends graph should be quite fine-grained");
}

#[test]
fn clique_is_a_single_cluster() {
    let e = [(0u32, 1u32, 1.0f64), (0, 2, 1.0), (1, 2, 1.0), (0, 3, 1.0), (1, 3, 1.0), (2, 3, 1.0)];
    let m = SparseMatrix::from_edges(4, &e);
    let r = moa_core::mcl::run(m, &MclParams::default(), &serial());
    assert_eq!(r.clusters.len(), 1);
}

#[test]
fn two_cliques_split_on_a_weak_bridge() {
    let mut e = vec![];
    for (a, b) in [(0, 1), (0, 2), (1, 2)] {
        e.push((a, b, 100.0));
    }
    for (a, b) in [(3, 4), (3, 5), (4, 5)] {
        e.push((a, b, 100.0));
    }
    e.push((2, 3, 0.01)); // faint bridge
    let m = SparseMatrix::from_edges(6, &e);
    let r = moa_core::mcl::run(m, &MclParams::default(), &serial());
    assert_eq!(r.clusters.len(), 2, "weakly bridged cliques must separate");
}

#[test]
fn isolated_nodes_become_singletons() {
    // Two edges plus two isolated nodes over n = 6.
    let e = [(0u32, 1u32, 1.0f64), (2, 3, 1.0)];
    let m = SparseMatrix::from_edges(6, &e);
    let r = moa_core::mcl::run(m, &MclParams::default(), &serial());
    // {0,1}, {2,3}, {4}, {5}
    assert_eq!(r.clusters.len(), 4);
    assert!(r.clusters.iter().any(|c| c == &vec![4]));
    assert!(r.clusters.iter().any(|c| c == &vec![5]));
}

#[test]
fn weighted_edge_text_parses_and_clusters() {
    // Tab- and space-separated, with a weight column and a comment.
    let text = "# a toy graph\nx\ty\t5\nx z 5\ny z 5\np q 9\n";
    let (clusters, stats) = cluster_labeled_graph(text, &MclParams::default(), &serial());
    assert_eq!(stats.n_nodes, 5); // x y z p q
    // {x,y,z} and {p,q}
    assert_eq!(clusters.len(), 2);
    assert!(clusters.iter().any(|c| c.contains(&"p".to_string()) && c.len() == 2));
}

#[test]
fn all_engines_agree_on_friends() {
    // Serial is the reference; rayon and HydraMPP are exact (f64); GPU is f32 but
    // must still land the same partition. GPU/Hydra unavailability is tolerated.
    let reference = friends(2.0);

    let rayon = {
        let (c, _) = cluster_labeled_graph(
            moa_core::FRIENDS_ABC,
            &MclParams::default(),
            &Backend::start(Engine::Rayon).unwrap(),
        );
        c.into_iter().map(|c| c.into_iter().collect::<BTreeSet<_>>()).collect::<BTreeSet<_>>()
    };
    assert_eq!(rayon, reference, "rayon must match serial exactly");

    if let Ok(hydra) = Backend::start(Engine::Hydra) {
        let (c, _) = cluster_labeled_graph(moa_core::FRIENDS_ABC, &MclParams::default(), &hydra);
        let part: BTreeSet<BTreeSet<String>> =
            c.into_iter().map(|c| c.into_iter().collect()).collect();
        assert_eq!(part, reference, "HydraMPP must match serial exactly");
    }

    if let Ok(gpu) = Backend::start(Engine::Gpu) {
        let (c, _) = cluster_labeled_graph(moa_core::FRIENDS_ABC, &MclParams::default(), &gpu);
        let part: BTreeSet<BTreeSet<String>> =
            c.into_iter().map(|c| c.into_iter().collect()).collect();
        assert_eq!(part, reference, "GPU must land the same partition as serial");
    }
}

#[test]
fn minhash_groups_sequence_families() {
    // Two families: three near-identical variants of A, two of B, plus one
    // unrelated singleton. Each variant differs by a couple of residues.
    let fasta = "\
>a1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR
>a2
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKS
>a3
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKT
>b1
GSHMLEDPYERVLSAWAFDVAKLNSVDGKQVQFVTGKPGSGKSTLARELAKQLDVPFYDLDDLIEEALRDRLTRSGDD
>b2
GSHMLEDPYERVLSAWAFDVAKLNSVDGKQVQFVTGKPGSGKSTLARELAKQLDVPFYDLDDLIEEALRDRLTRSGDE
>c1
WWWWCCCCDDDDEEEEFFFFGGGGHHHHIIIIKKKKLLLLMMMMNNNNPPPPQQQQRRRRSSSSTTTTVVVVYYYYAAAA
";
    let gp = GraphParams {
        sketch: SketchParams { k: 7, s: 128 },
        min_id: 0.90,
        len_cov: 0.5,
        ..GraphParams::default()
    };
    let r = cluster_sequences(fasta, &gp, &MclParams::default(), &serial());
    assert_eq!(r.n_seqs, 6);

    // a1,a2,a3 must share a cluster; b1,b2 must share a cluster; c1 stands alone.
    let cluster_of = |id: &str| -> Vec<String> {
        r.clusters
            .iter()
            .find(|c| c.iter().any(|x| x == id))
            .cloned()
            .unwrap_or_default()
    };
    let a = cluster_of("a1");
    assert!(a.contains(&"a2".to_string()) && a.contains(&"a3".to_string()), "A family: {a:?}");
    let b = cluster_of("b1");
    assert!(b.contains(&"b2".to_string()), "B family: {b:?}");
    assert!(!a.contains(&"b1".to_string()), "families must not merge");
    assert_eq!(cluster_of("c1"), vec!["c1".to_string()], "unrelated seq is a singleton");
}
