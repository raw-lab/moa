//! Bottom-`s` MinHash sketching and similarity-graph construction.
//!
//! Turns a set of protein (or nucleotide) sequences into the weighted graph MCL
//! consumes, without any alignment. Two sequences get an edge when their
//! estimated k-mer Jaccard clears a threshold derived from a target identity.
//!
//! The estimator's variance matters: with a bottom-`s` sketch the standard
//! error of a Jaccard estimate near `J` is `sqrt(J(1-J)/s)`. At `s = 48` and
//! `J ~ 0.54` that is +/-0.072, so a *true* ortholog sitting near the threshold
//! is rejected roughly half the time. Moa therefore defaults to `s = 256`,
//! where the same pair is rejected under 2% of the time. Candidate pairs are
//! gathered from shared sketch hashes rather than compared all-against-all.

use rayon::prelude::*;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
pub struct SketchParams {
    pub k: usize,
    pub s: usize,
}

impl Default for SketchParams {
    fn default() -> Self {
        SketchParams { k: 7, s: 256 }
    }
}

/// SplitMix64 — a fast, well-distributed finaliser.
#[inline]
fn hash64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Bottom-`s` sketch of the k-mers of `seq`: the `s` smallest distinct hashes,
/// sorted ascending.
pub fn sketch(seq: &[u8], p: SketchParams) -> Vec<u64> {
    if p.k == 0 || seq.len() < p.k {
        return Vec::new();
    }
    let mut hs: Vec<u64> = Vec::with_capacity(seq.len() - p.k + 1);
    for w in seq.windows(p.k) {
        let mut acc: u64 = 0;
        for &b in w {
            acc = acc.wrapping_mul(31).wrapping_add(b as u64);
        }
        hs.push(hash64(acc));
    }
    hs.sort_unstable();
    hs.dedup();
    hs.truncate(p.s);
    hs
}

/// Sketch many sequences in parallel (one rayon task per sequence). Sketching
/// dominates the wall-clock time on large FASTA inputs and each sequence is
/// independent, so this is the natural place to spend cores.
pub fn sketch_many(seqs: &[Vec<u8>], p: SketchParams) -> Vec<Vec<u64>> {
    seqs.par_iter().map(|s| sketch(s, p)).collect()
}

/// Bottom-s Jaccard estimate between two sorted, de-duplicated sketches.
pub fn jaccard(a: &[u64], b: &[u64], s: usize) -> f64 {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut considered, mut matched) = (0usize, 0usize);
    while considered < s && (i < a.len() || j < b.len()) {
        let take_a = j >= b.len() || (i < a.len() && a[i] <= b[j]);
        let take_b = i >= a.len() || (j < b.len() && b[j] <= a[i]);
        if take_a && take_b {
            matched += 1;
            i += 1;
            j += 1;
        } else if take_a {
            i += 1;
        } else {
            j += 1;
        }
        considered += 1;
    }
    if considered == 0 {
        0.0
    } else {
        matched as f64 / considered as f64
    }
}

/// Map a target sequence identity to a Jaccard cut-off, via the Mash relation:
/// the surviving k-mer fraction under substitutions is `f = p^k`, and the
/// Jaccard of two such sets is `f / (2 - f)`.
pub fn jaccard_threshold(min_id: f64, k: usize) -> f64 {
    let f = min_id.powi(k as i32);
    (f / (2.0 - f)).clamp(0.001, 1.0)
}

/// Invert the Mash relation: turn an observed Jaccard into an identity estimate.
pub fn mash_identity(j: f64, k: usize) -> f64 {
    if j <= 0.0 {
        return 0.0;
    }
    let f = 2.0 * j / (1.0 + j);
    f.powf(1.0 / k as f64)
}

pub struct GraphParams {
    pub sketch: SketchParams,
    /// Minimum identity for an edge.
    pub min_id: f64,
    /// Minimum length ratio (shorter / longer).
    pub len_cov: f64,
    /// Skip sketch hashes shared by more than this many sequences: they are
    /// low-complexity and would otherwise create a quadratic candidate list.
    pub max_bucket: usize,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            sketch: SketchParams::default(),
            min_id: 0.95,
            len_cov: 0.70,
            max_bucket: 512,
        }
    }
}

/// Build the weighted similarity graph over `sketches`.
///
/// Returns edges `(i, j, w)` with `i < j` and `w` the raw Jaccard in `(0, 1]`.
/// Only candidate pairs that share at least one sketch hash are ever scored.
pub fn build_graph(
    sketches: &[Vec<u64>],
    lengths: &[u32],
    p: &GraphParams,
) -> Vec<(u32, u32, f64)> {
    let thr = jaccard_threshold(p.min_id, p.sketch.k);

    // Invert the sketches: hash -> sequences carrying it.
    let mut buckets: HashMap<u64, Vec<u32>> = HashMap::new();
    for (i, sk) in sketches.iter().enumerate() {
        for &h in sk {
            buckets.entry(h).or_default().push(i as u32);
        }
    }

    // Candidate pairs: co-occurrence within a (non-degenerate) bucket.
    let mut pairs: Vec<(u32, u32)> = Vec::new();
    for list in buckets.values() {
        if list.len() < 2 || list.len() > p.max_bucket {
            continue;
        }
        for a in 0..list.len() {
            for b in (a + 1)..list.len() {
                let (x, y) = (list[a], list[b]);
                pairs.push(if x < y { (x, y) } else { (y, x) });
            }
        }
    }
    pairs.sort_unstable();
    pairs.dedup();

    // Score candidates in parallel; keep those over threshold and length ratio.
    pairs
        .par_iter()
        .filter_map(|&(a, b)| {
            let (la, lb) = (lengths[a as usize], lengths[b as usize]);
            let (lo, hi) = if la < lb { (la, lb) } else { (lb, la) };
            if hi == 0 || (lo as f64 / hi as f64) < p.len_cov {
                return None;
            }
            let j = jaccard(&sketches[a as usize], &sketches[b as usize], p.sketch.s);
            if j >= thr {
                // Weight by the raw Jaccard, not the Mash identity it implies.
                // MCL separates flow by *contrast* between weights, and the Mash
                // transform compresses it: 99%- vs 80%-identical pairs map to
                // identities 0.99 vs 0.80 (1.2x) but to Jaccards ~0.93 vs ~0.12
                // (8x). That contrast is what lets inflation resolve paralogous
                // subfamilies inside one component.
                Some((a, b, j))
            } else {
                None
            }
        })
        .collect()
}

/// A minimal, permissive multi-FASTA reader: returns `(ids, sequences)` with
/// sequence residues upper-cased and stripped of whitespace. Anything after the
/// first whitespace on a header line is dropped from the id (standard FASTA id).
pub fn read_fasta(text: &str) -> (Vec<String>, Vec<Vec<u8>>) {
    let mut ids = Vec::new();
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    let mut cur: Vec<u8> = Vec::new();
    let mut have = false;
    for line in text.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix('>') {
            if have {
                seqs.push(std::mem::take(&mut cur));
            }
            let id = rest.split_whitespace().next().unwrap_or("").to_string();
            ids.push(id);
            have = true;
        } else if have {
            for &b in line.as_bytes() {
                if !b.is_ascii_whitespace() {
                    cur.push(b.to_ascii_uppercase());
                }
            }
        }
    }
    if have {
        seqs.push(cur);
    }
    (ids, seqs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_sequences_have_jaccard_one() {
        let p = SketchParams::default();
        let s = sketch(b"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG", p);
        assert!(!s.is_empty());
        assert!((jaccard(&s, &s, p.s) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn threshold_matches_mash_relation() {
        let t = jaccard_threshold(0.95, 7);
        assert!((t - 0.5365).abs() < 1e-3, "got {t}");
    }

    #[test]
    fn mash_identity_inverts_threshold() {
        let k = 7;
        let j = jaccard_threshold(0.95, k);
        let id = mash_identity(j, k);
        assert!((id - 0.95).abs() < 1e-6, "got {id}");
    }

    #[test]
    fn divergent_sequences_lose_shared_kmers() {
        let p = SketchParams::default();
        let a = sketch(b"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG", p);
        let b = sketch(b"WWWWWWWWWWCCCCCCCCCCDDDDDDDDDDEEEEEEEEEEFFFFFFFFFF", p);
        assert!(jaccard(&a, &b, p.s) < 0.05);
    }

    #[test]
    fn fasta_roundtrip() {
        let (ids, seqs) = read_fasta(">a desc\nMKTA\nYIAK\n>b\nWWCC\n");
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(seqs[0], b"MKTAYIAK");
        assert_eq!(seqs[1], b"WWCC");
    }
}
