//! Bottom-`s` MinHash sketching and alignment-free similarity graphs.
//!
//! Turns protein (or nucleotide) sequences into the weighted graph MCL
//! consumes. Two sequences get an edge when their estimated k-mer Jaccard
//! clears the cut-off implied by a target identity (the Mash relation).
//!
//! The estimator's variance matters: with a bottom-`s` sketch the standard
//! error of a Jaccard estimate near `J` is `sqrt(J(1-J)/s)`. At `s = 48` and
//! `J ≈ 0.54` that is ±0.072, so a *true* ortholog sitting near the threshold
//! is rejected roughly half the time. moa therefore defaults to `s = 256`,
//! where the same pair is rejected under 2% of the time.
//!
//! Candidate pairs come from a [`CandidateIndex`] whose work is **linear** in
//! the number of postings — no bucket is ever dropped, so a core gene shared by
//! thousands of genomes stays one family. Sketching, indexing and scoring all
//! run in parallel with rayon.

use crate::util::{mix64, FxHashMap};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct SketchParams {
    pub k: usize,
    pub s: usize,
}

impl Default for SketchParams {
    fn default() -> Self {
        SketchParams { k: 7, s: 256 }
    }
}

#[inline]
fn hash64(x: u64) -> u64 {
    mix64(x)
}

/// Bottom-`s` sketch of the k-mers of `seq`: the `s` smallest distinct hashes,
/// ascending. The k-mer code is the polynomial `Σ b·31^(k-1-i)` (mod 2⁶⁴),
/// computed with a rolling update; it is unique for upper-case residues.
pub fn sketch(seq: &[u8], p: SketchParams) -> Vec<u64> {
    let k = p.k;
    if k == 0 || p.s == 0 || seq.len() < k {
        return Vec::new();
    }
    let top = (0..k - 1).fold(1u64, |a, _| a.wrapping_mul(31)); // 31^(k-1)
    let mut hs: Vec<u64> = Vec::with_capacity(seq.len() - k + 1);
    let mut acc: u64 = 0;
    for &b in &seq[..k] {
        acc = acc.wrapping_mul(31).wrapping_add(b as u64);
    }
    hs.push(hash64(acc));
    for i in k..seq.len() {
        acc = acc.wrapping_sub((seq[i - k] as u64).wrapping_mul(top)).wrapping_mul(31).wrapping_add(seq[i] as u64);
        hs.push(hash64(acc));
    }
    // Long sequences: partial selection first, full sort only as a fallback.
    if hs.len() > 8 * p.s {
        let keep = 4 * p.s;
        hs.select_nth_unstable(keep);
        let mut head = hs[..keep].to_vec();
        head.sort_unstable();
        head.dedup();
        if head.len() >= p.s {
            head.truncate(p.s);
            return head;
        }
    }
    hs.sort_unstable();
    hs.dedup();
    hs.truncate(p.s);
    hs
}

/// Sketch many sequences in parallel.
pub fn sketch_many<S: AsRef<[u8]> + Sync>(seqs: &[S], p: SketchParams) -> Vec<Vec<u64>> {
    seqs.par_iter().map(|s| sketch(s.as_ref(), p)).collect()
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
/// Used as the MCL edge weight so that flow is proportional to similarity.
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
    /// Buckets (sequences sharing one sketch hash) of at most this size
    /// contribute every member pair as a candidate.
    pub dense_bucket: usize,
    /// Larger buckets use bounded fan-out: each member is paired with its
    /// `fanout` nearest neighbours on each side of a per-bucket pseudo-random
    /// ring. See [`CandidateIndex`].
    pub fanout: usize,
    /// Anchors per large bucket (see [`DEFAULT_ANCHORS`]).
    pub anchors: usize,
}

impl Default for GraphParams {
    fn default() -> Self {
        GraphParams {
            sketch: SketchParams::default(),
            min_id: 0.95,
            len_cov: 0.70,
            dense_bucket: DEFAULT_DENSE_BUCKET,
            fanout: DEFAULT_FANOUT,
            anchors: DEFAULT_ANCHORS,
        }
    }
}

/// Default bucket size up to which all member pairs are candidates.
pub const DEFAULT_DENSE_BUCKET: usize = 64;
/// Default ring fan-out (neighbours on each side) for larger buckets.
pub const DEFAULT_FANOUT: usize = 6;
/// Anchors per large bucket: members paired with *every* participant.
///
/// A ring alone keeps a big family *connected*, but turns its near-clique into
/// a sparse expander — and MCL fragments expanders (1.1.0: a 1,500-member core
/// gene came out as 1,358 clusters). Anchors give each large bucket a few hubs
/// that see the whole family, the star-like structure MCL resolves into one
/// cluster, at `O(anchors · bucket)` extra work. Every anchor pair is still
/// scored and must clear the Jaccard and length cut-offs.
pub const DEFAULT_ANCHORS: usize = 4;
/// A sequence takes part in the ring of a large bucket only when that bucket's
/// hash is among its `RING_HASHES` smallest sketch hashes (LSH-style: two
/// sequences share their minimum hash with probability equal to their Jaccard,
/// so similar sequences meet in the same rings). This bounds ring degree at
/// `2 · fanout · RING_HASHES`, independent of sketch size and family size.
pub const RING_HASHES: usize = 8;

/// SplitMix-derived ring key: an independent pseudo-random order of the members
/// of each bucket, keyed by the bucket's hash.
#[inline]
fn ring_key(bucket: u64, member: u32) -> u64 {
    hash64(bucket ^ (member as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93))
}

/// Inverted index over sketch hashes that yields candidate pairs in time and
/// memory **linear** in the number of (sequence, hash) postings.
///
/// The naive approach — every pair inside every bucket — is quadratic in bucket
/// size, and a single-copy core gene shared by *N* genomes puts ~*N* sequences in
/// each of its buckets. Capping bucket size to avoid that blow-up (the previous
/// design: skip buckets over 512) silently deletes every candidate edge of every
/// core family once *N* exceeds the cap, so the pangenome collapses past a few
/// hundred genomes.
///
/// Instead, a bucket of size ≤ `dense_bucket` contributes all pairs, and a
/// larger bucket is laid out on a ring in a pseudo-random order unique to that
/// bucket; each member is paired with its `fanout` neighbours on either side.
/// Because a sequence sits in up to `s` buckets, each with an independent ring,
/// the candidate graph of a large family is an expander of degree up to
/// `2 · fanout · s` rather than a chain, so it stays connected for MCL, while
/// the work per bucket is `O(size · fanout)` instead of `O(size²)`.
///
/// The candidate relation is symmetric, so [`CandidateIndex::candidates`]
/// returning only `j > i` produces every unordered pair exactly once.
pub struct CandidateIndex {
    /// hash → bucket. Dense buckets list every member; ring buckets list only
    /// the members whose `RING_HASHES` smallest hashes include this one, sorted
    /// by `(ring_key, member)` so a member's ring position is a binary search.
    buckets: FxHashMap<u64, Bucket>,
    fanout: usize,
    anchors: usize,
}

enum Bucket {
    Dense(Vec<u32>),
    Ring(Vec<u32>),
}

/// Epoch-stamped visited set, one per worker thread.
pub struct Marker {
    stamp: Vec<u32>,
    epoch: u32,
}

impl Marker {
    pub fn new(n: usize) -> Self {
        Marker { stamp: vec![0; n], epoch: 0 }
    }
    #[inline]
    fn next(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }
    /// True the first time `j` is seen in the current epoch.
    #[inline]
    fn first(&mut self, j: u32) -> bool {
        let s = &mut self.stamp[j as usize];
        if *s == self.epoch {
            false
        } else {
            *s = self.epoch;
            true
        }
    }
}

impl CandidateIndex {
    /// Set the number of anchors per large bucket (see [`DEFAULT_ANCHORS`]).
    pub fn with_anchors(mut self, a: usize) -> Self {
        self.anchors = a;
        self
    }

    pub fn new<S: AsRef<[u64]> + Sync>(sketches: &[S], dense_bucket: usize, fanout: usize) -> Self {
        // Postings (hash, member, rank-in-member's-sketch), grouped by a
        // parallel sort instead of a sequential hash-map build.
        let mut post: Vec<(u64, u32, u32)> = sketches
            .par_iter()
            .enumerate()
            .flat_map_iter(|(i, sk)| sk.as_ref().iter().enumerate().map(move |(r, &h)| (h, i as u32, r as u32)))
            .collect();
        post.par_sort_unstable_by_key(|p| (p.0, p.1));
        let dense_bucket = dense_bucket.max(2);
        // Group boundaries, then build buckets in parallel.
        let mut starts = Vec::new();
        for k in 0..post.len() {
            if k == 0 || post[k].0 != post[k - 1].0 {
                starts.push(k);
            }
        }
        starts.push(post.len());
        let built: Vec<(u64, Bucket)> = starts
            .par_windows(2)
            .filter_map(|w| {
                let g = &post[w[0]..w[1]];
                let h = g[0].0;
                if g.len() < 2 {
                    None
                } else if g.len() <= dense_bucket {
                    Some((h, Bucket::Dense(g.iter().map(|p| p.1).collect())))
                } else {
                    let mut ring: Vec<u32> = g.iter().filter(|p| (p.2 as usize) < RING_HASHES).map(|p| p.1).collect();
                    if ring.len() >= 2 {
                        ring.sort_unstable_by_key(|&m| (ring_key(h, m), m));
                        Some((h, Bucket::Ring(ring)))
                    } else {
                        None
                    }
                }
            })
            .collect();
        let mut buckets: FxHashMap<u64, Bucket> = FxHashMap::default();
        buckets.reserve(built.len());
        buckets.extend(built);
        CandidateIndex { buckets, fanout: fanout.max(1), anchors: DEFAULT_ANCHORS }
    }

    /// Number of buckets that switched to bounded fan-out (diagnostics).
    pub fn ring_buckets(&self) -> usize {
        self.buckets.values().filter(|b| matches!(b, Bucket::Ring(_))).count()
    }

    /// Largest bucket membership kept (diagnostics).
    pub fn max_bucket_len(&self) -> usize {
        self.buckets
            .values()
            .map(|b| match b {
                Bucket::Dense(v) | Bucket::Ring(v) => v.len(),
            })
            .max()
            .unwrap_or(0)
    }

    /// Fill `out` with the distinct candidates `j > i` of sequence `i`, whose
    /// ascending bottom-s sketch is `sk`.
    pub fn candidates(&self, i: u32, sk: &[u64], mark: &mut Marker, out: &mut Vec<u32>) {
        out.clear();
        mark.next();
        mark.first(i); // never pair with self
        for (rank, &h) in sk.iter().enumerate() {
            match self.buckets.get(&h) {
                None => {}
                Some(Bucket::Dense(v)) => {
                    for &j in v {
                        if j > i && mark.first(j) {
                            out.push(j);
                        }
                    }
                }
                Some(Bucket::Ring(v)) => {
                    if rank >= RING_HASHES {
                        continue; // i is not a participant of this ring
                    }
                    let key = (ring_key(h, i), i);
                    let Ok(pos) = v.binary_search_by_key(&key, |&m| (ring_key(h, m), m)) else {
                        continue;
                    };
                    let len = v.len();
                    let w = self.fanout.min((len - 1) / 2).max(1);
                    for d in 1..=w {
                        for &j in &[v[(pos + d) % len], v[(pos + len - d) % len]] {
                            if j > i && mark.first(j) {
                                out.push(j);
                            }
                        }
                    }
                    // Anchors: the first `anchors` members in ring order are
                    // paired with every participant (symmetric: an anchor
                    // sees everyone, everyone sees the anchors).
                    let a = self.anchors.min(len);
                    let targets: &[u32] = if pos < a { v } else { &v[..a] };
                    for &j in targets {
                        if j > i && mark.first(j) {
                            out.push(j);
                        }
                    }
                }
            }
        }
    }
}

/// Build the weighted similarity graph over already-sketched sequences.
///
/// Returns edges `(i, j, w)` with `i < j` and `w` the raw Jaccard estimate.
/// Work is linear in postings (see [`CandidateIndex`]); there is no bucket cap
/// that could drop a large gene family.
pub fn build_graph<S: AsRef<[u64]> + Sync>(sketches: &[S], lengths: &[u32], p: &GraphParams) -> Vec<(u32, u32, f64)> {
    let n = sketches.len();
    let thr = jaccard_threshold(p.min_id, p.sketch.k);
    let idx = CandidateIndex::new(sketches, p.dense_bucket, p.fanout).with_anchors(p.anchors);
    let per: Vec<Vec<(u32, u32, f64)>> = (0..n)
        .into_par_iter()
        .with_min_len(64)
        .map_init(
            || (Marker::new(n), Vec::new()),
            |(mark, cand), i| {
                let si = sketches[i].as_ref();
                idx.candidates(i as u32, si, mark, cand);
                cand.iter()
                    .filter_map(|&j| {
                        score_pair(
                            si,
                            sketches[j as usize].as_ref(),
                            lengths[i],
                            lengths[j as usize],
                            p.len_cov,
                            p.sketch.s,
                            thr,
                        )
                        .map(|w| (i as u32, j, w))
                    })
                    .collect()
            },
        )
        .collect();
    per.concat()
}

/// A parsed FASTA file.
pub struct Fasta {
    pub ids: Vec<String>,
    pub seqs: Vec<Vec<u8>>,
}

/// Permissive multi-FASTA reader. Ids are the header up to the first
/// whitespace. Residues are upper-cased and anything non-alphabetic (gaps,
/// stop `*`, digits, CR) is dropped, so soft-masked and aligned input sketch
/// identically to plain input. Text before the first header is ignored.
pub fn read_fasta(text: &[u8]) -> Result<Fasta, String> {
    let mut ids = Vec::new();
    let mut seqs: Vec<Vec<u8>> = Vec::new();
    for line in text.split(|&b| b == b'\n') {
        if let Some(h) = line.strip_prefix(b">") {
            let id = h.split(|b| b.is_ascii_whitespace()).find(|t| !t.is_empty()).unwrap_or(b"");
            let id =
                if id.is_empty() { format!("seq{}", ids.len() + 1) } else { String::from_utf8_lossy(id).into_owned() };
            ids.push(id);
            seqs.push(Vec::new());
        } else if let Some(cur) = seqs.last_mut() {
            cur.extend(line.iter().filter(|b| b.is_ascii_alphabetic()).map(|b| b.to_ascii_uppercase()));
        }
    }
    if ids.is_empty() {
        return Err("no FASTA records found (expected lines starting with '>')".into());
    }
    Ok(Fasta { ids, seqs })
}

/// Score one candidate pair: `Some(jaccard)` when it clears the length ratio and
/// the Jaccard threshold.
///
/// Weighted by the raw Jaccard, not by the Mash identity it implies: MCL
/// separates flow by *contrast* between edge weights, and the Mash transform
/// compresses it (a 99%- vs 80%-identical pair is 0.99 vs 0.80 as identity —
/// 1.2x — but ~0.93 vs ~0.12 as Jaccard — 8x). The Jaccard contrast is what
/// lets inflation resolve paralogous subfamilies inside one component.
#[inline]
pub fn score_pair(a: &[u64], b: &[u64], la: u32, lb: u32, len_cov: f64, s: usize, thr: f64) -> Option<f64> {
    let (lo, hi) = if la < lb { (la, lb) } else { (lb, la) };
    if hi == 0 || (lo as f64 / hi as f64) < len_cov {
        return None;
    }
    let j = jaccard(a, b, s);
    if j >= thr {
        Some(j)
    } else {
        None
    }
}

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;

    /// Deterministic protein family: a founder plus `n` copies with ~0.5%
    /// substitutions each — a single-copy core gene seen in `n` genomes.
    fn family(n: usize, len: usize, seed: u64) -> Vec<Vec<u8>> {
        const AA: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
        let mut x = seed;
        let mut rnd = || {
            x = hash64(x);
            x
        };
        let founder: Vec<u8> = (0..len).map(|_| AA[(rnd() % 20) as usize]).collect();
        (0..n)
            .map(|_| founder.iter().map(|&c| if rnd() % 200 == 0 { AA[(rnd() % 20) as usize] } else { c }).collect())
            .collect()
    }

    /// Regression for the >512-genome collapse: with the old `max_bucket = 512`
    /// every bucket of a 2000-member family was skipped, so the family produced
    /// zero edges. It must now form ONE connected component.
    #[test]
    fn large_family_stays_connected() {
        let p = GraphParams::default();
        let seqs = family(2000, 300, 7);
        let sk: Vec<Vec<u64>> = seqs.iter().map(|s| sketch(s, p.sketch)).collect();
        let lens: Vec<u32> = seqs.iter().map(|s| s.len() as u32).collect();
        let edges = build_graph(&sk, &lens, &p);
        assert!(!edges.is_empty(), "large family produced no edges");
        let comps = crate::graph::Graph::from_edges(seqs.len(), &edges).components();
        assert_eq!(comps.len(), 1, "family split into {} components", comps.len());
    }

    /// Candidate work must be linear in postings, not quadratic in bucket size:
    /// a 2000-member family must not generate ~2M pairs.
    #[test]
    fn candidate_count_is_linear() {
        let p = GraphParams::default();
        let n = 2000;
        let seqs = family(n, 300, 11);
        let sk: Vec<Vec<u64>> = seqs.iter().map(|s| sketch(s, p.sketch)).collect();
        let idx = CandidateIndex::new(&sk, p.dense_bucket, p.fanout);
        assert!(idx.ring_buckets() > 0);
        let mut mark = Marker::new(n);
        let mut out = Vec::new();
        let mut total = 0usize;
        for i in 0..n {
            idx.candidates(i as u32, &sk[i], &mut mark, &mut out);
            total += out.len();
        }
        // Per gene: ≤ 2·fanout ring neighbours + `anchors` anchors per ring it
        // joins, plus (as an anchor) at most one slot per ring — independent of s.
        let bound = n * RING_HASHES * (2 * p.fanout + 2 * p.anchors);
        let quadratic = n * (n - 1) / 2;
        assert!(total <= bound, "{total} candidates > linear bound {bound}");
        assert!(total * 5 < quadratic, "{total} candidates is not sub-quadratic ({quadratic})");
    }

    /// Small buckets keep the exact all-pairs behaviour.
    #[test]
    fn dense_buckets_give_all_pairs() {
        let p = GraphParams::default();
        let seqs = family(20, 200, 3);
        let sk: Vec<Vec<u64>> = seqs.iter().map(|s| sketch(s, p.sketch)).collect();
        let idx = CandidateIndex::new(&sk, p.dense_bucket, p.fanout);
        assert_eq!(idx.ring_buckets(), 0);
        let mut mark = Marker::new(20);
        let mut out = Vec::new();
        let mut pairs = 0;
        for i in 0..20 {
            idx.candidates(i as u32, &sk[i], &mut mark, &mut out);
            pairs += out.len();
        }
        assert_eq!(pairs, 20 * 19 / 2);
    }

    /// Each unordered pair is produced once (j > i, no duplicates).
    #[test]
    fn candidates_are_unique_and_ordered() {
        let p = GraphParams::default();
        let seqs = family(500, 250, 5);
        let sk: Vec<Vec<u64>> = seqs.iter().map(|s| sketch(s, p.sketch)).collect();
        let idx = CandidateIndex::new(&sk, p.dense_bucket, p.fanout);
        let mut mark = Marker::new(500);
        let mut out = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for i in 0..500u32 {
            idx.candidates(i, &sk[i as usize], &mut mark, &mut out);
            for &j in &out {
                assert!(j > i);
                assert!(seen.insert((i, j)), "duplicate pair ({i},{j})");
            }
        }
    }

    #[test]
    fn identical_sequences_have_jaccard_one() {
        let p = SketchParams::default();
        let s = sketch(b"MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSG", p);
        assert!(!s.is_empty());
        assert!((jaccard(&s, &s, p.s) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn threshold_matches_mash_relation() {
        // p^k with p=0.95, k=7 -> f=0.6983; J = f/(2-f) = 0.5365
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
}
