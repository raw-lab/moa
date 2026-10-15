//! Graph input and the matrix MCL runs on.
//!
//! * [`Csc`] — a flat compressed-sparse-column matrix with `f32` values. The
//!   reference `mcl` stores every value as a C `float` (`pval`, "cell contents:
//!   int and float"), so `f32` storage is both the faithful choice and half the
//!   memory traffic of `f64`. Accumulation always happens in `f64`, as in mcl.
//! * [`parse_abc`] — a fast label-pair reader. Tokenising runs in parallel;
//!   labels are interned in first-seen order, which is how mcl numbers nodes.
//! * [`Graph::from_edges`] — mirror + symmetrise with **max**-merge, exactly
//!   `mcl --abc` (`MCLXIO_STREAM_MIRROR | MCLXIO_STREAM_SYMMETRIC`,
//!   `mclpMergeMax`): `a b 3` and `b a 5` become one undirected edge of weight 5.
//! * [`Graph::components`] — connected components, so each one can be clustered
//!   independently (and in parallel) while reproducing whole-graph MCL exactly.

use crate::util::FxHashMap;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Value type of a matrix cell. See the module docs for why this is `f32`.
pub type Val = f32;

/// Flat CSC matrix: column `j` occupies `idx/val[ptr[j]..ptr[j+1]]`, rows
/// strictly ascending.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Csc {
    pub n: usize,
    pub ptr: Vec<usize>,
    pub idx: Vec<u32>,
    pub val: Vec<Val>,
}

impl Csc {
    pub fn empty(n: usize) -> Self {
        Csc { n, ptr: vec![0; n + 1], idx: Vec::new(), val: Vec::new() }
    }
    #[inline]
    pub fn nnz(&self) -> usize {
        self.idx.len()
    }
    #[inline]
    pub fn col(&self, j: usize) -> (&[u32], &[Val]) {
        let (a, b) = (self.ptr[j], self.ptr[j + 1]);
        (&self.idx[a..b], &self.val[a..b])
    }
    /// Value at `(row, col)` (0 when absent).
    #[inline]
    pub fn get(&self, row: u32, col: usize) -> Val {
        let (ix, vx) = self.col(col);
        match ix.binary_search(&row) {
            Ok(p) => vx[p],
            Err(_) => 0.0,
        }
    }

    /// `mclxAdjustLoops(mx, mclxLoopCBmax)`: zero any existing loop, then give
    /// every node a loop equal to its column maximum (1.0 for an isolated node).
    /// Followed by `mclxMakeStochastic`. This is mcl's start matrix.
    pub fn into_start_matrix(self) -> Csc {
        let n = self.n;
        let mut ptr = Vec::with_capacity(n + 1);
        let mut idx = Vec::with_capacity(self.nnz() + n);
        let mut val = Vec::with_capacity(self.nnz() + n);
        ptr.push(0);
        for j in 0..n {
            let (ix, vx) = self.col(j);
            let jj = j as u32;
            let mut mx: Val = 0.0;
            for (&i, &v) in ix.iter().zip(vx) {
                if i != jj && v > mx {
                    mx = v;
                }
            }
            let lw = if mx != 0.0 { mx } else { 1.0 };
            let start = idx.len();
            let mut placed = false;
            for (&i, &v) in ix.iter().zip(vx) {
                if !placed && i >= jj {
                    idx.push(jj);
                    val.push(lw);
                    placed = true;
                    if i == jj {
                        continue;
                    }
                }
                idx.push(i);
                val.push(v);
            }
            if !placed {
                idx.push(jj);
                val.push(lw);
            }
            // mclvNormalize: sum in double, divide each float.
            let s: f64 = val[start..].iter().map(|&v| v as f64).sum();
            if s > 0.0 {
                for v in &mut val[start..] {
                    *v = (*v as f64 / s) as Val;
                }
            }
            ptr.push(idx.len());
        }
        Csc { n, ptr, idx, val }
    }
}

/// An undirected weighted graph after mcl's input conventions have been
/// applied, stored as a symmetric [`Csc`] (no loops added yet).
#[derive(Clone)]
pub struct Graph {
    pub n: usize,
    pub adj: Csc,
    /// Undirected edges (`a <= b`) after max-merge — handy for statistics.
    pub n_edges: usize,
    /// Input edges ignored for a non-positive or non-finite weight (mcl turns
    /// these into meaningless negative "flow"; moa drops them and says so).
    pub dropped: usize,
}

impl Graph {
    /// Build from raw (possibly duplicated, possibly one-directional) edges.
    /// Non-positive and non-finite weights are discarded. Duplicates and the
    /// two directions of a pair merge with **max**, as `mcl --abc` does.
    pub fn from_edges(n: usize, edges: &[(u32, u32, f64)]) -> Graph {
        let dropped = edges.par_iter().filter(|e| !(e.2.is_finite() && e.2 > 0.0)).count();
        // Mirror every edge, then sort by (col,row) and max-merge.
        let mut ent: Vec<(u32, u32, Val)> = edges
            .par_iter()
            .filter(|e| e.2.is_finite() && e.2 > 0.0 && (e.0 as usize) < n && (e.1 as usize) < n)
            .flat_map_iter(|&(a, b, w)| {
                let w = w as Val; // mcl parses with strtod, stores as float
                let mirror = if a != b { Some((a, b, w)) } else { None };
                std::iter::once((b, a, w)).chain(mirror)
            })
            .filter(|e| e.2 > 0.0) // a tiny double can underflow to 0.0f
            .collect();
        ent.par_sort_unstable_by_key(|e| ((e.0 as u64) << 32) | e.1 as u64);
        let mut ptr = vec![0usize; n + 1];
        let mut idx: Vec<u32> = Vec::with_capacity(ent.len());
        let mut val: Vec<Val> = Vec::with_capacity(ent.len());
        let mut last: Option<(u32, u32)> = None;
        let mut n_edges = 0usize;
        for (c, r, w) in ent {
            if last == Some((c, r)) {
                let v = val.last_mut().unwrap();
                if w > *v {
                    *v = w;
                }
                continue;
            }
            last = Some((c, r));
            ptr[c as usize + 1] += 1;
            idx.push(r);
            val.push(w);
            if r >= c {
                n_edges += 1;
            }
        }
        for j in 0..n {
            ptr[j + 1] += ptr[j];
        }
        Graph { n, adj: Csc { n, ptr, idx, val }, n_edges, dropped }
    }

    /// Connected components, largest first (ties: smallest node first). Each
    /// component's nodes are listed ascending, so local ids preserve the
    /// global id order that mcl's tie-breaks depend on.
    pub fn components(&self) -> Vec<Component> {
        let n = self.n;
        let mut uf = UnionFind::new(n);
        for j in 0..n {
            let (ix, _) = self.adj.col(j);
            for &i in ix {
                uf.union(i as usize, j);
            }
        }
        let mut comp_of = vec![u32::MAX; n];
        let mut members: Vec<Vec<u32>> = Vec::new();
        for v in 0..n {
            let r = uf.find(v);
            if comp_of[r] == u32::MAX {
                comp_of[r] = members.len() as u32;
                members.push(Vec::new());
            }
            let c = comp_of[r];
            comp_of[v] = c;
            members[c as usize].push(v as u32);
        }
        let mut local = vec![0u32; n];
        for m in &members {
            for (li, &g) in m.iter().enumerate() {
                local[g as usize] = li as u32;
            }
        }
        let adj = &self.adj;
        let mut comps: Vec<Component> = members
            .into_par_iter()
            .map(|nodes| {
                let k = nodes.len();
                let mut ptr = Vec::with_capacity(k + 1);
                ptr.push(0);
                let mut idx = Vec::new();
                let mut val = Vec::new();
                for &g in &nodes {
                    let (ix, vx) = adj.col(g as usize);
                    idx.extend(ix.iter().map(|&i| local[i as usize]));
                    val.extend_from_slice(vx);
                    ptr.push(idx.len());
                }
                Component { nodes, adj: Csc { n: k, ptr, idx, val } }
            })
            .collect();
        comps.sort_by(|a, b| b.nodes.len().cmp(&a.nodes.len()).then(a.nodes[0].cmp(&b.nodes[0])));
        comps
    }
}

/// One connected component: global ids (ascending) and its local adjacency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Component {
    pub nodes: Vec<u32>,
    pub adj: Csc,
}

/// A labelled edge stream, parsed.
#[derive(Debug)]
pub struct Labeled {
    pub labels: Vec<String>,
    pub edges: Vec<(u32, u32, f64)>,
}

/// Parse an `--abc` stream: `labelA labelB [weight]` per line, any ASCII
/// whitespace between fields, a missing weight meaning 1.0. Blank lines and
/// `#` comments are skipped. Errors name the offending line.
pub fn parse_abc(text: &[u8]) -> Result<Labeled, String> {
    // 1. Split into chunks at line boundaries and tokenise them in parallel.
    let threads = rayon::current_num_threads().max(1);
    let target = (text.len() / (threads * 4)).max(1 << 16);
    let mut bounds = vec![0usize];
    let mut pos = 0;
    while pos < text.len() {
        let mut end = (pos + target).min(text.len());
        while end < text.len() && text[end - 1] != b'\n' {
            end += 1;
        }
        bounds.push(end);
        pos = end;
    }
    // Line numbers for error messages: count newlines before each chunk.
    let line0: Vec<usize> = {
        let counts: Vec<usize> =
            bounds.par_windows(2).map(|w| text[w[0]..w[1]].iter().filter(|&&b| b == b'\n').count()).collect();
        let mut acc = 1;
        counts
            .iter()
            .map(|c| {
                let s = acc;
                acc += c;
                s
            })
            .collect()
    };
    type Tok<'a> = (&'a [u8], &'a [u8], f64);
    let chunks: Vec<Result<Vec<Tok>, String>> = bounds
        .par_windows(2)
        .zip(line0.par_iter())
        .map(|(w, &l0)| {
            let mut out = Vec::new();
            for (k, line) in text[w[0]..w[1]].split(|&b| b == b'\n').enumerate() {
                let mut f = line.split(|b| b.is_ascii_whitespace()).filter(|t| !t.is_empty());
                let a = match f.next() {
                    None => continue,
                    Some(a) if a[0] == b'#' => continue,
                    Some(a) => a,
                };
                let b = f.next().ok_or_else(|| format!("line {}: expected 'labelA labelB [weight]'", l0 + k))?;
                let wt = match f.next() {
                    None => 1.0,
                    Some(t) => std::str::from_utf8(t)
                        .ok()
                        .and_then(|s| s.parse::<f64>().ok())
                        .ok_or_else(|| format!("line {}: bad weight '{}'", l0 + k, String::from_utf8_lossy(t)))?,
                };
                out.push((a, b, wt));
            }
            Ok(out)
        })
        .collect();
    // 2. Intern labels sequentially (first-seen order, as mcl numbers them).
    let mut ids: FxHashMap<&[u8], u32> = FxHashMap::default();
    let mut labels: Vec<&[u8]> = Vec::new();
    let mut edges = Vec::new();
    fn intern<'t>(ids: &mut FxHashMap<&'t [u8], u32>, labels: &mut Vec<&'t [u8]>, s: &'t [u8]) -> u32 {
        *ids.entry(s).or_insert_with(|| {
            labels.push(s);
            (labels.len() - 1) as u32
        })
    }
    for ch in chunks {
        for (a, b, w) in ch? {
            let ia = intern(&mut ids, &mut labels, a);
            let ib = intern(&mut ids, &mut labels, b);
            edges.push((ia, ib, w));
        }
    }
    let labels = labels.into_iter().map(|l| String::from_utf8_lossy(l).into_owned()).collect();
    Ok(Labeled { labels, edges })
}

/// Union-find with path halving and union by size.
pub struct UnionFind {
    parent: Vec<u32>,
    size: Vec<u32>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        UnionFind { parent: (0..n as u32).collect(), size: vec![1; n] }
    }
    #[inline]
    pub fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] as usize != x {
            let p = self.parent[x] as usize;
            self.parent[x] = self.parent[p];
            x = p;
        }
        x
    }
    #[inline]
    pub fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra as u32;
        self.size[ra] += self.size[rb];
    }
}
