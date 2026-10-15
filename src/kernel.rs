//! The fused MCL column kernel.
//!
//! One MCL iteration is column-local: column `j` of the next iterand depends
//! only on column `j` of `M*M`, which is a linear combination of the columns of
//! `M` named in column `j`. So instead of materialising `M*M` and then pruning
//! and inflating it (two extra passes over the largest matrix in the run), each
//! column is taken from compose to its final inflated form in one go, inside a
//! per-worker scratch buffer:
//!
//! ```text
//!   compose      out = Σ_k M[k][j] · M[:,k]        (f64 accumulate, f32 store)
//!   measure      maxval, center = Σ v²             (mcl: vecMeasure)
//!   prune        keep v ≥ 0.99999/P                (mclvSelectGqBar)
//!   recover      if n < R and mass < pct: top-R    (mclvKBar, ties kept)
//!   | select     elif n > S: top-S, nested recover (mclvKBar, ties kept)
//!   rescale      v /= mass_final                   (mclvScale)
//!   chaos        (maxval − center) · n_final       (expand.c:503)
//!   inflate      v = v^I / Σ v^I                   (mclvInflate)
//! ```
//!
//! Every step follows `mcl 22-282` (`src/mcl/expand.c:mclExpandVector`,
//! `src/impala/vector.c`) including its corner cases: `mclvKBar` keeps **all**
//! entries tied at the bar; the nested recover restores the *post-prune*
//! vector; the emergency self-loop only fires when recovery is disabled.

use crate::graph::{Csc, Val};
use crate::mcl::MclParams;

/// Per-worker scratch: dense accumulator plus reusable buffers. Allocated once
/// per worker per component size; nothing is allocated per column.
pub struct Scratch {
    acc: Vec<f64>,
    touched: Vec<u32>,
    fidx: Vec<u32>,
    fval: Vec<Val>,
    pidx: Vec<u32>,
    pval: Vec<Val>,
    sel: Vec<Val>,
}

impl Scratch {
    pub fn new(n: usize) -> Self {
        Scratch {
            acc: vec![0.0; n],
            touched: Vec::new(),
            fidx: Vec::new(),
            fval: Vec::new(),
            pidx: Vec::new(),
            pval: Vec::new(),
            sel: Vec::new(),
        }
    }
    fn fit(&mut self, n: usize) {
        if self.acc.len() < n {
            self.acc.resize(n, 0.0);
        }
    }
}

/// `acc[idx[t]] += val[t] · w` for every t — the hot loop, unrolled 4× over
/// raw pointers. Row indices within one column are distinct, so the four
/// updates of a round never alias; the order of additions into any single
/// `acc[i]` is unchanged, so results are bit-identical to the plain loop.
#[inline(always)]
fn scatter(acc: &mut [f64], idx: &[u32], val: &[Val], w: f64) {
    let n = idx.len();
    debug_assert_eq!(n, val.len());
    let a = acc.as_mut_ptr();
    let (ip, vp) = (idx.as_ptr(), val.as_ptr());
    let mut t = 0;
    // SAFETY: t < n = idx.len() = val.len(); every row index is < acc.len().
    unsafe {
        while t + 4 <= n {
            let (i0, i1, i2, i3) =
                (*ip.add(t) as usize, *ip.add(t + 1) as usize, *ip.add(t + 2) as usize, *ip.add(t + 3) as usize);
            let (v0, v1, v2, v3) =
                (*vp.add(t) as f64, *vp.add(t + 1) as f64, *vp.add(t + 2) as f64, *vp.add(t + 3) as f64);
            *a.add(i0) += v0 * w;
            *a.add(i1) += v1 * w;
            *a.add(i2) += v2 * w;
            *a.add(i3) += v3 * w;
            t += 4;
        }
        while t < n {
            *a.add(*ip.add(t) as usize) += *vp.add(t) as f64 * w;
            t += 1;
        }
    }
}

/// `mclvKBar(vec, k, ignore, KBAR_SELECT_LARGE)`: the k-th largest value among
/// those strictly below `ignore`. `n_total` is the vector size mcl tests `k`
/// against (it short-circuits to "select everything" when `k >= n_ivps`).
#[inline]
fn kbar_large(vals: &[Val], n_total: usize, k: usize, ignore: f64, buf: &mut Vec<Val>) -> f64 {
    if k >= n_total {
        return -(f32::MAX as f64);
    }
    buf.clear();
    buf.extend(vals.iter().copied().filter(|&v| (v as f64) < ignore));
    if buf.is_empty() || k == 0 {
        return f32::MAX as f64; // mcl's heap would be empty: select nothing more
    }
    let k = k.min(buf.len());
    // k-th largest == element at k-1 in descending order.
    let (_, kth, _) = buf.select_nth_unstable_by(k - 1, |a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    *kth as f64
}

/// Keep `(i, v)` with `v >= bar`, appending to `out_*`; returns the kept mass.
#[inline]
fn select_gq(idx: &[u32], val: &[Val], bar: f64, oi: &mut Vec<u32>, ov: &mut Vec<Val>) -> f64 {
    let mut mass = 0.0f64;
    for (&i, &v) in idx.iter().zip(val) {
        if v as f64 >= bar {
            mass += v as f64;
            oi.push(i);
            ov.push(v);
        }
    }
    mass
}

/// Compute next-iterand column `j` from `m`, appending it to `oi`/`ov`.
/// Returns the column's chaos contribution.
pub fn step_column(m: &Csc, j: usize, p: &MclParams, s: &mut Scratch, oi: &mut Vec<u32>, ov: &mut Vec<Val>) -> f64 {
    let n = m.n;
    s.fit(n);

    // ---- compose: dense accumulate over the columns named in column j -------
    // Two regimes, chosen per column from the predicted output size `est`:
    // * dense  — a branchless scatter-add, then one sweep of the accumulator;
    // * sparse — track touched rows so gathering costs O(output), not O(n).
    // Both sum the same products in the same order: bit-identical results.
    s.touched.clear();
    let (cj_idx, cj_val) = m.col(j);
    let est: usize = cj_idx.iter().map(|&k| m.ptr[k as usize + 1] - m.ptr[k as usize]).sum();
    let dense = est * 4 >= n;
    let acc = &mut s.acc[..n];
    for (&k, &wkj) in cj_idx.iter().zip(cj_val) {
        if wkj == 0.0 {
            continue;
        }
        let wkj = wkj as f64;
        let (ck_idx, ck_val) = m.col(k as usize);
        if dense {
            scatter(acc, ck_idx, ck_val, wkj);
        } else {
            for (&i, &wik) in ck_idx.iter().zip(ck_val) {
                let slot = unsafe { acc.get_unchecked_mut(i as usize) };
                if *slot == 0.0 {
                    s.touched.push(i);
                }
                *slot += wik as f64 * wkj;
            }
        }
    }
    // Gather in ascending row order: sweep the accumulator when the column is
    // dense relative to n, otherwise sort the (short) touched list.
    s.fidx.clear();
    s.fval.clear();
    if dense || s.touched.len() * 12 >= n {
        for i in 0..n {
            let v = s.acc[i];
            if v != 0.0 {
                s.fidx.push(i as u32);
                s.fval.push(v as Val);
                s.acc[i] = 0.0;
            }
        }
    } else {
        s.touched.sort_unstable();
        for &i in &s.touched {
            let v = s.acc[i as usize];
            s.acc[i as usize] = 0.0;
            s.fidx.push(i);
            s.fval.push(v as Val);
        }
    }
    let n_full = s.fidx.len();

    // ---- measure on the full (float) expanded vector ------------------------
    let (mut maxval, mut center) = (0.0f64, 0.0f64);
    for &v in &s.fval {
        let v = v as f64;
        center += v * v;
        if v > maxval {
            maxval = v;
        }
    }

    // ---- prune ---------------------------------------------------------------
    let cut = p.precision();
    s.pidx.clear();
    s.pval.clear();
    let mut mass_prune = if cut != 0.0 {
        select_gq(&s.fidx, &s.fval, cut, &mut s.pidx, &mut s.pval)
    } else {
        s.pidx.extend_from_slice(&s.fidx);
        s.pval.extend_from_slice(&s.fval);
        s.fval.iter().map(|&v| v as f64).sum()
    };
    // Emergency measure (only when recovery is off): keep the node alive.
    if p.recover == 0 && s.pidx.is_empty() {
        s.pidx.push(j as u32);
        s.pval.push(1.0);
        mass_prune = 1.0;
    }
    let n_prune = s.pidx.len();

    let base = oi.len();
    let mass_final;
    if p.recover > 0 && n_prune < p.recover && mass_prune < p.pct {
        // ---- recover from the full pre-prune vector -------------------------
        let rbar =
            if n_full > p.recover { kbar_large(&s.fval, n_full, p.recover - n_prune, cut, &mut s.sel) } else { 0.0 };
        mass_final = select_gq(&s.fidx, &s.fval, rbar, oi, ov);
    } else if p.select > 0 && n_prune > p.select {
        // ---- select (k-th largest; both of mcl's KBar branches give it) ------
        let sbar = kbar_large(&s.pval, n_prune, p.select, f64::INFINITY, &mut s.sel);
        let mass_select = select_gq(&s.pidx, &s.pval, sbar, oi, ov);
        let n_select = oi.len() - base;
        if p.recover > 0 && n_select < p.recover && mass_select < p.pct {
            // ---- nested recover, from the *post-prune* vector ----------------
            oi.truncate(base);
            ov.truncate(base);
            let rbar = if n_prune > p.recover {
                kbar_large(&s.pval, n_prune, p.recover - n_select, sbar, &mut s.sel)
            } else {
                0.0
            };
            mass_final = select_gq(&s.pidx, &s.pval, rbar, oi, ov);
        } else {
            mass_final = mass_select;
        }
    } else {
        oi.extend_from_slice(&s.pidx);
        ov.extend_from_slice(&s.pval);
        mass_final = mass_prune;
    }

    // ---- rescale, chaos, inflate --------------------------------------------
    let col = &mut ov[base..];
    if mass_final != 0.0 {
        for v in col.iter_mut() {
            *v = (*v as f64 / mass_final) as Val;
        }
    }
    let chaos = (maxval - center) * col.len() as f64;
    inflate(col, p.inflation);
    chaos
}

/// `mclvInflate`: `pow` in double, store float, sum the floats, divide.
#[inline]
fn inflate(col: &mut [Val], power: f64) {
    if col.is_empty() {
        return;
    }
    let mut powsum = 0.0f64;
    if power == 2.0 {
        for v in col.iter_mut() {
            let x = *v as f64;
            *v = (x * x) as Val;
            powsum += *v as f64;
        }
    } else {
        for v in col.iter_mut() {
            *v = (*v as f64).powf(power) as Val;
            powsum += *v as f64;
        }
    }
    if powsum <= 0.0 {
        for v in col.iter_mut() {
            *v = 0.0;
        }
        return;
    }
    for v in col.iter_mut() {
        *v = (*v as f64 / powsum) as Val;
    }
}

/// Output of stepping a contiguous range of columns.
#[derive(Default, serde::Serialize, serde::Deserialize)]
pub struct Block {
    pub lens: Vec<u32>,
    pub idx: Vec<u32>,
    pub val: Vec<Val>,
    pub chaos: f64,
}

/// Step columns `lo..hi` serially into one [`Block`].
pub fn step_block(m: &Csc, lo: usize, hi: usize, p: &MclParams, s: &mut Scratch) -> Block {
    let mut b = Block { lens: Vec::with_capacity(hi - lo), ..Default::default() };
    let est = m.ptr[hi] - m.ptr[lo];
    b.idx.reserve(est + (hi - lo));
    b.val.reserve(est + (hi - lo));
    for j in lo..hi {
        let before = b.idx.len();
        let c = step_column(m, j, p, s, &mut b.idx, &mut b.val);
        // An inflate over a non-positive sum zeroes the column; mcl resizes it
        // to empty. Drop those zero entries so the column really is empty.
        if b.val[before..].iter().all(|&v| v == 0.0) && b.idx.len() > before {
            b.idx.truncate(before);
            b.val.truncate(before);
        }
        b.lens.push((b.idx.len() - before) as u32);
        if c > b.chaos {
            b.chaos = c;
        }
    }
    b
}

/// Stitch consecutive blocks back into one matrix without a second full copy:
/// the first block's buffers become the output (a large `realloc` is usually
/// an in-place `mremap`), and every other block is freed as soon as it has
/// been appended. Peak memory stays ≈ old iterand + new iterand, as in mcl.
pub fn stitch(n: usize, blocks: Vec<Block>) -> (Csc, f64) {
    let nnz: usize = blocks.iter().map(|b| b.idx.len()).sum();
    let mut ptr = Vec::with_capacity(n + 1);
    ptr.push(0usize);
    let mut chaos = 0.0f64;
    let mut it = blocks.into_iter();
    let (mut idx, mut val) = match it.next() {
        None => (Vec::new(), Vec::new()),
        Some(b) => {
            for &l in &b.lens {
                let last = *ptr.last().unwrap();
                ptr.push(last + l as usize);
            }
            chaos = chaos.max(b.chaos);
            (b.idx, b.val)
        }
    };
    idx.reserve_exact(nnz - idx.len());
    val.reserve_exact(nnz - val.len());
    for b in it {
        for &l in &b.lens {
            let last = *ptr.last().unwrap();
            ptr.push(last + l as usize);
        }
        idx.extend_from_slice(&b.idx);
        val.extend_from_slice(&b.val);
        chaos = chaos.max(b.chaos);
        // `b` dropped here: its memory is released before the next copy.
    }
    if idx.capacity() > idx.len() + idx.len() / 4 + 64 {
        idx.shrink_to_fit();
        val.shrink_to_fit();
    }
    debug_assert_eq!(ptr.len(), n + 1);
    (Csc { n, ptr, idx, val }, chaos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kbar_is_kth_largest_below_ignore() {
        let v: Vec<Val> = vec![0.1, 0.4, 0.4, 0.3, 0.9];
        let mut buf = Vec::new();
        assert_eq!(kbar_large(&v, v.len(), 1, f64::INFINITY, &mut buf), 0.9f32 as f64);
        assert_eq!(kbar_large(&v, v.len(), 2, f64::INFINITY, &mut buf), 0.4f32 as f64);
        assert_eq!(kbar_large(&v, v.len(), 3, f64::INFINITY, &mut buf), 0.4f32 as f64);
        // ignore values >= 0.5 → 0.4 is the largest remaining
        assert_eq!(kbar_large(&v, v.len(), 1, 0.5, &mut buf), 0.4f32 as f64);
        // k >= n selects everything (mclvKBar's early return)
        assert_eq!(kbar_large(&v, v.len(), 5, f64::INFINITY, &mut buf), -(f32::MAX as f64));
    }

    #[test]
    fn select_keeps_every_entry_tied_at_the_bar() {
        // Select S=2 from [0.1, 0.3, 0.3, 0.3]: mcl keeps all three 0.3s.
        let idx = [0u32, 1, 2, 3];
        let val: [Val; 4] = [0.1, 0.3, 0.3, 0.3];
        let mut buf = Vec::new();
        let bar = kbar_large(&val, 4, 2, f64::INFINITY, &mut buf);
        let (mut oi, mut ov) = (vec![], vec![]);
        select_gq(&idx, &val, bar, &mut oi, &mut ov);
        assert_eq!(oi, vec![1, 2, 3]);
    }
}
