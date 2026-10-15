//! Deterministic synthetic inputs for `doctor`, the tests and benchmarks.

use crate::util::mix64;

/// Deterministic synthetic graph: one big planted-community component plus
/// thousands of small ones, weighted. Big enough to exercise column-parallel
/// steps and HydraMPP block tasks, small enough to finish in about a second.
pub fn synthetic(seed: u64, big: usize, small: usize) -> (usize, Vec<(u32, u32, f64)>) {
    let mut x = seed;
    let mut rnd = move || {
        x = mix64(x);
        x
    };
    let mut e = Vec::new();
    let comm = 40usize;
    for i in 0..big {
        for _ in 0..8 {
            let same = rnd() % 10 < 9;
            let j = if same { (i / comm) * comm + (rnd() as usize % comm) } else { rnd() as usize % big };
            if j != i && j < big {
                e.push((i as u32, j as u32, 0.5 + (rnd() % 1000) as f64 / 250.0));
            }
        }
    }
    let mut next = big;
    for _ in 0..small {
        let k = 2 + (rnd() % 9) as usize;
        for a in 0..k {
            for b in a + 1..k {
                if rnd() % 3 != 0 {
                    e.push(((next + a) as u32, (next + b) as u32, 1.0 + (rnd() % 7) as f64));
                }
            }
        }
        next += k;
    }
    (next, e)
}

/// Founder protein plus `n` copies with ~0.5% substitutions each.
pub fn family(n: usize, len: usize, seed: u64) -> Vec<Vec<u8>> {
    const AA: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";
    let mut x = seed;
    let mut rnd = move || {
        x = mix64(x);
        x
    };
    let founder: Vec<u8> = (0..len).map(|_| AA[(rnd() % 20) as usize]).collect();
    (0..n)
        .map(|_| founder.iter().map(|&c| if rnd() % 200 == 0 { AA[(rnd() % 20) as usize] } else { c }).collect())
        .collect()
}
