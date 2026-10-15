//! Small shared helpers: a fast non-cryptographic hasher and a visited-set.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash (the rustc hasher): one multiply-rotate per word. Labels and sketch
/// hashes are not attacker-controlled, so DoS-resistant SipHash buys nothing.
#[derive(Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

impl FxHasher {
    #[inline]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut c = bytes.chunks_exact(8);
        for w in &mut c {
            self.add(u64::from_le_bytes(w.try_into().unwrap()));
        }
        let r = c.remainder();
        if !r.is_empty() {
            let mut buf = [0u8; 8];
            buf[..r.len()].copy_from_slice(r);
            self.add(u64::from_le_bytes(buf) ^ ((r.len() as u64) << 56));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(i as u64)
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64)
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i)
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64)
    }
    /// The multiply in `add` only carries entropy *upward*, so keys sharing a
    /// long prefix (`gene_1…`, `NZ_CP012345.1_1…`) differ only in a few bits of
    /// the last word and collide in hashbrown's low-bit buckets (measured: 206
    /// of 4096 buckets used for `gene_0..4095`). A full SplitMix64 finaliser
    /// restores SipHash-level spread for a couple of nanoseconds.
    #[inline]
    fn finish(&self) -> u64 {
        mix64(self.hash)
    }
}

pub type FxBuild = BuildHasherDefault<FxHasher>;
pub type FxHashMap<K, V> = HashMap<K, V, FxBuild>;

/// Epoch-stamped visited set: `clear` is O(1).
pub struct Marker {
    stamp: Vec<u32>,
    epoch: u32,
}

impl Marker {
    pub fn new(n: usize) -> Self {
        Marker { stamp: vec![0; n], epoch: 1 }
    }
    #[inline]
    pub fn clear(&mut self) {
        self.epoch = self.epoch.wrapping_add(1);
        if self.epoch == 0 {
            self.stamp.iter_mut().for_each(|s| *s = 0);
            self.epoch = 1;
        }
    }
    /// True the first time `j` is seen since the last `clear`.
    #[inline]
    pub fn first(&mut self, j: u32) -> bool {
        let s = &mut self.stamp[j as usize];
        if *s == self.epoch {
            false
        } else {
            *s = self.epoch;
            true
        }
    }
}

/// SplitMix64 finaliser.
#[inline]
pub fn mix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
