```
              .-"-.
             ( o o )
              \ v /
               '-'
   ███╗   ███╗ ██████╗  █████╗
   ████╗ ████║██╔═══██╗██╔══██╗
   ██╔████╔██║██║   ██║███████║
   ██║╚██╔╝██║██║   ██║██╔══██║
   ██║ ╚═╝ ██║╚██████╔╝██║  ██║
   ╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝
```

# moa

**Markov clustering + MinHash, forged 100% in Rust.**

> *Extinct bird. Living clusters.*

`moa` is a faithful, dependency-light reimplementation of **Markov Cluster
Algorithm (MCL)** — Stijn van Dongen's graph-flow clusterer — with an
alignment-free **MinHash** front-end for turning sequences into the graph MCL
eats. It is written entirely in Rust and ships **four interchangeable expansion
engines**: a serial reference, a rayon data-parallel engine, a distributed
**HydraMPP** engine (laptop → SLURM), and a **GPU** engine (wgpu compute, runs on
Vulkan/Metal/DX12/GL, software `lavapipe` included).

The point of `moa` is *fidelity*: it reproduces the reference `mcl` binary's
partitions **exactly** — 25/25 identical clusterings across a battery of graphs
and inflation values — while giving you modern parallelism and a clean CLI.

---

## Why another MCL?

* **It's faithful.** The expansion/inflation loop, the `prune → select →
  recover` sparsification, the *chaos* convergence test, and the DAG-based
  interpretation are ported line-for-line from the reference C
  (`mcl/{expand,inflate,proc,interpret}.c`, `impala/vector.c`). Same results,
  no surprises.
* **It's 100% Rust.** No C to build, no BLAS to find. The GPU path is wgpu
  (pure-Rust WebGPU) so it binds whatever adapter your machine has — or a
  software one when it has none.
* **It scales the same code four ways.** Expansion is the whole cost of MCL, and
  it's embarrassingly parallel over columns. Pick `serial`, `rayon`, `hydra`, or
  `gpu`; the clustering is identical.
* **It clusters sequences directly.** `moa cluster proteins.faa` sketches every
  sequence with bottom-`s` MinHash, builds a Jaccard similarity graph, and runs
  MCL on it — no all-vs-all alignment.

---

## Install

Requires **Rust ≥ 1.75** (the pinned floor; see *MSRV* below).

```bash
# Build everything (GPU engine on by default).
cargo build --release

# The binary:
./target/release/moa doctor
```

If you have no GPU stack at all and don't want wgpu in the tree:

```bash
cargo build --release --no-default-features   # serial + rayon + hydra only
```

To exercise a software GPU (Mesa lavapipe) when there's no hardware device, point
wgpu at the ICD:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
export XDG_RUNTIME_DIR=/tmp/xdg && mkdir -p /tmp/xdg
```

---

## Quick start

```bash
# Cluster a labelled graph (one edge per line: "labelA labelB [weight]").
moa mcl examples/friends.abc -I 2.0
#   dan   harriet  kelly  indy  lucy
#   ann   fred     john   gilian
#   bob   charlie  esme

# Coarser clustering: lower the inflation.
moa mcl examples/friends.abc -I 1.4        # -> two groups

# Pick an engine (identical output, different hardware).
moa mcl examples/friends.abc --engine gpu
moa mcl big.abc --engine hydra -o clusters.tsv

# Cluster sequences straight from FASTA (MinHash + MCL, no alignment).
moa cluster examples/demo.faa --min-id 0.90
#   orfA_1  orfA_2  orfA_3
#   orfB_1  orfB_2  orfB_3
#   orfC_1  orfC_2
#   lonely_1

# Prove every engine agrees with the serial reference on the friends graph.
moa doctor

# Build / backend / environment details.
moa info
```

Output is one cluster per line, members tab-separated, largest cluster first —
the same shape `mcl -o` emits, so it drops into existing pipelines.

---

## The algorithm (what "faithful" means here)

MCL runs on a **column-stochastic** matrix `M` (each column is where a node's
flow goes) and alternates two operators until a fixed point:

1. **Expansion** — `M := M · M`. Flow spreads along length-two paths. This is the
   only expensive step, and the only one `moa` parallelises.
2. **Inflation** — `M[i][j] := M[i][j]^r`, renormalised per column. Raising to
   the inflation power `r` (`-I`) sharpens strong flow and starves weak flow;
   higher `r` ⇒ finer clusters.

Between the two, the reference keeps `M` sparse without changing the result,
exactly as `moa` does:

```
precision = 0.99999 / prune                 # the absolute prune cut
prune :  drop entries below `precision`
recover: if a column kept < recover entries AND lost > (1 - pct) of its mass,
         restore its `recover` largest pre-prune entries
select : else if a column kept > select entries, keep only its `select` largest
```

Progress is measured by **chaos**, per the reference `vecMeasure`:

```
center    = Σ vᵢ²                      (over the full expanded column)
maxval    = max vᵢ
chaos_col = (maxval − center) · nnz    (nnz counted after pruning)
chaos     = max over columns
```

Iteration stops when `chaos ≤ chaos_limit` (default `1e-4`). At a clean fixed
point each column concentrates on one attractor, so `maxval == center` and chaos
is exactly **0**.

Clusters are then read off by `interpret.c`'s two-step: **`mclDag`** prunes `M`
to its flow DAG using the per-column bar

```
bar = selfval < maxval ? 0.999·selfval + 0.001·maxval
                       : selfval / (1 + 0.01)
```

then **`mclInterpret`** marks *attractors* (nodes whose DAG column keeps their
own self-loop), unions connected attractors into *attractor systems*, and
attaches every remaining node to the system its flow reaches (a union-find over
DAG edges, equivalent to the reference's flow closure).

Defaults match the `mcl` binary's `dir_scheme` row 6: `prune 10000, select 1100,
recover 1400, pct 0.90`, inflation `2.0`, chaos limit `1e-4`.

---

## The four engines

Expansion is a pure function of `M`, so the engine can never change the
clustering (up to GPU single-precision rounding). That is exactly what lets
`moa doctor` cross-check them on a live graph.

| Engine   | How it expands `M·M`                                             | Precision | When to use |
|----------|------------------------------------------------------------------|-----------|-------------|
| `serial` | One thread; the correctness reference.                           | f64       | Small graphs, CI, ground truth. |
| `rayon`  | Columns composed in parallel (`par_iter`), per-worker scratch.   | f64       | The default. Any multicore box. |
| `hydra`  | Column blocks farmed out via **HydraMPP** (`map_owned`).         | f64       | Many cores / a cluster (host + clients). |
| `gpu`    | Dense `C = A·A` compute shader via **wgpu**; re-sparsified.      | f32       | Dense-ish or mid-size graphs with a GPU. |

The GPU engine densifies `M` into an `n×n` `f32` buffer, runs a tiled GEMM WGSL
shader, reads it back, and re-sparsifies; the per-column prune and chaos then run
on the CPU exactly as for the other engines. WGSL has no `f64`, so GPU is single
precision — which never moves these partitions. Above `n = 2048` the GPU engine
transparently falls back to rayon for that expansion (dense storage is `O(n²)`)
and says so once.

`moa doctor` on a machine with only software Vulkan:

```
  reference (serial): 3 clusters at -I 2.0  [expected 3]
  serial  OK   3 clusters, 0.32 ms   [serial (1 thread)]
  rayon   OK   3 clusters, 0.93 ms   [rayon (N threads)]
  hydra   OK   3 clusters, 1.55 ms   [HydraMPP local (…: N CPUs, …)]
  gpu     OK   3 clusters, 99.0 ms   [wgpu Vulkan — llvmpipe … (f32, dense GEMM, n<=2048)]
  -I 1.4 -> 2 clusters (chaos 2.09e-5)  [expected 2]
  ALL CHECKS PASSED
```

---

## Validation

`moa` is checked against the genuine reference binary (`mcl 22-282`, installed
from the same source tree it ports):

* **25 / 25 exact partition match** across five graphs (random Erdős–Rényi and
  planted-community, 20–80 nodes) × five inflations `{1.2, 1.4, 2.0, 3.0, 5.0}`.
  Every clustering is set-for-set identical to `mcl`.
* **9 / 9 cross-engine agreement** — `serial == rayon == hydra == gpu` on the
  same graphs and inflations.
* The canonical **friends** example gives the three named friend groups at
  `-I 2.0` and coarsens to two at `-I 1.4`, with chaos → 0.
* **18 tests pass** (`cargo test`): the friends proof, chaos convergence,
  granularity monotonicity, algebraic edge cases (cliques, weak bridges, isolated
  nodes), all-engine equivalence, and the MinHash family-recovery pipeline.

### How it compares

The only other Markov clusterer on crates.io is **`ratio-markov`** — a dense
`nalgebra` variant with an integer expansion power, an evaporation constant, and
an argmax-per-column interpretation. It is a *different algorithm*, and it shows:

| Graph / inflation      | reference `mcl` | `moa`            | `ratio-markov`      |
|------------------------|-----------------|------------------|---------------------|
| friends, `-I 2.0`      | 3 clusters      | **3 (identical)**| 2 (merges two groups) |
| friends, `-I 1.4`      | 2 clusters      | **2 (identical)**| 1 (everything merges) |
| planted-3, `-I 2.0`    | 3 clusters      | **3 (identical)**| 3 (agrees here)     |
| dense random, `-I 2.0` | 5 clusters      | **5 (identical)**| 2                   |
| sparse random, `-I 2.0`| 10 clusters     | **10 (identical)**| 5                  |

`moa` tracks the reference exactly; `ratio-markov` gives coarser, different
partitions. Both are legitimate — only `moa` is *MCL*.

---

## CLI reference

```
moa <command> [options]

COMMANDS
  mcl <graph>       Cluster a labelled edge graph ("labelA labelB [weight]").
  cluster <fasta>   MinHash similarity graph over FASTA sequences, then MCL.
  doctor            Self-test every engine against the serial reference.
  info              Build, feature, and backend/environment details.
  version | help

COMMON OPTIONS
  -I, --inflation <f>   Inflation / granularity (default 2.0; higher = finer).
      --engine <e>      serial | rayon | hydra | gpu | auto  (default rayon).
  -o, --output <file>   Write clusters to a file (default: stdout).
  -P, --prune <n>       Prune number (default 10000; cut = 0.99999/n).
  -S, --select <n>      Max entries kept per column (default 1100).
  -R, --recover <n>     Recover number (default 1400).
      --max-iter <n>    Iteration cap (default 10000; convergence is far sooner).
  -q, --quiet           Suppress the per-run summary on stderr.

CLUSTER-ONLY OPTIONS
  -k <n>                k-mer size (default 7).
  -s <n>                Bottom-s sketch size (default 256; larger = less noise).
      --min-id <f>      Minimum identity for an edge (default 0.95).
      --len-cov <f>     Minimum length ratio, shorter/longer (default 0.70).
```

---

## The MinHash front-end

`moa cluster` turns sequences into a weighted graph without alignment:

1. **Sketch.** Each sequence becomes its bottom-`s` MinHash sketch of `k`-mers —
   the `s` smallest distinct 64-bit hashes. Default `k = 7`, `s = 256`. Sketching
   is done in parallel (one rayon task per sequence).
2. **Candidate pairs.** Sequences sharing a sketch hash are candidates; hashes
   shared by more than `max_bucket` sequences (low-complexity) are skipped so the
   candidate list never goes quadratic.
3. **Score & threshold.** Each candidate pair's Jaccard is estimated from the
   sketches and kept if it clears the cut-off derived from `--min-id` via the
   **Mash** relation `J = f/(2−f)`, `f = p^k`, and passes the `--len-cov` length
   ratio.
4. **Weight.** Edges are weighted by the **raw Jaccard**, not the Mash identity —
   the Jaccard contrast between a 99%- and an 80%-identical pair is ~8× versus
   ~1.2× for identity, and that contrast is what lets inflation resolve
   paralogous subfamilies inside a component.

Why `s = 256` and not less: the standard error of a bottom-`s` Jaccard estimate
near `J` is `sqrt(J(1−J)/s)`. At `s = 48`, `J ≈ 0.54` that is ±0.072 — a true
ortholog at the threshold is rejected ~half the time. At `s = 256` the same pair
is rejected under 2% of the time.

---

## Project layout

```
moa/
├── Cargo.toml            workspace root (release LTO profile, MSRV pins)
├── Cargo.lock            committed, pinned for Rust 1.75
├── examples/
│   ├── friends.abc       the canonical MCL example
│   └── demo.faa          a small multi-family FASTA
├── moa/                  the crate
│   ├── src/
│   │   ├── matrix.rs     CSC sparse matrix (build, loops, stochastic)
│   │   ├── mcl.rs        expansion/inflation loop, prune/select/recover, chaos
│   │   ├── interpret.rs  mclDag + mclInterpret (attractors → clusters)
│   │   ├── minhash.rs    bottom-s sketching, similarity graph, FASTA reader
│   │   ├── backend/
│   │   │   ├── mod.rs    engine dispatch + the shared column-compose kernel
│   │   │   ├── cpu.rs    serial + rayon expansion
│   │   │   ├── hydra.rs  HydraMPP distributed expansion
│   │   │   └── gpu.rs    wgpu dense-GEMM expansion
│   │   ├── art.rs        the moa mascot & banners
│   │   ├── lib.rs        pipelines, graph parsing, embedded friends graph
│   │   └── main.rs       CLI
│   └── tests/
│       └── integration.rs   the friends proof, engine equivalence, MinHash
└── vendor/
    └── hydra-mpp-core/   HydraMPP (vendored, path dependency)
```

### MSRV

The floor is **Rust 1.75**. The `wgpu 0.19` dependency tree's newest transitive
*patch* releases have crept past that (they require the unstable `edition2024`
cargo feature), so `Cargo.toml` pins `hashbrown = 0.14.5`, `indexmap = 2.2.6`,
`zerocopy = 0.7.35`, and `Cargo.lock` pins `rayon-core = 1.12.1`, behind the GPU
feature. Bump these together with the Rust floor, not before. Building
`--no-default-features` drops the GPU tree and all of its pins.

---

## Credits & provenance

* **MCL** — the algorithm and the reference implementation are Stijn van
  Dongen's (*Graph Clustering by Flow Simulation*, PhD thesis, Utrecht, 2000).
  `moa` is an independent Rust port validated against `mcl 22-282`; it bundles no
  MCL source.
* **HydraMPP** — the vendored distributed task engine is by Jose L. Figueroa III
  and Richard Allen White III (RAW Lab, github.com/raw-lab/HydraMPP).
* **wgpu / rayon / serde** — their respective authors.
* **`ratio-markov`** (comparison only) — used to contrast a different Rust
  Markov-clustering variant; not a dependency of `moa`.

Part of the RAW Lab open-source toolkit — www.rawlab.org · github.com/raw-lab

License: **CC-BY-NC-4.0**.
