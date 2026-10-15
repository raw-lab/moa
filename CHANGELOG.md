# Changelog

## 2.0.0

Merges the 1.0.0 and 1.1.0 lines and fixes the 24 bugs they shipped with. See [BUGS.md](BUGS.md)
for the full checklist with reproductions.

### Fidelity
- Output is byte-identical to `mcl 22-282` on all 167 validation cases, under every engine.
- `--abc` input is mirrored and symmetrised with **max**-merge, as mcl does (was: sum).
- Faithful `mclDag` + `mclInterpret` + `-overlap cut`, including system order
  and gathering unattached nodes (was: a union-find that merged systems).
- `mclvKBar` / `mclvSelectGqBar` semantics: select and recover keep ties.
- Nested recover restores the post-prune vector; emergency loop only without recovery.
- Strict `chaos < 1e-4` stop rule; `f32` cells with `f64` accumulation.
- Per-component runs are reconciled to mcl's global stop rule, so they are exact.

### Performance
- Fused per-column kernel: no unpruned `M²` is ever stored.
- Dense/sparse compose chosen per column; unrolled branchless scatter.
- rayon over components **and** nnz-balanced column blocks, chosen per iteration.
- Parallel parsing and symmetrisation; label hashing with a SplitMix finaliser.
- 1.16–8.9× faster than `mcl -te 2` on the benchmark graphs, with the same results.

### HydraMPP
- Depends on `hydra-mpp = "1.0"` from crates.io (was a vendored copy).
- Batch tasks for small components, column-block tasks for big ones, `Arc` iterands.
- `--hydra-host` / `--hydra-client` / `--hydra-slurm` pass-through, `moa worker`,
  SLURM-relaunched workers detected before reading input.

### MinHash
- Linear-time candidate index from 1.1.0, now built in parallel, plus 4 anchors per
  large bucket so large families come out as one cluster (was: fragmented).
- Rolling k-mer code; partial selection for long sequences; parallel scoring.
- FASTA reader upper-cases residues and drops non-letters; `-k 0` / `-s 0` rejected.

### CLI
- New `graph`, `worker`, `info`, `version --short`, per-command `help`, `--stats`,
  `-v` per-phase and per-iteration trace, `--color`, `-t`/`-te`, `-R`, `-pct`, `-L`, `--abc`.
- New `doctor`: friends proof, mcl-recorded regression corpus, bit-identical engines,
  MinHash family check, live comparison when `mcl` is installed.
- Bird-face + MOA wordmark banner with 256-colour palette (respects `NO_COLOR`).

### Removed
- The dense GPU engine (O(n²), 2,048-node cap, `f32` accumulation).
- Rust 1.75 dependency pins (MSRV is now 1.80).

## 1.1.0
Per-component MCL; linear-time candidate index; zero dependencies.

## 1.0.0
rayon / HydraMPP / GPU expansion engines; ASCII art; MinHash front-end.
