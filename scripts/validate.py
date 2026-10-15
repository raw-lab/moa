#!/usr/bin/env python3
"""
Reproduce moa's validation:

  1. moa (any engine) vs the reference `mcl` binary  -> exact partition match
  2. moa serial == rayon == hydra == gpu             -> engine equivalence

Usage:
    python3 scripts/validate.py [path/to/moa]

`moa` defaults to ./target/release/moa. The `mcl` comparison is skipped if the
`mcl` binary is not on PATH (everything else still runs). Graphs are generated
here, so no data files are needed.
"""
import os, sys, shutil, subprocess, tempfile, random

MOA = sys.argv[1] if len(sys.argv) > 1 else "./target/release/moa"
HAVE_MCL = shutil.which("mcl") is not None
INFLATIONS = [1.2, 1.4, 2.0, 3.0, 5.0]


def gen_graphs(d):
    def er(n, p, wlo, whi, seed, path):
        random.seed(seed)
        with open(path, "w") as f:
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        f.write(f"n{i} n{j} {round(random.uniform(wlo, whi), 3)}\n")

    def planted(sizes, pin, pout, seed, path):
        random.seed(seed)
        off, s = [], 0
        for z in sizes:
            off.append(s); s += z
        def comm(v):
            for c, o in enumerate(off):
                if o <= v < o + sizes[c]:
                    return c
        with open(path, "w") as f:
            for i in range(s):
                for j in range(i + 1, s):
                    p = pin if comm(i) == comm(j) else pout
                    if random.random() < p:
                        f.write(f"n{i} n{j} 1\n")

    er(30, 0.15, 1, 5, 1, f"{d}/g1.abc")
    er(50, 0.08, 0.1, 1, 2, f"{d}/g2.abc")
    er(20, 0.30, 1, 10, 3, f"{d}/g3.abc")
    planted([12, 15, 10], 0.6, 0.02, 7, f"{d}/g4.abc")
    planted([20, 20, 20, 20], 0.5, 0.01, 9, f"{d}/g5.abc")
    return [f"{d}/g{i}.abc" for i in range(1, 6)]


def sets(text):
    return frozenset(frozenset(l.split()) for l in text.strip().splitlines() if l.split())


def run(cmd):
    return sets(subprocess.run(cmd, capture_output=True, text=True).stdout)


def moa(f, I, eng="rayon"):
    return run([MOA, "mcl", f, "-I", str(I), "--engine", eng, "-q"])


def mcl(f, I):
    return run(["mcl", f, "--abc", "-I", str(I), "-o", "-"])


def main():
    if not os.path.exists(MOA):
        sys.exit(f"moa binary not found at {MOA} (pass a path, or `cargo build --release`)")
    d = tempfile.mkdtemp(prefix="moa-val-")
    graphs = gen_graphs(d)

    if HAVE_MCL:
        ok = tot = 0
        print("== moa vs reference mcl ==")
        for g in graphs:
            for I in INFLATIONS:
                a, b = mcl(g, I), moa(g, I)
                tot += 1; ok += (a == b)
                if a != b:
                    print(f"  DIFF {os.path.basename(g)} -I {I}: mcl={len(a)} moa={len(b)}")
        print(f"  exact partition match: {ok}/{tot}")
    else:
        print("== moa vs reference mcl == (skipped: `mcl` not on PATH)")

    print("== cross-engine equivalence (serial==rayon==hydra==gpu) ==")
    ok = tot = 0
    for g in graphs:
        for I in [1.4, 2.0, 3.0]:
            ref = moa(g, I, "serial")
            tot += 1
            agree = all(moa(g, I, e) == ref for e in ["rayon", "hydra", "gpu"])
            ok += agree
            if not agree:
                print(f"  DISAGREE {os.path.basename(g)} -I {I}")
    print(f"  all engines agree: {ok}/{tot}")
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
