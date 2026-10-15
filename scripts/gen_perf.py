#!/usr/bin/env python3
"""Benchmark graphs. usage: gen_perf.py OUTDIR
  blast   — 60k proteins in planted families, BLAST-like reciprocal hits (both directions, asymmetric scores)
  dense   — one 15k-node component, avg degree ~40 (select/recover active at defaults)
  pangen  — 300k nodes in ~40k components (pangenome-style many families)
"""
import os, random, sys
out = sys.argv[1]; os.makedirs(out, exist_ok=True)

def blast(path, n, seed):
    r = random.Random(seed)
    fam, i = [], 0
    while i < n:
        k = min(n - i, max(2, int(r.paretovariate(1.3) * 3)))
        fam.append(range(i, i + k)); i += k
    with open(path, "w") as f:
        for F in fam:
            F = list(F)
            for a in F:
                for _ in range(min(len(F) - 1, 12)):
                    b = r.choice(F)
                    if b != a:
                        s = round(r.uniform(40, 400), 1)
                        f.write(f"p{a}\tp{b}\t{s}\n")
                        f.write(f"p{b}\tp{a}\t{round(s * r.uniform(.9, 1.1), 1)}\n")
                if r.random() < .05:  # spurious cross-family hit
                    b = r.randrange(n)
                    f.write(f"p{a}\tp{b}\t{round(r.uniform(20, 60), 1)}\n")

def dense(path, n, deg, comm, seed):
    r = random.Random(seed)
    with open(path, "w") as f:
        for a in range(n):
            for _ in range(deg // 2):
                b = (a // comm) * comm + r.randrange(comm) if r.random() < .85 else r.randrange(n)
                if b != a and b < n:
                    f.write(f"d{a}\td{b}\t{round(r.uniform(.5, 3), 3)}\n")

def pangen(path, n, seed):
    r = random.Random(seed)
    i = 0
    with open(path, "w") as f:
        while i < n:
            k = min(n - i, r.choice([1, 2, 3, 4, 6, 8, 10, 15, 25, 40]))
            for a in range(i, i + k):
                for b in range(a + 1, i + k):
                    if r.random() < .7:
                        f.write(f"g{a}\tg{b}\t{round(r.uniform(.6, 1), 3)}\n")
            i += k

blast(os.path.join(out, "blast60k.abc"), 60_000, 1)
dense(os.path.join(out, "dense15k.abc"), 15_000, 40, 300, 2)
pangen(os.path.join(out, "pangen300k.abc"), 300_000, 3)
for fn in sorted(os.listdir(out)):
    p = os.path.join(out, fn); print(fn, sum(1 for _ in open(p)), "lines", os.path.getsize(p) // 1024, "KiB")
