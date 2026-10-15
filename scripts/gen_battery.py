#!/usr/bin/env python3
"""Generate the moa fidelity battery: graphs + a manifest of (graph, inflation, scheme) cases.

Categories (each targets a code path the old 25-case check never reached):
  basic      small ER / planted graphs (the original check)
  weighted   real-valued weights
  recip      every edge given twice (a b w1 / b a w2) + duplicates  -> abc MAX-merge
  loops      input self-loops                                        -> loop reset
  multi      many connected components of mixed size                 -> per-component MCL
  tight      tight -P/-S/-R schemes so select AND recover fire, with ties
  noselect   -S 0 / -R 0 schemes (prune-only, select-without-recover)
  large      n=2500, dense two-hop neighbourhoods: default scheme select fires
  overlap    graphs on which mcl itself reports interpretation overlap
Usage: gen_battery.py OUTDIR
"""
import json, os, random, subprocess, sys

OUT = sys.argv[1]
G = os.path.join(OUT, "graphs")
os.makedirs(G, exist_ok=True)
cases = []
SMALL_I = [1.2, 1.4, 2.0, 3.0, 5.0]
DEFAULT = {"P": 10000, "S": 1100, "R": 1400}


def write(name, lines):
    p = os.path.join(G, name)
    with open(p, "w") as f:
        f.write("".join(lines))
    return p


def er_edges(n, p, rnd, wfn, pre="n"):
    return [(f"{pre}{i}", f"{pre}{j}", wfn()) for i in range(n) for j in range(i + 1, n) if rnd.random() < p]


def planted_edges(sizes, pin, pout, rnd, wfn, pre="n"):
    off, s = [], 0
    for z in sizes:
        off.append(s); s += z
    comm = [c for c, z in enumerate(sizes) for _ in range(z)]
    return [(f"{pre}{i}", f"{pre}{j}", wfn()) for i in range(s) for j in range(i + 1, s)
            if rnd.random() < (pin if comm[i] == comm[j] else pout)]


def fmt(edges, weighted=True):
    return [f"{a}\t{b}\t{w}\n" if weighted else f"{a} {b}\n" for a, b, w in edges]


def add(cat, path, infls, scheme=DEFAULT):
    for I in infls:
        cases.append({"cat": cat, "graph": path, "I": I, **scheme})


one = lambda: 1
# --- basic (the original five) ---------------------------------------------
r = random.Random(1); add("basic", write("b1.abc", fmt(er_edges(30, .15, r, lambda: round(r.uniform(1, 5), 3)))), SMALL_I)
r = random.Random(2); add("basic", write("b2.abc", fmt(er_edges(50, .08, r, lambda: round(r.uniform(.1, 1), 3)))), SMALL_I)
r = random.Random(3); add("basic", write("b3.abc", fmt(er_edges(20, .30, r, lambda: round(r.uniform(1, 10), 3)))), SMALL_I)
r = random.Random(7); add("basic", write("b4.abc", fmt(planted_edges([12, 15, 10], .6, .02, r, one), False)), SMALL_I)
r = random.Random(9); add("basic", write("b5.abc", fmt(planted_edges([20] * 4, .5, .01, r, one), False)), SMALL_I)
# --- weighted ----------------------------------------------------------------
for s in range(3):
    r = random.Random(100 + s)
    e = planted_edges([25, 30, 18, 27], .35, .03, r, lambda: round(r.lognormvariate(0, 1), 4))
    add("weighted", write(f"w{s}.abc", fmt(e)), SMALL_I)
# --- reciprocal / duplicate edges ----------------------------------------------
for s in range(3):
    r = random.Random(200 + s)
    e = planted_edges([15, 20, 12], .45, .04, r, lambda: round(r.uniform(.5, 9), 3))
    lines = []
    for a, b, w in e:
        lines.append(f"{a}\t{b}\t{w}\n")
        lines.append(f"{b}\t{a}\t{round(w * r.uniform(.2, 3), 3)}\n")   # asymmetric reverse
        if r.random() < .2:
            lines.append(f"{a}\t{b}\t{round(r.uniform(.5, 9), 3)}\n")    # plain duplicate
    r.shuffle(lines)
    add("recip", write(f"r{s}.abc", lines), SMALL_I)
# --- input self loops ------------------------------------------------------------
for s in range(2):
    r = random.Random(300 + s)
    e = planted_edges([14, 16, 11], .5, .05, r, lambda: round(r.uniform(1, 4), 3))
    nodes = sorted({x for a, b, _ in e for x in (a, b)})
    e += [(v, v, round(r.uniform(.1, 20), 3)) for v in nodes if r.random() < .4]
    add("loops", write(f"l{s}.abc", fmt(e)), SMALL_I)
# --- many components -------------------------------------------------------------
for s in range(3):
    r = random.Random(400 + s)
    e = []
    for c in range(40):
        k = r.choice([2, 2, 3, 4, 5, 8, 13, 21, 34])
        if k <= 3:
            e += [(f"c{c}_0", f"c{c}_{i}", round(r.uniform(1, 5), 2)) for i in range(1, k)]
        else:
            e += planted_edges([k // 2, k - k // 2], .7, .15, r, lambda: round(r.uniform(1, 5), 2), pre=f"c{c}_")
    add("multi", write(f"m{s}.abc", fmt(e)), SMALL_I)
# --- tight schemes: select + recover fire, lots of ties (unweighted) -------------
TIGHT = [{"P": 500, "S": 15, "R": 25}, {"P": 300, "S": 8, "R": 40}, {"P": 2000, "S": 20, "R": 22}]
for s in range(3):
    r = random.Random(500 + s)
    e = planted_edges([40, 50, 35, 45], .3, .02, r, one)
    p = write(f"t{s}.abc", fmt(e, False))
    for sc in TIGHT:
        add("tight", p, [1.4, 2.0, 3.0], sc)
    r = random.Random(550 + s)
    e = planted_edges([40, 50, 35], .3, .03, r, lambda: round(r.uniform(.5, 3), 2))
    p = write(f"tw{s}.abc", fmt(e))
    for sc in TIGHT:
        add("tight", p, [1.4, 2.0, 3.0], sc)
# --- prune-only / select-without-recover ------------------------------------------
NOSEL = [{"P": 200, "S": 0, "R": 0}, {"P": 1000, "S": 12, "R": 0}, {"P": 50, "S": 0, "R": 0}]
for s in range(2):
    r = random.Random(600 + s)
    p = write(f"ns{s}.abc", fmt(planted_edges([30, 40, 30], .35, .03, r, lambda: round(r.uniform(.5, 3), 2))))
    for sc in NOSEL:
        add("noselect", p, [1.4, 2.0, 3.0], sc)
# --- large: default scheme select fires -----------------------------------------
r = random.Random(700)
add("large", write("big0.abc", fmt(planted_edges([500] * 5, .06, .004, r, one), False)), [1.4, 2.0, 3.0])
r = random.Random(701)
add("large", write("big1.abc", fmt(planted_edges([400, 600, 500, 300, 700], .05, .003, r, lambda: round(r.uniform(.2, 4), 2)))), [1.4, 2.0, 3.0])
# --- overlap: search graphs where mcl reports interpretation overlap -------------
found, tries = 0, 0
while found < 6 and tries < 4000:
    tries += 1
    r = random.Random(10_000 + tries)
    n = r.randint(6, 14)
    e = er_edges(n, r.uniform(.2, .5), r, lambda: r.choice([1, 1, 1, 2, 3]))
    if not e:
        continue
    p = write(f"ov{found}.abc", fmt(e))
    hits = []
    for I in [1.4, 2.0, 3.0, 5.0]:
        res = subprocess.run(["mcl", p, "--abc", "-I", str(I), "-o", "/dev/null"], capture_output=True, text=True)
        if "instances of overlap" in res.stderr:
            hits.append(I)
    if hits:
        add("overlap", p, hits)
        found += 1
    else:
        os.remove(p)

json.dump(cases, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
from collections import Counter
print(len(cases), "cases", dict(Counter(c["cat"] for c in cases)), f"(overlap search: {tries} tries)")
