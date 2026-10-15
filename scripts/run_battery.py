#!/usr/bin/env python3
"""Run every manifest case through reference mcl and one or more moa binaries.

usage: run_battery.py BATTERY_DIR name=/path/to/moa[:style] ...
  style: v100 (moa mcl g -I -P -S -R), v110 (moa mcl g -I -P -S; no -R),
         new  (moa mcl g -I -P -S -R --pct)       default: new
Extra env: MOA_ENGINE=rayon|hydra|serial (new style only)
Prints per-category exact-match tables and writes BATTERY_DIR/results_<name>.json
"""
import json, os, subprocess, sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

B = sys.argv[1]
tools = []
for a in sys.argv[2:]:
    name, rest = a.split("=", 1)
    path, _, style = rest.partition(":")
    tools.append((name, path, style or "new"))
cases = json.load(open(os.path.join(B, "manifest.json")))


def sets(text):
    return frozenset(frozenset(l.split("\t")) for l in text.strip("\n").split("\n") if l.strip())


cache_dir = os.path.join(B, "mcl_cache"); os.makedirs(cache_dir, exist_ok=True)


def mcl_ref(c):
    key = f"{os.path.basename(c['graph'])}_I{c['I']}_P{c['P']}_S{c['S']}_R{c['R']}.txt"
    kp = os.path.join(cache_dir, key)
    if not os.path.exists(kp):
        out = subprocess.run(["mcl", c["graph"], "--abc", "-I", str(c["I"]), "-P", str(c["P"]), "-S", str(c["S"]),
                              "-R", str(c["R"]), "-o", kp], capture_output=True, text=True)
        if out.returncode:
            raise RuntimeError(out.stderr)
    return sets(open(kp).read())


def run_tool(path, style, c):
    cmd = [path, "mcl", c["graph"], "-I", str(c["I"]), "-P", str(c["P"]), "-S", str(c["S"])]
    if style == "v110":
        if c["R"] != 1400:
            return None                      # v1.1.0 has no -R option
    else:
        cmd += ["-R", str(c["R"])]
    if style == "new":
        cmd += ["-q"]
        if os.environ.get("MOA_ENGINE"):
            cmd += ["--engine", os.environ["MOA_ENGINE"]]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode:
        return "ERR:" + out.stderr[-300:]
    return sets(out.stdout)


with ThreadPoolExecutor(2) as ex:
    refs = list(ex.map(mcl_ref, cases))

for name, path, style in tools:
    per = defaultdict(lambda: [0, 0, 0])   # match, total, n/a
    diffs = []
    def job(i):
        return i, run_tool(path, style, cases[i])
    with ThreadPoolExecutor(2) as ex:
        results = list(ex.map(job, range(len(cases))))
    for i, got in results:
        c = cases[i]; d = per[c["cat"]]
        if got is None:
            d[2] += 1; continue
        d[1] += 1
        if got == refs[i]:
            d[0] += 1
        else:
            diffs.append({**c, "mcl": len(refs[i]), "tool": (len(got) if not isinstance(got, str) else got)})
    tm = sum(v[0] for v in per.values()); tt = sum(v[1] for v in per.values())
    print(f"\n== {name}: {tm}/{tt} exact partition matches vs mcl 22-282 ==")
    for cat in dict.fromkeys(c["cat"] for c in cases):
        m, t, na = per[cat]
        print(f"   {cat:<9} {m:>3}/{t:<3}" + (f"   ({na} n/a)" if na else ""))
    json.dump({"per": per, "diffs": diffs}, open(os.path.join(B, f"results_{name}.json"), "w"), indent=1)
    for d in diffs[:6]:
        print("   diff:", os.path.basename(d["graph"]), f"-I {d['I']} -P {d['P']} -S {d['S']} -R {d['R']}",
              f"mcl={d['mcl']} {name}={d['tool']}")
