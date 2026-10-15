#!/usr/bin/env python3
"""Needs GNU time (/usr/bin/time).
usage: perf.py GRAPH name=cmd... ; '{g}' is replaced by the graph path. Reports wall, max RSS, cluster count, agreement with first tool."""
import os, subprocess, sys, tempfile, time
OUT = os.path.join(tempfile.mkdtemp(prefix="moa-perf-"), "out.txt")
g = sys.argv[1]; tools = [a.split("=", 1) for a in sys.argv[2:]]
def sets(t): return frozenset(frozenset(l.split("\t")) for l in t.strip("\n").split("\n") if l.strip())
ref = None
for name, cmd in tools:
    cmd = cmd.replace("{g}", g)
    t0 = time.time()
    p = subprocess.run(["/usr/bin/time", "-f", "%M", "sh", "-c", cmd + " 2>/dev/null >" + OUT], capture_output=True, text=True)
    dt = time.time() - t0
    rss = int(p.stderr.strip().split("\n")[-1]) / 1024
    s = sets(open(OUT).read())
    if ref is None: ref = s
    print(f"  {name:<16} {dt:8.2f} s   {rss:8.0f} MiB   {len(s):7d} clusters   {'= ref' if s == ref else 'DIFFERS'}", flush=True)
