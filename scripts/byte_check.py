import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
B, MOA, ENG = sys.argv[1], sys.argv[2], sys.argv[3]
cases = json.load(open(os.path.join(B, "manifest.json")))
def one(c):
    ref = open(os.path.join(B, "mcl_cache", f"{os.path.basename(c['graph'])}_I{c['I']}_P{c['P']}_S{c['S']}_R{c['R']}.txt"), "rb").read()
    out = subprocess.run([MOA, "mcl", c["graph"], "-I", str(c["I"]), "-P", str(c["P"]), "-S", str(c["S"]), "-R", str(c["R"]), "-q", "--engine", ENG], capture_output=True).stdout
    return out == ref
with ThreadPoolExecutor(2) as ex: r = list(ex.map(one, cases))
print(f"engine={ENG:<7} byte-identical to mcl output: {sum(r)}/{len(r)}")
