#!/usr/bin/env bash
# Reproduce moa's fidelity claims against the reference mcl binary.
#
#   scripts/validate.sh            # build, generate the 167-case battery, score it
#
# Needs: cargo, python3, and mcl 22-282 on PATH (e.g. `apt install mcl`).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v mcl >/dev/null || { echo "mcl not on PATH (apt install mcl)"; exit 1; }
cargo build --release
MOA="$PWD/target/release/moa"
BAT="${BAT:-$(mktemp -d)/battery}"
python3 -I scripts/gen_battery.py "$BAT"
python3 -I scripts/run_battery.py "$BAT" "moa=$MOA"
for e in serial rayon hydra; do
  python3 -I scripts/byte_check.py "$BAT" "$MOA" "$e"
done
echo "battery kept in $BAT"
