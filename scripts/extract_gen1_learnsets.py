"""Extract gen 1 learnsets from learnset.ts for all Pokemon in species.json.
TODO: the generated set is a superset of the actual gen 1 learnsets."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent

with open(ROOT / "data" / "species.json") as f:
    species = json.load(f)

with open(ROOT / "data" / "moves.json") as f:
    moves_data = json.load(f)

def normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())

species_map = {normalize(name): name for name in species}
gen1_moves = {normalize(name): name for name in moves_data}

learnset_path = ROOT / "data" / "raw" / "learnset.ts"
lines = learnset_path.read_text(encoding="utf-8").splitlines()

move_re = re.compile(r'^\s{12}(\w+):\s*\[([^\]]*)\]')

result = {}
current_poke = None
in_learnset = False

for line in lines:
    poke_match = re.match(r'^\s{4}(\w+):\s*\{', line)
    if poke_match and not line.startswith(" " * 8):
        key = poke_match.group(1).lower()
        current_poke = species_map.get(key)
        in_learnset = False
        continue

    if re.match(r'^\s{8}learnset:\s*\{', line):
        in_learnset = True
        if current_poke and current_poke not in result:
            result[current_poke] = []
        continue

    if in_learnset and re.match(r'^\s{8}\},', line):
        in_learnset = False
        continue

    if in_learnset and current_poke:
        m = move_re.match(line)
        if m:
            move_name = m.group(1)
            entries_raw = m.group(2)
            entries = [e.strip().strip('"') for e in entries_raw.split(",") if e.strip().strip('"')]
            if "7V" in entries and move_name in gen1_moves:
                result[current_poke].append(gen1_moves[move_name])

result = {k: v for k, v in result.items() if v}
result = dict(sorted(result.items()))

out_path = ROOT / "data" / "gen1_learnsets.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print(f"Extracted learnsets for {len(result)} Pokemon -> {out_path}")
missing = set(species_map.values()) - set(result.keys())
if missing:
    print(f"Missing ({len(missing)}): {sorted(missing)}")
