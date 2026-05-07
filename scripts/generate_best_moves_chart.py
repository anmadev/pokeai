"""
Generate two charts of best moves by effective damage:

1. best_moves_by_type_chart.json — best moves for each (offensive type, defensive type combo)
2. best_moves.json — best moves per Pokémon for each defensive type combo, split by
   physical vs special (using the Gen 1 type-based split)

Effective BP = bp * STAB * type_effectiveness * hits
  STAB = 1.5 if move type matches one of the Pokémon's own types, else 1.0
  type_effectiveness = product of type_chart[move_type][def_type] for each defensive type
  hits = min or max number of hits (from multihit field, defaults to 1)

Gen 1 physical/special split:
  Special types: Fire, Water, Grass, Electric, Ice, Psychic, Dragon
  Physical types: everything else (Normal, Fighting, Flying, Ground, Rock, Bug, Ghost, Poison)
"""

import json
from itertools import combinations
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

with open(DATA_DIR / "moves.json") as f:
    moves = json.load(f)

with open(DATA_DIR / "type_chart.json") as f:
    type_chart = json.load(f)

with open(DATA_DIR / "types.json") as f:
    types = json.load(f)

with open(DATA_DIR / "species.json") as f:
    species = json.load(f)

with open(DATA_DIR / "gen1_learnsets.json") as f:
    learnsets = json.load(f)

GEN1_SPECIAL_TYPES = {"Fire", "Water", "Grass", "Electric", "Ice", "Psychic", "Dragon"}


def get_effectiveness(move_type: str, def_types: list[str]) -> float:
    if move_type not in type_chart:
        return None
    eff = 1.0
    for dt in def_types:
        eff *= type_chart[move_type].get(dt, 1.0)
    return eff


def compute_damage_range(move_data: dict, stab_types: list[str], def_types: list[str]):
    """
    Return (min_effective_bp, max_effective_bp) or (None, None) if not a damaging move.
    WARNING: it is an approximation of the damage range, since it does not take 
    into account the defender's stats or the attacker's level. However, it is 
    sufficient for move selection since these factors are the same for all moves 
    being compared. The idea is that once the best move is selected based on 
    effective BP, the actual damage range can be computed more accurately with 
    the full formula.
    """
    bp = move_data.get("bp", 0)
    category = move_data.get("category")
    move_type = move_data.get("type", "Normal")

    if category == "Status" or bp == 0:
        return None, None

    eff = get_effectiveness(move_type, def_types)
    if eff is None:
        return None, None

    stab = 1.5 if move_type in stab_types else 1.0

    multihit = move_data.get("multihit", 1)
    if isinstance(multihit, list):
        min_hits, max_hits = multihit
    else:
        min_hits = max_hits = int(multihit)

    min_dmg = bp * stab * eff * min_hits
    max_dmg = bp * stab * eff * max_hits
    return min_dmg, max_dmg


def get_move_category_gen1(move_data: dict) -> str:
    """Return 'physical' or 'special' based on the Gen 1 type-based split."""
    move_type = move_data.get("type", "Normal")
    return "special" if move_type in GEN1_SPECIAL_TYPES else "physical"


# Build all defensive type combinations: singles + pairs
def_combos: list[list[str]] = [[t] for t in types]
for t1, t2 in combinations(types, 2):
    def_combos.append([t1, t2])

result = {}

for off_type in types:
    result[off_type] = {}

    for def_combo in def_combos:
        key = "/".join(def_combo)

        best_min_move = None
        best_min_dmg = -1.0
        best_max_move = None
        best_max_dmg = -1.0

        for move_name, move_data in moves.items():
            if move_name == "(No Move)":
                continue
            min_dmg, max_dmg = compute_damage_range(move_data, [off_type], def_combo)
            if min_dmg is None:
                continue

            if min_dmg > best_min_dmg:
                best_min_dmg = min_dmg
                best_min_move = move_name

            if max_dmg > best_max_dmg:
                best_max_dmg = max_dmg
                best_max_move = move_name

        result[off_type][key] = {
            "best_min": {
                "move": best_min_move,
                "effective_bp": best_min_dmg if best_min_dmg >= 0 else None,
            },
            "best_max": {
                "move": best_max_move,
                "effective_bp": best_max_dmg if best_max_dmg >= 0 else None,
            },
        }

out_path = DATA_DIR / "best_moves_by_type_chart.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Written to {out_path}")
print(f"  Offensive types: {len(result)}")
print(f"  Defensive combos per type: {len(def_combos)}")

# ── Per-Pokémon best moves ────────────────────────────────────────────────────
# Keyed by Pokémon -> defensive type combo -> physical/special -> best_min/best_max

pokemon_result = {}

for pokemon_name, learnset in learnsets.items():
    species_data = species.get(pokemon_name)
    if species_data is None:
        continue

    off_types: list[str] = species_data.get("types", [])
    pokemon_result[pokemon_name] = {}

    for def_combo in def_combos:
        key = "/".join(def_combo)

        bests: dict[str, dict] = {
            "physical": {"best_min": (None, -1.0), "best_max": (None, -1.0)},
            "special":  {"best_min": (None, -1.0), "best_max": (None, -1.0)},
        }

        for move_name in learnset:
            move_data = moves.get(move_name)
            if move_data is None:
                continue
            if move_name == "(No Move)":
                continue

            min_dmg, max_dmg = compute_damage_range(move_data, off_types, def_combo)
            if min_dmg is None:
                continue

            cat = get_move_category_gen1(move_data)

            if min_dmg > bests[cat]["best_min"][1]:
                bests[cat]["best_min"] = (move_name, min_dmg)
            if max_dmg > bests[cat]["best_max"][1]:
                bests[cat]["best_max"] = (move_name, max_dmg)

        def _entry(name, dmg):
            return {"move": name, "effective_bp": dmg if name is not None else None}

        pokemon_result[pokemon_name][key] = {
            "physical": {
                "best_min": _entry(*bests["physical"]["best_min"]),
                "best_max": _entry(*bests["physical"]["best_max"]),
            },
            "special": {
                "best_min": _entry(*bests["special"]["best_min"]),
                "best_max": _entry(*bests["special"]["best_max"]),
            },
        }

out_path2 = DATA_DIR / "best_moves.json"
with open(out_path2, "w") as f:
    json.dump(pokemon_result, f, indent=2)

print(f"Written to {out_path2}")
print(f"  Pokémon: {len(pokemon_result)}")
print(f"  Defensive combos per Pokémon: {len(def_combos)}")
