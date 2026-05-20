"""
Observation vector builder v2 for Gen 1 random battles.

Encoding convention (applied universally):
  TRUE    =  1.0   known true / active
  FALSE   = -1.0   known false / inactive
  UNKNOWN =  0.0   no information available

  [0, 1]  → [-1, 1]:   norm(value) = value * 2.0 - 1.0
  [-6, 6] → [-1, 1]:   stage / 6.0
  [-4, 4] → [-1, 1]:   priority / 4.0

Species encoding:
  Dropped for v1. The network infers species identity from base stats,
  types, and observed moves, which carry all strategically relevant
  information.
  TODO MovesetPrior v2: re-introduce species as a 151-dim one-hot
  to condition unknown move slots on the legal Gen 1 learnset.
  Source: server/data/learnsets.js

Total vector size: 2623 floats.

Layout:
  [0        :  246]  Our active Pokémon         (246)
  [246      : 1311]  Our bench * 5              (213 each = 1065)
  [1311     : 1557]  Opponent active Pokémon    (246)
  [1557     : 2622]  Opponent bench * 5         (213 each = 1065)
  [2622     : 2623]  Turn                       (1)
"""

from __future__ import annotations

import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon

# ── Encoding constants ───────────────────────────────────────────────

TRUE    =  1.0
FALSE   = -1.0
UNKNOWN =  0.0


def norm(value: float) -> float:
    """Maps a value in [0, 1] to [-1, 1]."""
    return float(np.clip(value * 2.0 - 1.0, -1.0, 1.0))


def norm_range(value: float, lo: float, hi: float) -> float:
    """Maps a value in [lo, hi] to [-1, 1]."""
    if hi == lo:
        return UNKNOWN
    return float(np.clip((value - lo) / (hi - lo) * 2.0 - 1.0, -1.0, 1.0))


# ── Gen 1 constants ──────────────────────────────────────────────────

GEN1_TYPES: list[str] = [
    "NORMAL", "FIRE", "WATER", "ELECTRIC", "GRASS", "ICE",
    "FIGHTING", "POISON", "GROUND", "FLYING", "PSYCHIC",
    "BUG", "ROCK", "GHOST", "DRAGON",
]  # 15 types in Gen 1 (no Dark/Steel/Fairy)
TYPE_TO_IDX: dict[str, int] = {t: i for i, t in enumerate(GEN1_TYPES)}
N_TYPES = len(GEN1_TYPES)  # 15

# Gen 1: no TOX (badly poisoned) - PSN covers both regular and toxic poison
GEN1_STATUS: list[str] = ["BRN", "FRZ", "PAR", "PSN", "SLP"]
STATUS_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(GEN1_STATUS)}
N_STATUS_CONDITIONS = len(GEN1_STATUS)  # 5
N_STATUS_ONEHOT = N_STATUS_CONDITIONS + 1  # 6: index 5 = healthy

MAX_STATUS_TURN = 7.0  # maximum number of turns a status condition can last in Gen 1

GEN1_STAT_KEYS: list[str] = ["hp", "atk", "def", "spa", "spe"]
N_STATS = len(GEN1_STAT_KEYS)  # 5

STAT_NORM: dict[str, float] = {
    "hp":  700.0,
    "atk": 400.0,
    "def": 400.0,
    "spa": 400.0,
    "spe": 400.0,
}

# Gen 1 boosts: unified special (spa covers both sp.atk and sp.def)
GEN1_BOOST_KEYS: list[str] = ["atk", "def", "spe", "spa", "accuracy", "evasion"]
N_BOOSTS = len(GEN1_BOOST_KEYS)  # 6

# Volatile effects applicable in Gen 1.
# next choose_move call; the agent never observes it as persistent state.
GEN1_VOLATILE_EFFECTS: list[str] = [
    "CONFUSION",
    "LEECHSEED",
    "SUBSTITUTE",
    "INVULNERABILITY",   # mid-Dig / mid-Fly
    "PARTIALLYTRAPPED",  # Wrap / Bind / Clamp / Fire Spin
    "THRASH",            # locked into Thrash / Petal Dance
    "RAGE",              # locked into Rage, boosts on hit
    "FOCUSENERGY",       # crit boost (bugged in Gen 1 but still encodable)
    "BIDE",              # storing damage to release
]
VOLATILE_SIZE = len(GEN1_VOLATILE_EFFECTS)  # 9

# ── Feature size constants ───────────────────────────────────────────
#
# Computed bottom-up: any change to a sub-component propagates
# automatically. The assert in build_observation catches drift.

MOVE_SIZE = (
    1                  # exists
    + 1                # is_disabled
    + 3                # category one-hot: physical / special / status
    + N_TYPES          # type one-hot = 15
    + 1                # priority / 4
    + 1                # accuracy
    + 2                # n_hit: (min, max) each / 5
    + 1                # base_power / 150
    + 1                # crit_ratio / 6
    + 1                # pp_fraction
    + 1                # drain
    + 1                # heal
    + 1                # recoil
    + 1                # bypasses_accuracy
    + N_STATS          # self_boosts: one per Gen 1 stat = 5
    + N_STATS          # boosts on opponent = 5
    + N_STATUS_ONEHOT  # inflicted status one-hot = 6
)
# 1+1+3+15+1+1+2+1+1+1+1+1+1+5+5+6 = 46

N_MOVE_SLOTS = 4
MOVES_SIZE = N_MOVE_SLOTS * MOVE_SIZE  # 184

BASE_POKEMON_SIZE = (
    # species dropped - see module docstring
    N_TYPES           # types binary vector = 15
    + N_STATS         # base/effective stats = 5
    + 1               # hp_fraction
    + N_STATUS_ONEHOT # status one-hot = 6
    + 1               # status_counter
    + MOVES_SIZE      # 184
)
# 15+5+1+6+1+184 = 212

ACTIVE_EXTRA_SIZE = (
    N_BOOSTS        # 6: atk/def/spe/spa/accuracy/evasion
    + VOLATILE_SIZE # 9: volatile condition binary flags
    + 1             # must_recharge
    + 1             # first_turn
    + 1             # last_move_slot (normalized slot index)
    + 1             # is_preparing
    + N_TYPES       # preparing_move_type one-hot = 15
)
# 6+9+1+1+1+1+15 = 34

BENCH_EXTRA_SIZE = 1  # revealed flag

ACTIVE_POKEMON_SIZE = BASE_POKEMON_SIZE + ACTIVE_EXTRA_SIZE  # 212 + 34 = 246
BENCH_POKEMON_SIZE  = BASE_POKEMON_SIZE + BENCH_EXTRA_SIZE   # 212 + 1  = 213

N_BENCH_SLOTS = 5

OBS_SIZE = (
    ACTIVE_POKEMON_SIZE                   # our active:     246
    + N_BENCH_SLOTS * BENCH_POKEMON_SIZE  # our bench:     1065
    + ACTIVE_POKEMON_SIZE                 # opp active:     246
    + N_BENCH_SLOTS * BENCH_POKEMON_SIZE  # opp bench:     1065
    + 1                                   # turn:             1
)
# 246 + 1065 + 246 + 1065 + 1 = 2623


# ── Individual encoding functions ────────────────────────────────────

def encode_types(pokemon: Pokemon, known: bool = True) -> np.ndarray:
    """
    15-dim binary vector.
    TRUE if the Pokémon has that type, FALSE otherwise.
    Dual-type Pokémon have two TRUE positions.
    UNKNOWN for unrevealed Pokémon.
    """
    if not known or pokemon is None:
        return np.full(N_TYPES, UNKNOWN, dtype=np.float32)
    vec = np.full(N_TYPES, FALSE, dtype=np.float32)
    for ptype in (pokemon.types or []):
        if ptype is None:
            continue  # just in case of malformed data
        name = ptype.name.upper() if hasattr(ptype, "name") else str(ptype).upper()
        idx = TYPE_TO_IDX.get(name, -1)
        if idx >= 0:
            vec[idx] = TRUE
    return vec


def encode_stats(pokemon: Pokemon, known: bool = True) -> np.ndarray:
    """
    5-float normalized stat vector.
    Our Pokémon: uses actual current stats when available.
    Opponent: estimated from base stats assuming max DVs / StatExp,
              using the same formula as the damage calculator.
    """
    if not known or pokemon is None:
        return np.full(N_STATS, UNKNOWN, dtype=np.float32)
    vec = np.full(N_STATS, UNKNOWN, dtype=np.float32)
    for i, key in enumerate(GEN1_STAT_KEYS):
        val: int | None = None
        try:
            if getattr(pokemon, "stats", None):
                val = pokemon.stats.get(key)
        except AttributeError:
            pass
        except Exception:
            vec[i] = norm(val / STAT_NORM[key])
            print(f"non-existing stat accessed: {key}")
            raise
    return vec


def encode_hp_fraction(pokemon: Pokemon, known: bool = True) -> float:
    if not known or pokemon is None:
        return UNKNOWN
    return norm(pokemon.current_hp_fraction)


def encode_status(pokemon: Pokemon, known: bool = True) -> np.ndarray:
    """
    6-dim one-hot.
    Indices 0-4: BRN / FRZ / PAR / PSN / SLP.
    Index 5: healthy (no status condition).

    The explicit healthy dimension disambiguates UNKNOWN (all zeros)
    from confirmed healthy (index 5 = TRUE, rest = FALSE).
    This matters for opponent Pokémon where the two cases should
    produce different agent behaviour.
    """
    if not known or pokemon is None:
        return np.full(N_STATUS_ONEHOT, UNKNOWN, dtype=np.float32)
    vec = np.full(N_STATUS_ONEHOT, FALSE, dtype=np.float32)
    st = pokemon.status
    if st is None:
        vec[N_STATUS_CONDITIONS] = TRUE  # index 5: healthy
    else:
        name = st.name.upper() if hasattr(st, "name") else str(st).upper()
        idx = STATUS_TO_IDX.get(name, N_STATUS_CONDITIONS)
        vec[idx] = TRUE
    return vec


def encode_status_counter(pokemon: Pokemon, known: bool = True) -> float:
    """
    Normalized sleep turn counter (max 7 turns in Gen 1).
    Returns UNKNOWN if not applicable or attribute unavailable.
    """
    if not known or pokemon is None:
        return UNKNOWN
    counter = pokemon.status_counter or None
    if counter is None:
        return UNKNOWN
    return norm(min(int(counter) / MAX_STATUS_TURN, 1.0))


def encode_move(
    move: Move | None,
    opponent: Pokemon | None,
    known: bool = True,
    is_disabled: bool = False,
) -> np.ndarray:
    """
    46-float encoding for a single move slot.

    known=False → entire slot is UNKNOWN.
                  MovesetPrior v2 hook: replace with the average
                  feature vector of remaining legal Gen 1 moves for
                  the opponent's species.
                  Source: server/data/learnsets.js

    move=None   → exists=FALSE, all other features=FALSE (empty slot).
    is_disabled → sourced from per-Pokémon DISABLE effect check.
    """
    if not known:
        return np.full(MOVE_SIZE, UNKNOWN, dtype=np.float32)

    vec = np.full(MOVE_SIZE, FALSE, dtype=np.float32)
    idx = 0

    # exists
    vec[idx] = TRUE if move is not None else FALSE
    idx += 1

    # is_disabled
    vec[idx] = TRUE if is_disabled else FALSE
    idx += 1

    if move is None:
        # Remaining features stay FALSE - empty slot, fully known
        return vec

    # category (3-dim one-hot: physical / special / status)
    if hasattr(move, "category") and move.category:
        cat = move.category.name.upper() if hasattr(move.category, "name") else ""
        if "PHYSICAL" in cat:
            vec[idx] = TRUE
        elif "SPECIAL" in cat:
            vec[idx + 1] = TRUE
        else:
            vec[idx + 2] = TRUE
    idx += 3

    # type (15-dim one-hot)
    if hasattr(move, "type") and move.type:
        tname = move.type.name.upper() if hasattr(move.type, "name") else ""
        tidx = TYPE_TO_IDX.get(tname, -1)
        if tidx >= 0:
            vec[idx + tidx] = TRUE
    idx += N_TYPES

    # priority / 4
    vec[idx] = (getattr(move, "priority", 0) or 0) / 4.0
    idx += 1

    # accuracy
    acc = getattr(move, "accuracy", True)
    if acc is True or acc is None:
        vec[idx] = TRUE  # always hits (e.g. Swift)
    else:
        raw = float(acc)
        vec[idx] = norm(raw / 100.0 if raw > 1.0 else raw)
    idx += 1

    # n_hit: (min_hits, max_hits) each normalized by 5
    nhit = getattr(move, "n_hit", None)
    if nhit and isinstance(nhit, (tuple, list)) and len(nhit) == 2:
        vec[idx]     = norm(nhit[0] / 5.0)
        vec[idx + 1] = norm(nhit[1] / 5.0)
    else:
        vec[idx] = vec[idx + 1] = norm(1 / 5.0)  # single hit
    idx += 2

    # base_power / 150
    bp = getattr(move, "base_power", 0) or 0
    vec[idx] = norm(min(bp / 150.0, 1.0))
    idx += 1

    # crit_ratio / 6
    crit = getattr(move, "crit_ratio", 0) or 0
    vec[idx] = norm(min(crit / 6.0, 1.0))
    idx += 1

    # pp_fraction
    cur_pp = getattr(move, "current_pp", None)
    max_pp = getattr(move, "max_pp", None)
    if cur_pp is not None and max_pp and max_pp > 0:
        vec[idx] = norm(cur_pp / max_pp)
    else:
        vec[idx] = UNKNOWN
    idx += 1

    # drain / heal / recoil - floats in [0, 1]
    for attr in ("drain", "heal", "recoil"):
        val = abs(float(getattr(move, attr, 0.0) or 0.0))
        vec[idx] = norm(min(val, 1.0))
        idx += 1

    # bypasses_accuracy (Swift-like moves in Gen 1)
    # TODO: verify correct attribute name in your poke-env version
    bypasses = getattr(move, "ignore_defensive", False) or False
    vec[idx] = TRUE if bypasses else FALSE
    idx += 1

    # self_boosts (5 stats)
    self_boost = getattr(move, "self_boost", None) or {}
    for key in GEN1_STAT_KEYS:
        val = self_boost.get(key, 0) if isinstance(self_boost, dict) else 0
        vec[idx] = (val or 0) / 6.0
        idx += 1

    # boosts on opponent (5 stats)
    boosts = getattr(move, "boosts", None) or {}
    for key in GEN1_STAT_KEYS:
        val = boosts.get(key, 0) if isinstance(boosts, dict) else 0
        vec[idx] = (val or 0) / 6.0
        idx += 1

    # inflicted status (6-dim one-hot, same convention as Pokémon status)
    inflicted = getattr(move, "status", None)
    status_vec = np.full(N_STATUS_ONEHOT, FALSE, dtype=np.float32)
    if inflicted is None:
        status_vec[N_STATUS_CONDITIONS] = TRUE  # no status inflicted
    else:
        sname = inflicted.name.upper() if hasattr(inflicted, "name") \
            else str(inflicted).upper()
        sidx = STATUS_TO_IDX.get(sname, N_STATUS_CONDITIONS)
        status_vec[sidx] = TRUE
    vec[idx: idx + N_STATUS_ONEHOT] = status_vec
    idx += N_STATUS_ONEHOT

    assert idx == MOVE_SIZE, f"Move encoding size mismatch: {idx} != {MOVE_SIZE}"
    return vec


def encode_moves(
    move_list: list[Move | None],
    opponent: Pokemon | None,
    unknown_empty_slots: bool = False,
    disabled_moves: set | None = None,
) -> np.ndarray:
    """
    Encodes exactly N_MOVE_SLOTS (4) moves into MOVES_SIZE (184) floats.

    unknown_empty_slots=True:  slots beyond known moves → UNKNOWN.
      Use for opponent Pokémon - unseen moves are genuinely unknown.
    unknown_empty_slots=False: slots beyond known moves → FALSE (empty).
      Use for our own Pokémon - empty slots are confirmed empty.

    disabled_moves: set of Move objects absent from pokemon.available_moves
      but present in pokemon.moves, indicating they are currently disabled.
    """
    vec = np.empty(MOVES_SIZE, dtype=np.float32)
    for slot in range(N_MOVE_SLOTS):
        start = slot * MOVE_SIZE
        end   = start + MOVE_SIZE
        if slot < len(move_list):
            m = move_list[slot]
            vec[start:end] = encode_move(
                m,
                opponent,
                known=True,
                is_disabled=(m is not None and disabled_moves is not None and m in disabled_moves),
            )
        else:
            if unknown_empty_slots:
                # TODO MovesetPrior v2: replace with species-conditioned
                # average move feature vector from Gen 1 learnset data
                # (server/data/learnsets.js)
                vec[start:end] = np.full(MOVE_SIZE, UNKNOWN, dtype=np.float32)
            else:
                vec[start:end] = encode_move(None, opponent, known=True)
    return vec


def encode_boosts(pokemon: Pokemon) -> np.ndarray:
    """6-float vector, each stage / 6 ∈ [-1, 1]."""
    boosts = getattr(pokemon, "boosts", {}) or {}
    return np.array(
        [boosts.get(k, 0) / 6.0 for k in GEN1_BOOST_KEYS],
        dtype=np.float32,
    )


def encode_volatile(pokemon: Pokemon) -> np.ndarray:
    """
    9-float binary vector of Gen 1 volatile conditions.
    Order defined by GEN1_VOLATILE_EFFECTS.

    Reads pokemon.effects (Dict[Effect, int]).
    Effect enum names are matched as uppercase strings for
    forward-compatibility with poke-env version differences.
    """
    effects = getattr(pokemon, "effects", {}) or {}
    effect_names = {
        (k.name.upper() if hasattr(k, "name") else str(k).upper())
        for k in effects
    }
    return np.array(
        [TRUE if name in effect_names else FALSE for name in GEN1_VOLATILE_EFFECTS],
        dtype=np.float32,
    )


def encode_base_pokemon(
    pokemon: Pokemon | None,
    opponent: Pokemon | None,
    available_moves: list[Move] | None = None,
    known: bool = True,
    is_opponent: bool = False,
) -> np.ndarray:
    """
    212-float base Pokémon encoding shared by active and bench slots.

    available_moves: pass battle.available_moves for our active Pokémon
                     (reflects current PP and locked-out state correctly).
                     Pass None for bench Pokémon - uses pokemon.moves.
    is_opponent:     True → unknown move slots encoded as UNKNOWN.
                     False → unknown move slots encoded as FALSE (empty).
    """
    if not known or pokemon is None:
        return np.full(BASE_POKEMON_SIZE, UNKNOWN, dtype=np.float32)

    vec = np.empty(BASE_POKEMON_SIZE, dtype=np.float32)
    idx = 0

    vec[idx: idx + N_TYPES] = encode_types(pokemon)
    idx += N_TYPES

    vec[idx: idx + N_STATS] = encode_stats(pokemon)
    idx += N_STATS

    vec[idx] = encode_hp_fraction(pokemon)
    idx += 1

    vec[idx: idx + N_STATUS_ONEHOT] = encode_status(pokemon)
    idx += N_STATUS_ONEHOT

    vec[idx] = encode_status_counter(pokemon)
    idx += 1

    move_list = list((pokemon.moves or {}).values())

    disabled_moves: set | None = None
    if not is_opponent and available_moves is not None:
        available_set = set(available_moves)
        disabled_moves = {m for m in move_list if m is not None and m not in available_set}

    vec[idx: idx + MOVES_SIZE] = encode_moves(
        move_list,
        opponent,
        unknown_empty_slots=is_opponent,
        disabled_moves=disabled_moves,
    )
    idx += MOVES_SIZE

    assert idx == BASE_POKEMON_SIZE, \
        f"Base Pokémon encoding mismatch: {idx} != {BASE_POKEMON_SIZE}"
    return vec


def encode_active_extra(
    pokemon: Pokemon | None,
    known: bool = True,
) -> np.ndarray:
    """
    34-float active-only features: boosts, volatile conditions,
    recharge / preparation state, turn tracking.
    """
    if not known or pokemon is None:
        return np.full(ACTIVE_EXTRA_SIZE, UNKNOWN, dtype=np.float32)

    vec = np.empty(ACTIVE_EXTRA_SIZE, dtype=np.float32)
    idx = 0

    # Boosts (6)
    vec[idx: idx + N_BOOSTS] = encode_boosts(pokemon)
    idx += N_BOOSTS

    # Volatile effects (9)
    vec[idx: idx + VOLATILE_SIZE] = encode_volatile(pokemon)
    idx += VOLATILE_SIZE

    # must_recharge
    vec[idx] = TRUE if (getattr(pokemon, "must_recharge", False) or False) else FALSE
    idx += 1

    # first_turn
    vec[idx] = TRUE if (getattr(pokemon, "first_turn", False) or False) else FALSE
    idx += 1

    # last_move_slot: normalized index 0–3, UNKNOWN if no last move
    last_move = getattr(pokemon, "last_move", None)
    if last_move is None:
        vec[idx] = UNKNOWN
    else:
        moves_list = list((pokemon.moves or {}).values())
        try:
            slot = moves_list.index(last_move)
            vec[idx] = norm(slot / 3.0)
        except ValueError:
            vec[idx] = UNKNOWN
    idx += 1

    # is_preparing + preparing_move_type (1 + 15)
    preparing = getattr(pokemon, "preparing_move", None)
    vec[idx] = TRUE if preparing else FALSE
    idx += 1

    type_vec = np.full(N_TYPES, FALSE, dtype=np.float32)
    if preparing and hasattr(preparing, "type") and preparing.type:
        tname = preparing.type.name.upper() \
            if hasattr(preparing.type, "name") else ""
        tidx = TYPE_TO_IDX.get(tname, -1)
        if tidx >= 0:
            type_vec[tidx] = TRUE
    vec[idx: idx + N_TYPES] = type_vec
    idx += N_TYPES

    assert idx == ACTIVE_EXTRA_SIZE, \
        f"Active extra encoding mismatch: {idx} != {ACTIVE_EXTRA_SIZE}"
    return vec


# ── Main assembler ───────────────────────────────────────────────────

def build_observation(battle: AbstractBattle) -> np.ndarray:
    """
    Assembles the full 2623-float observation vector from a live battle.

    Layout:
      [0        :  246]  Our active Pokémon
      [246      : 1311]  Our bench * 5  (213 each)
      [1311     : 1557]  Opponent active Pokémon
      [1557     : 2622]  Opponent bench * 5  (213 each)
      [2622     : 2623]  Turn
    """
    obs = np.empty(OBS_SIZE, dtype=np.float32)
    ptr = 0

    active     = battle.active_pokemon
    opp_active = battle.opponent_active_pokemon

    # ── Our active Pokémon (246) ─────────────────────────────────────
    obs[ptr: ptr + BASE_POKEMON_SIZE] = encode_base_pokemon(
        active,
        opp_active,
        available_moves=list(battle.available_moves),
        known=True,
        is_opponent=False,
    )
    ptr += BASE_POKEMON_SIZE

    obs[ptr: ptr + ACTIVE_EXTRA_SIZE] = encode_active_extra(active, known=True)
    ptr += ACTIVE_EXTRA_SIZE

    # ── Our bench × 5 (1065) ────────────────────────────────────────
    bench = [p for p in battle.team.values() if p != active]
    for slot in range(N_BENCH_SLOTS):
        if slot < len(bench):
            p = bench[slot]
            obs[ptr: ptr + BASE_POKEMON_SIZE] = encode_base_pokemon(
                p, opp_active, known=True, is_opponent=False,
            )
            obs[ptr + BASE_POKEMON_SIZE] = TRUE  # always revealed (ours)
        else:
            obs[ptr: ptr + BENCH_POKEMON_SIZE] = np.full(
                BENCH_POKEMON_SIZE, UNKNOWN, dtype=np.float32,
            )
        ptr += BENCH_POKEMON_SIZE

    # ── Opponent active Pokémon (246) ────────────────────────────────
    opp_known = opp_active is not None
    obs[ptr: ptr + BASE_POKEMON_SIZE] = encode_base_pokemon(
        opp_active,
        active,
        available_moves=None,
        known=opp_known,
        is_opponent=True,
    )
    ptr += BASE_POKEMON_SIZE

    obs[ptr: ptr + ACTIVE_EXTRA_SIZE] = encode_active_extra(
        opp_active, known=opp_known,
    )
    ptr += ACTIVE_EXTRA_SIZE

    # ── Opponent bench × 5 (1065) ────────────────────────────────────
    opp_bench = [p for p in battle.opponent_team.values() if p != opp_active]
    for slot in range(N_BENCH_SLOTS):
        if slot < len(opp_bench):
            p = opp_bench[slot]
            obs[ptr: ptr + BASE_POKEMON_SIZE] = encode_base_pokemon(
                p, active, known=True, is_opponent=True,
            )
            obs[ptr + BASE_POKEMON_SIZE] = TRUE   # revealed
        else:
            obs[ptr: ptr + BASE_POKEMON_SIZE] = np.full(
                BASE_POKEMON_SIZE, UNKNOWN, dtype=np.float32,
            )
            obs[ptr + BASE_POKEMON_SIZE] = FALSE  # not yet revealed
        ptr += BENCH_POKEMON_SIZE

    # ── Turn (1) ─────────────────────────────────────────────────────
    turn = getattr(battle, "turn", 0) or 0
    obs[ptr] = norm(min(turn / 100.0, 1.0))
    ptr += 1

    assert ptr == OBS_SIZE, \
        f"Observation vector size mismatch: got {ptr}, expected {OBS_SIZE}"
    return obs
