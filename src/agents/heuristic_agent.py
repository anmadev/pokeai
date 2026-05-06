import json
import re
from pathlib import Path
from typing import Optional

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player import Player

from src.engine.damage_calc import (
    best_move_by_damage,
    calculate_damage_range,
    projected_stats,
)


# -------------------------------------------------------------------
# Thresholds — named constants so they're easy to tune and document
# -------------------------------------------------------------------
LOW_HP_THRESHOLD = 0.25          # switch out if HP drops below this
SWITCH_ADVANTAGE_THRESHOLD = 2.0 # switch if best move effectiveness is 2x+
                                  # in favor of opponent's type


# -------------------------------------------------------------------
# Best-moves lookup data (loaded once at import time)
# -------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent.parent.parent / "data"

with open(_DATA_DIR / "best_moves.json") as _f:
    _BEST_MOVES: dict = json.load(_f)

# Map poke-env species IDs (lowercase, no punctuation) → JSON key
_SPECIES_TO_JSON_KEY: dict[str, str] = {
    re.sub(r"[^a-z0-9]", "", k.lower()): k for k in _BEST_MOVES
}

# Type ordering as used when generating best_moves.json
_TYPE_ORDER: dict[str, int] = {
    t: i for i, t in enumerate([
        "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
        "Fighting", "Poison", "Ground", "Flying", "Psychic",
        "Bug", "Rock", "Ghost", "Dragon",
    ])
}


# -------------------------------------------------------------------
# Potential-threat helpers
# -------------------------------------------------------------------

def _defender_type_key(pokemon: Pokemon) -> str:
    """
    Builds the type-combo key used in best_moves.json for a given defender.
    Types are ordered by their position in the Gen 1 type list.
    """
    types = [t.name.capitalize() for t in pokemon.types if t is not None]
    types.sort(key=lambda t: _TYPE_ORDER.get(t, 99))
    return "/".join(types)


def _potential_threat_damage(attacker: Pokemon, defender: Pokemon) -> float:
    """
    Estimates the expected damage the attacker could deal to the defender
    using the best physical and special moves available in its learnset.

    This provides a threat floor even before the opponent has revealed any
    moves, preventing the heuristic from being blind to dangerous matchups
    on turn 1.

    Returns the higher of the two (physical / special) expected damage values.
    """
    json_key = _SPECIES_TO_JSON_KEY.get(attacker.species)
    if json_key is None:
        return 0.0

    type_key = _defender_type_key(defender)
    matchup = _BEST_MOVES.get(json_key, {}).get(type_key)
    if matchup is None:
        return 0.0

    worst = 0.0

    for category in ("physical", "special"):
        move_name = matchup.get(category, {}).get("best_max", {}).get("move")
        if not move_name:
            continue
        move_id = re.sub(r"[^a-z0-9]", "", move_name.lower())
        try:
            move = Move(move_id, gen=1)
        except Exception:
            continue
        result = calculate_damage_range(move, attacker, defender)
        if result is None:
            continue
#        expected = (result.min_damage + result.max_damage) / 2
        expected = result.max_damage # be pessimistic about potential threats — assume max damage roll and hits 
        worst = max(worst, expected)

    return worst


# -------------------------------------------------------------------
# Scoring helpers
# -------------------------------------------------------------------

def offensive_score(candidate: Pokemon, opponent: Pokemon) -> float:
    """
    Estimates how effectively a Pokémon can hit the opponent.

    Uses the candidate's known moves if available (our own team's moves
    are always known in a random battle). Falls back to 1.0 (neutral)
    if no damaging moves are found — this shouldn't happen in practice
    but makes the function safe.
    """
    damaging_moves = [
        m for m in candidate.moves.values() if m.base_power > 0
    ]
    if not damaging_moves:
        return 1.0

    result = best_move_by_damage(candidate, opponent, damaging_moves)
    if result is None:
        return 1.0
    _, damage_range = result
    return (damage_range.min_damage + damage_range.max_damage) / 2


def defensive_score(candidate: Pokemon, opponent: Pokemon) -> float:
    """
    Estimates how safely a Pokémon can switch into the opponent.

    Considers two sources of threat:
    1. Revealed moves: moves the opponent has already used this battle.
    2. Potential best moves: the highest-damage physical and special move
       the opponent *could* have, based on its species learnset and the
       candidate's type. This provides a non-zero threat floor on turn 1
       before any opponent moves are known.

    Returns the inverse of the worst expected damage across both sources.
    """
    worst_expected = 0.0

    # Source 1: known revealed opponent moves
    for m in opponent.moves.values():
        if m.base_power == 0:
            continue
        result = calculate_damage_range(m, opponent, candidate)
        if result is None:
            continue
        expected = (result.min_damage + result.max_damage) / 2
        if expected > worst_expected:
            worst_expected = expected

    # Source 2: best potential moves from the opponent's learnset
    potential = _potential_threat_damage(opponent, candidate)
    worst_expected = max(worst_expected, potential)

    return 1.0 / worst_expected if worst_expected > 0 else 1.0


def score_switch_candidate(
    candidate: Pokemon,
    opponent: Pokemon,
) -> float:
    """
    Combined score for a potential switch target.

    Components:
    - Offensive matchup against the opponent
    - Defensive resistance to known opponent moves
    - HP ratio: prefer healthy Pokémon over nearly-fainted ones

    All three are multiplied together so a Pokémon that is strong
    offensively but nearly fainted still ranks lower than a healthy
    neutral option.
    """
    return (
        offensive_score(candidate, opponent)
        * defensive_score(candidate, opponent)
        * candidate.current_hp_fraction
    )


# -------------------------------------------------------------------
# Decision functions
# -------------------------------------------------------------------

def _active_can_outspeed_and_ko(battle) -> bool:
    """
    Returns True if all three conditions hold:
    - The active Pokémon is faster than the opponent (by base-stat estimate).
    - The opponent threatens an OHKO on the active Pokémon (potential max
      damage ≥ current HP), meaning staying in is otherwise fatal.
    - The active Pokémon has a move that could OHKO the opponent (max
      damage ≥ opponent's estimated remaining HP).

    When all three hold, attacking is strictly better than switching: we
    move first, remove the threat, and deny the opponent a free turn.
    """
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon

    # --- Speed: ours (known battle stat) vs opponent (base-stat estimate) ---
    opp_projected_stats = projected_stats(opponent)
    our_speed = active.stats.get("spe")
    opp_speed = opp_projected_stats.get("spe")

    if our_speed <= opp_speed:
        return False  # opponent moves first — we can't guarantee the KO

    # --- Does the opponent threaten to OHKO us? ---
    opp_threat = _potential_threat_damage(opponent, active)
    if opp_threat < active.current_hp:
        return False  # not a lethal threat — veto doesn't apply

    # --- Can we OHKO the opponent with our best available move? ---
    damaging = [m for m in battle.available_moves if m.base_power > 0]
    if not damaging:
        return False

    result = best_move_by_damage(active, opponent, damaging)
    if result is None:
        return False

    _, damage_range = result
    opp_max_hp = projected_stats(opponent).get("hp")
    opp_remaining_hp = opp_max_hp * opponent.current_hp_fraction

    return damage_range.max_damage >= opp_remaining_hp


def should_switch(battle) -> bool:
    """
    Returns True if switching is worth considering.

    Two conditions trigger a switch evaluation:
    1. Low HP: active Pokémon is below the survival threshold
    2. Type trap: our best move against the opponent is resisted (< 1x),
       meaning we are likely in a losing offensive matchup

    Veto: even when a switch looks warranted, stay in if the active
    Pokémon outspeeds the opponent, the opponent threatens an OHKO, and
    we can OHKO first — switching gives the opponent a free turn.

    We only recommend switching if there is actually a healthy switch
    option available — no point flagging a switch if all bench Pokémon
    are fainted or equally threatened.
    """
    active = battle.active_pokemon
    opponent = battle.opponent_active_pokemon

    # Must have at least one switch option worth using
    healthy_switches = [
        p for p in battle.available_switches
        if p.current_hp_fraction > LOW_HP_THRESHOLD
    ]
    if not healthy_switches:
        return False

    wants_switch = False

    # Condition 1: low HP
    if active.current_hp_fraction < LOW_HP_THRESHOLD:
        wants_switch = True

    # Condition 2: all our damaging moves are resisted
    if not wants_switch:
        damaging_moves = [m for m in battle.available_moves if m.base_power > 0]
        if damaging_moves:
            best_effectiveness = max(
                opponent.damage_multiplier(m) for m in damaging_moves
            )
            if best_effectiveness < 1.0:
                wants_switch = True

    if not wants_switch:
        return False

    # Veto: we outspeed and can KO first — attacking is better than switching
    if _active_can_outspeed_and_ko(battle):
        return False

    return True


def best_switch(battle) -> Optional[Pokemon]:
    """
    Picks the best switch candidate from the available bench.

    Excludes Pokémon below the low HP threshold — switching to a
    near-fainted Pokémon is almost always wrong. If all available
    switches are below the threshold (unlikely but possible), falls
    back to the highest-scored regardless.
    """
    opponent = battle.opponent_active_pokemon
    candidates = battle.available_switches

    healthy = [p for p in candidates if p.current_hp_fraction > LOW_HP_THRESHOLD]
    pool = healthy if healthy else candidates

    if not pool:
        return None

    return max(pool, key=lambda p: score_switch_candidate(p, opponent))


def best_move(battle):
    """
    Picks the best damaging move, falling back to any move if needed.

    Uses the full Gen 1 damage formula to rank moves. Avoids pure
    status moves entirely when a damaging option exists — using Growl
    when you could use Thunderbolt is strictly wrong here.
    """
    damaging = [m for m in battle.available_moves if m.base_power > 0]
    opponent = battle.opponent_active_pokemon

    if damaging:
        result = best_move_by_damage(battle.active_pokemon, opponent, damaging)
        if result is not None:
            move, _ = result
            return move

    # No damaging moves — all status. Pick any (nothing better we can do)
    if battle.available_moves:
        return battle.available_moves[0]

    return None


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

class HeuristicPlayer(Player):
    """
    A rule-based Pokémon player that makes decisions using game knowledge.

    Decision priority:
    1. If switching is warranted (low HP or bad matchup) → switch to
       the best available candidate
    2. Otherwise → use the best available damaging move

    This agent has no learning component — it encodes human intuition
    about Gen 1 mechanics directly as rules. Its win rate against
    MaxDamagePlayer establishes the ceiling that the RL agent must
    surpass to be considered meaningful.
    """

    def choose_move(self, battle):
        if battle.available_switches and should_switch(battle):
            target = best_switch(battle)
            if target:
                return self.create_order(target)

        move = best_move(battle)
        if move:
            return self.create_order(move)

        # True fallback — should be unreachable in normal play
        return self.choose_random_move(battle)
