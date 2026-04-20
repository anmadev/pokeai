import os
from typing import Optional

from poke_env import ServerConfiguration, LocalhostServerConfiguration
from poke_env.battle.pokemon import Pokemon
from poke_env.player import Player


# -------------------------------------------------------------------
# Thresholds — named constants so they're easy to tune and document
# -------------------------------------------------------------------
LOW_HP_THRESHOLD = 0.25          # switch out if HP drops below this
SWITCH_ADVANTAGE_THRESHOLD = 2.0 # switch if best move effectiveness is 2x+
                                  # in favor of opponent's type


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

    return max(
        m.base_power * opponent.damage_multiplier(m)
        for m in damaging_moves
    )


def defensive_score(candidate: Pokemon, opponent: Pokemon) -> float:
    """
    Estimates how safely a Pokémon can switch into the opponent.

    Checks the opponent's known (revealed) moves. If we haven't seen
    any opponent moves yet, returns 1.0 (neutral) — we have no
    information to penalize on.

    Returns the inverse of the worst hit the opponent can land:
    a candidate that takes 0.5x damage scores 2.0, one that takes
    2x damage scores 0.5.
    """
    known_opponent_moves = [
        m for m in opponent.moves.values() if m.base_power > 0
    ]
    if not known_opponent_moves:
        return 1.0

    worst_multiplier = max(
        candidate.damage_multiplier(m) for m in known_opponent_moves
    )
    return 1.0 / worst_multiplier if worst_multiplier > 0 else 1.0


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

def should_switch(battle) -> bool:
    """
    Returns True if switching is worth considering.

    Two conditions trigger a switch evaluation:
    1. Low HP: active Pokémon is below the survival threshold
    2. Type trap: our best move against the opponent is resisted (< 1x),
       meaning we are likely in a losing offensive matchup

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

    # Condition 1: low HP
    if active.current_hp_fraction < LOW_HP_THRESHOLD:
        return True

    # Condition 2: all our damaging moves are resisted
    damaging_moves = [m for m in battle.available_moves if m.base_power > 0]
    if damaging_moves:
        best_effectiveness = max(
            opponent.damage_multiplier(m) for m in damaging_moves
        )
        if best_effectiveness < 1.0:
            return True

    return False


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

    Prefers damaging moves weighted by type effectiveness.
    Avoids pure status moves entirely when a damaging option exists —
    using Growl when you could use Thunderbolt is strictly wrong here.
    """
    damaging = [m for m in battle.available_moves if m.base_power > 0]
    opponent = battle.opponent_active_pokemon

    if damaging:
        return max(
            damaging,
            key=lambda m: m.base_power * opponent.damage_multiplier(m),
        )

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
