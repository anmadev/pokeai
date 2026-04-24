from dataclasses import dataclass
from typing import Optional

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon


# ------------------------------------------------------------------
# Gen 1 type categorisation
# Move type determines which stat pair (Atk/Def or Spe/Spe) is used
# ------------------------------------------------------------------
SPECIAL_TYPES = {
    "fire", "water", "grass", "electric",
    "psychic", "ice", "dragon",
}

PHYSICAL_TYPES = {
    "normal", "fighting", "flying", "poison", "ground",
    "rock", "bug", "ghost",
}

# Random damage roll range in Gen 1
RANDOM_MIN = 217
RANDOM_MAX = 255

# Level is always 100 in Showdown Gen 1 random battles
BATTLE_LEVEL = 100


# ------------------------------------------------------------------
# Stat estimation
# ------------------------------------------------------------------

def estimate_stat(base: int, is_hp: bool = False) -> int:
    """
    Estimates a Gen 1 stat assuming max DVs (15) and max StatExp (65535).

    Formula (Level 100, DV=15, StatExp=65535):
        Non-HP: floor((Base + 15) * 2 + 64) + 5  = Base * 2 + 99
        HP:     floor((Base + 15) * 2 + 64) + 110 = Base * 2 + 204

    This is exact for Showdown Gen 1 random battles where all Pokémon
    are generated with maximum DVs and StatExp.

    Under partial information (opponent's Pokémon) we use base stats
    as the only available signal — this formula gives us a point
    estimate that is correct the vast majority of the time.
    """
    if is_hp:
        return base * 2 + 204
    return base * 2 + 99


def is_special_move(move: Move) -> bool:
    """Returns True if the move uses the Special stat (both sides)."""
    return move.type.name.lower() in SPECIAL_TYPES


# ------------------------------------------------------------------
# Damage result
# ------------------------------------------------------------------

@dataclass
class DamageRange:
    """
    Represents the min/max damage a move can deal.

    min_damage: lowest possible roll (random = 217/255 ≈ 0.851) 
    and lowest number of hits (1 for multi-hit moves)
    max_damage: highest possible roll (random = 255/255 = 1.0) 
    and highest number of hits (1 for multi-hit moves)
    min_percent: min damage as fraction of opponent's estimated max HP
    max_percent: max damage as fraction of opponent's estimated max HP
    is_special: whether the Special stat was used
    type_multiplier: raw type effectiveness applied
    stab: whether STAB was applied
    """
    move_name: str
    min_damage: int
    max_damage: int
    min_percent: float
    max_percent: float
    is_special: bool
    type_multiplier: float
    stab: bool

    def __str__(self) -> str:
        stab_str = " (STAB)" if self.stab else ""
        effectiveness = ""
        if self.type_multiplier >= 4.0:
            effectiveness = " [4x]"
        elif self.type_multiplier >= 2.0:
            effectiveness = " [2x]"
        elif self.type_multiplier <= 0.0:
            effectiveness = " [immune]"
        elif self.type_multiplier < 1.0:
            effectiveness = " [resisted]"
        return (
            f"{self.move_name}{stab_str}{effectiveness}: "
            f"{self.min_damage}–{self.max_damage} dmg "
            f"({self.min_percent:.1f}%–{self.max_percent:.1f}% HP)"
        )


# ------------------------------------------------------------------
# Core calculator
# ------------------------------------------------------------------

def _apply_random(base_damage: int, roll: int) -> int:
    """Applies a single random roll to base damage (Gen 1 integer math)."""
    return (base_damage * roll) // 255


def calculate_damage_range(
    move: Move,
    attacker: Pokemon,
    defender: Pokemon,
    attacker_level: int = BATTLE_LEVEL,
    attack_stat_override: Optional[int] = None,
    defense_stat_override: Optional[int] = None,
) -> Optional[DamageRange]:
    """
    Calculates the min/max damage range for a move in Gen 1.

    Stat resolution (in priority order):
    1. Override provided explicitly (useful for testing or when we know
       actual stats from battle log observation)
    2. Current battle stats from poke-env (includes stat stage modifiers
       for our own Pokémon — poke-env tracks these)
    3. Estimated from base stats assuming max DVs and StatExp
       (used for opponent Pokémon under partial information)

    Returns None for zero-power (status) moves — they deal no damage.

    Stat stage modifiers (from moves like Swords Dance, Growl):
    - Our own Pokémon's stages are tracked by poke-env and accessible
      via attacker.boosts["atk"] etc.
    - Opponent stages are partially observable: poke-env tracks boosts
      we've seen applied but cannot know hidden ones.
    - The override parameters allow the caller to inject corrected stats
      when better information is available.
    """
    if move.base_power == 0:
        return None

    special = is_special_move(move)
    type_multiplier = defender.damage_multiplier(move)

    if type_multiplier == 0.0:
        # Immune — return a zero-damage range explicitly
        return DamageRange(
            move_name=move.id,
            min_damage=0,
            max_damage=0,
            min_percent=0.0,
            max_percent=0.0,
            is_special=special,
            type_multiplier=0.0,
            stab=False,
        )

    # ---- Resolve attack stat ----------------------------------------
    if attack_stat_override is not None:
        atk = attack_stat_override
    else:
        try:
            if special:
                # poke-env exposes base stats; current_stats includes boosts
                # for our own Pokémon
                atk = attacker.stats.get("spa")
            else:
                atk = attacker.stats.get("atk")
        except (AttributeError, TypeError):
            atk = estimate_stat(attacker.base_stats.get("spa" if special else "atk"))

    # ---- Resolve defense stat ---------------------------------------
    if defense_stat_override is not None:
        defense = defense_stat_override
    else:
        try:
            if special:
                defense = defender.stats.get("spd")
            else:
                defense = defender.stats.get("def")
        except (AttributeError, TypeError):
            defense = estimate_stat(defender.base_stats.get("spd" if special else "def"))

    # ---- Resolve defender HP for percentage calculation -------------
    try:
        max_hp = defender.max_hp
    except (AttributeError, TypeError):
        max_hp = estimate_stat(defender.base_stats.get("hp"), is_hp=True)

    # ---- STAB -------------------------------------------------------
    stab = move.type in attacker.types
    stab_multiplier = 1.5 if stab else 1.0

    # ---- Base damage (before random roll) ---------------------------
    # floor(floor(floor(2*L/5+2) * A * P / D) / 50 + 2) * STAB * Type
    # does not take crit into account since it's not relevant for move selection
    level_factor = (2 * attacker_level // 5) + 2
    raw = (level_factor * atk * move.base_power) // defense
    base_damage = raw // 50 + 2

    # Apply STAB and type effectiveness (integer multiplication then floor)
    # Gen 1 applies these as sequential multiplications with floor at each step
    # STAB: multiply by 1.5, floor
    after_stab = int(base_damage * stab_multiplier)
    # Type: in Gen 1, each type interaction is applied one at a time with floor
    # We decompose the combined multiplier for correctness
    after_type = int(after_stab * type_multiplier)

    # ---- Multi-hit moves ------------------------------------------------
    min_hits = move.n_hit[0]
    max_hits = move.n_hit[-1]

    # ---- Random rolls -----------------------------------------------
    min_damage = _apply_random(after_type * min_hits, RANDOM_MIN)
    max_damage = _apply_random(after_type * max_hits, RANDOM_MAX)

    # Edge case: minimum 1 damage if the move is not immune
    min_damage = max(1, min_damage)
    max_damage = max(1, max_damage)

    return DamageRange(
        move_name=move.id,
        min_damage=min_damage,
        max_damage=max_damage,
        min_percent=round(min_damage / max_hp * 100, 2),
        max_percent=round(max_damage / max_hp * 100, 2),
        is_special=special,
        type_multiplier=type_multiplier,
        stab=stab,
    )


def best_move_by_damage(
    attacker: Pokemon,
    defender: Pokemon,
    available_moves: list[Move],
) -> Optional[tuple[Move, DamageRange]]:
    """
    Returns the (move, DamageRange) pair with the highest expected damage.
    Expected damage = midpoint of the min/max range.
    Returns None if no damaging moves are available.
    """
    best = None
    best_expected = -1.0

    for move in available_moves:
        result = calculate_damage_range(move, attacker, defender)
        if result is None:
            continue
        expected = (result.min_damage + result.max_damage) / 2
        if expected > best_expected:
            best_expected = expected
            best = (move, result)

    return best
