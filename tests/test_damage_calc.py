"""
Unit and regression tests for engine/damage_calc.py.

Tests are structured in three groups:
  1. is_special_move — type classification
  2. calculate_damage_range — core formula, edge cases, overrides
  3. best_move_by_damage — selection logic

All tests use Mock objects — no server connection required.
"""

from __future__ import annotations

import pytest

from engine.damage_calc import (
    RANDOM_MAX,
    RANDOM_MIN,
    DamageRange,
    _apply_random,
    best_move_by_damage,
    calculate_damage_range,
    is_special_move,
)
from tests.conftest import make_move, make_pokemon


# ── _apply_random ─────────────────────────────────────────────────────

class TestApplyRandom:
    def test_max_roll_returns_full_damage(self):
        assert _apply_random(100, 255) == 100

    def test_min_roll_reduces_damage(self):
        result = _apply_random(100, 217)
        assert result == (100 * 217) // 255
        assert result == 85

    def test_zero_base_damage_returns_zero(self):
        assert _apply_random(0, 255) == 0
        assert _apply_random(0, 217) == 0

    def test_integer_truncation(self):
        # (100 * 240) // 255 = 24000 // 255 = 94 (not 94.1...)
        assert _apply_random(100, 240) == 94


# ── is_special_move ───────────────────────────────────────────────────

class TestIsSpecialMove:
    @pytest.mark.parametrize("type_name", [
        "fire", "water", "grass", "electric", "psychic", "ice", "dragon"
    ])
    def test_special_types(self, type_name):
        move = make_move(type_name=type_name.upper())
        assert is_special_move(move) is True

    @pytest.mark.parametrize("type_name", [
        "normal", "fighting", "flying", "poison", "ground",
        "rock", "bug", "ghost"
    ])
    def test_physical_types(self, type_name):
        move = make_move(type_name=type_name.upper())
        assert is_special_move(move) is False


# ── calculate_damage_range ────────────────────────────────────────────

class TestCalculateDamageRange:

    def test_status_move_returns_none(self):
        """Zero base power moves (status moves) return None."""
        move = make_move("thunderwave", base_power=0, type_name="ELECTRIC")
        attacker = make_pokemon("pikachu", types=["ELECTRIC"])
        defender = make_pokemon("rattata", types=["NORMAL"])
        assert calculate_damage_range(move, attacker, defender) is None

    def test_immune_move_returns_zero_damage(self):
        """Type immunity returns DamageRange with 0 damage."""
        move = make_move("tackle", base_power=40, type_name="NORMAL")
        attacker = make_pokemon("rattata", types=["NORMAL"])
        defender = make_pokemon("gengar", types=["GHOST", "POISON"])
        defender.damage_multiplier.return_value = 0.0

        result = calculate_damage_range(move, attacker, defender)

        assert result is not None
        assert result.min_damage == 0
        assert result.max_damage == 0
        assert result.type_multiplier == 0.0

    def test_basic_damage_range_min_less_than_max(self):
        """Min damage must always be ≤ max damage."""
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")
        attacker = make_pokemon("pikachu", types=["ELECTRIC"],
                                stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})
        defender = make_pokemon("starmie", types=["WATER", "PSYCHIC"],
                                stats={"hp": 293, "atk": 230, "def": 230, "spa": 278, "spe": 295},
                                max_hp=293)
        defender.damage_multiplier.return_value = 1.0

        result = calculate_damage_range(move, attacker, defender)

        assert result is not None
        assert result.min_damage <= result.max_damage

    def test_stab_increases_damage(self):
        """STAB (1.5×) must produce higher damage than non-STAB."""
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")

        attacker_stab = make_pokemon("pikachu", types=["ELECTRIC"],
                                    stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})
        attacker_no_stab = make_pokemon("rattata", types=["NORMAL"],
                                        stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})

        defender = make_pokemon("starmie", types=["WATER", "PSYCHIC"],
                                stats={"hp": 293, "atk": 230, "def": 230, "spa": 278, "spe": 295},
                                max_hp=293)
        defender.damage_multiplier.return_value = 1.0

        stab_result = calculate_damage_range(move, attacker_stab, defender)
        no_stab_result = calculate_damage_range(move, attacker_no_stab, defender)

        assert stab_result is not None
        assert no_stab_result is not None
        assert stab_result.max_damage > no_stab_result.max_damage
        assert stab_result.stab is True
        assert no_stab_result.stab is False

    def test_type_effectiveness_2x(self):
        """2× effective move deals more damage than neutral."""
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")
        attacker = make_pokemon("pikachu", types=["ELECTRIC"],
                                stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})
        defender = make_pokemon("gyarados", types=["WATER", "FLYING"],
                                stats={"hp": 323, "atk": 278, "def": 213, "spa": 195, "spe": 230},
                                max_hp=323)

        neutral_result = calculate_damage_range(move, attacker, defender)
        defender.damage_multiplier.return_value = 2.0
        effective_result = calculate_damage_range(move, attacker, defender)

        assert effective_result.max_damage > neutral_result.max_damage
        assert effective_result.type_multiplier == 2.0

    def test_attack_override_is_used(self):
        """Explicit attack override replaces computed stat."""
        move = make_move("tackle", base_power=40, type_name="NORMAL", category="PHYSICAL")
        attacker = make_pokemon("rattata", types=["NORMAL"],
                                stats={"hp": 251, "atk": 230, "def": 158, "spa": 158, "spe": 230})
        defender = make_pokemon("slowpoke", types=["WATER", "PSYCHIC"],
                                stats={"hp": 383, "atk": 158, "def": 195, "spa": 195, "spe": 113},
                                max_hp=383)
        defender.damage_multiplier.return_value = 1.0

        normal_result = calculate_damage_range(move, attacker, defender)
        boosted_result = calculate_damage_range(move, attacker, defender,
                                                attack_stat_override=460)  # doubled

        assert boosted_result.max_damage > normal_result.max_damage

    def test_min_damage_at_least_1(self):
        """Damage is never 0 for a non-immune, non-zero power move."""
        move = make_move("scratch", base_power=40, type_name="NORMAL", category="PHYSICAL")
        attacker = make_pokemon(stats={"hp": 100, "atk": 5, "def": 5, "spa": 5, "spe": 5})
        defender = make_pokemon(stats={"hp": 500, "atk": 5, "def": 999, "spa": 5, "spe": 5},
                                max_hp=500)
        defender.damage_multiplier.return_value = 1.0

        result = calculate_damage_range(move, attacker, defender)

        assert result is not None
        assert result.min_damage >= 1
        assert result.max_damage >= 1

    def test_percent_calculation_consistent(self):
        """min_percent = min_damage / max_hp * 100, rounded to 2 dp."""
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")
        attacker = make_pokemon("pikachu", types=["ELECTRIC"],
                                stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})
        defender = make_pokemon("starmie", types=["WATER", "PSYCHIC"],
                                stats={"hp": 293, "atk": 230, "def": 230, "spa": 278, "spe": 295},
                                max_hp=293)
        defender.damage_multiplier.return_value = 1.0

        result = calculate_damage_range(move, attacker, defender)

        assert result is not None
        assert abs(result.min_percent - (result.min_damage / 293 * 100)) < 0.01
        assert abs(result.max_percent - (result.max_damage / 293 * 100)) < 0.01

    def test_damage_range_str_representation(self):
        """DamageRange.__str__ includes move name and damage numbers."""
        dr = DamageRange(
            move_name="thunderbolt",
            min_damage=80,
            max_damage=95,
            min_percent=27.3,
            max_percent=32.4,
            is_special=True,
            type_multiplier=1.0,
            stab=True,
        )
        s = str(dr)
        assert "thunderbolt" in s
        assert "80" in s
        assert "95" in s
        assert "STAB" in s

    def test_immune_str_representation(self):
        dr = DamageRange("tackle", 0, 0, 0.0, 0.0, False, 0.0, False)
        assert "immune" in str(dr).lower()


# ── best_move_by_damage ───────────────────────────────────────────────

class TestBestMoveByDamage:

    def test_selects_highest_expected_damage(self):
        """Among multiple moves, the one with highest midpoint is selected."""
        weak_move  = make_move("scratch",     base_power=40,  type_name="NORMAL",   category="PHYSICAL")
        strong_move = make_move("hyperbeam",  base_power=150, type_name="NORMAL",   category="PHYSICAL")
        attacker = make_pokemon("snorlax", types=["NORMAL"],
                                stats={"hp": 523, "atk": 278, "def": 230, "spa": 195, "spe": 113})
        defender = make_pokemon("slowpoke", types=["WATER", "PSYCHIC"],
                                stats={"hp": 383, "atk": 158, "def": 195, "spa": 195, "spe": 113},
                                max_hp=383)
        defender.damage_multiplier.return_value = 1.0

        result = best_move_by_damage(attacker, defender, [weak_move, strong_move])

        assert result is not None
        best_move, _ = result
        assert best_move is strong_move

    def test_returns_none_for_all_status_moves(self):
        """If all moves have base_power 0, returns None."""
        status_moves = [
            make_move("thunderwave", base_power=0),
            make_move("growl",       base_power=0),
        ]
        attacker = make_pokemon()
        defender = make_pokemon()
        defender.damage_multiplier.return_value = 1.0

        assert best_move_by_damage(attacker, defender, status_moves) is None

    def test_returns_none_for_empty_move_list(self):
        attacker = make_pokemon()
        defender = make_pokemon()
        assert best_move_by_damage(attacker, defender, []) is None

    def test_skips_immune_matchups_for_selection(self):
        """
        If the best move by raw power is immune, the next best is chosen.
        Normal is immune to Ghost-type Pokémon via Ghost immunity, but here
        we test via damage_multiplier=0 mock.
        """
        immune_move  = make_move("hyperbeam",    base_power=150, type_name="NORMAL",   category="PHYSICAL")
        fallback_move = make_move("earthquake",   base_power=100, type_name="GROUND",   category="PHYSICAL")

        attacker = make_pokemon("snorlax", types=["NORMAL"],
                                stats={"hp": 523, "atk": 278, "def": 230, "spa": 195, "spe": 113})
        defender = make_pokemon("gengar", types=["GHOST", "POISON"],
                                stats={"hp": 261, "atk": 230, "def": 195, "spa": 278, "spe": 278},
                                max_hp=261)

        def multiplier(move):
            if move.type.name == "NORMAL":
                return 0.0  # Ghost immune to Normal
            return 1.0

        defender.damage_multiplier.side_effect = multiplier

        result = best_move_by_damage(attacker, defender, [immune_move, fallback_move])

        assert result is not None
        best_move, _ = result
        assert best_move is fallback_move

    def test_returns_damage_range_object(self):
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")
        attacker = make_pokemon("pikachu", types=["ELECTRIC"],
                                stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278})
        defender = make_pokemon("starmie", types=["WATER", "PSYCHIC"],
                                stats={"hp": 293, "atk": 230, "def": 230, "spa": 278, "spe": 295},
                                max_hp=293)
        defender.damage_multiplier.return_value = 1.0

        result = best_move_by_damage(attacker, defender, [move])

        assert result is not None
        _, damage_range = result
        assert isinstance(damage_range, DamageRange)
