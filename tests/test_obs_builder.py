"""
Unit and property tests for environment/obs_builder.py.

Focus areas:
  1. Encoding conventions — TRUE/FALSE/UNKNOWN values are correct
  2. Vector sizes — every function returns exactly the declared size
  3. Value bounds — all values in [-1, 1]
  4. OBS_SIZE constant — matches the assembled vector
  5. Known/unknown flag semantics — unknown → all zeros
  6. Edge cases — None inputs, empty moves, missing attributes
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.obs_builder import (
    ACTIVE_EXTRA_SIZE,
    ACTIVE_POKEMON_SIZE,
    BASE_POKEMON_SIZE,
    BENCH_POKEMON_SIZE,
    FALSE,
    GEN1_TYPES,
    MOVE_SIZE,
    MOVES_SIZE,
    N_BENCH_SLOTS,
    N_MOVE_SLOTS,
    N_STATUS_CONDITIONS,
    N_TYPES,
    OBS_SIZE,
    TRUE,
    UNKNOWN,
    VOLATILE_SIZE,
    encode_active_extra,
    encode_base_pokemon,
    encode_boosts,
    encode_hp_fraction,
    encode_move,
    encode_moves,
    encode_status,
    encode_types,
    encode_volatile,
    norm,
    norm_range,
)
from tests.conftest import make_battle, make_move, make_pokemon


# ── norm / norm_range ─────────────────────────────────────────────────

class TestNorm:
    def test_zero_maps_to_minus_one(self):
        assert norm(0.0) == pytest.approx(-1.0)

    def test_one_maps_to_plus_one(self):
        assert norm(1.0) == pytest.approx(1.0)

    def test_half_maps_to_zero(self):
        assert norm(0.5) == pytest.approx(0.0)

    def test_clamps_above_one(self):
        assert norm(2.0) == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert norm(-1.0) == pytest.approx(-1.0)

    def test_norm_range_equal_bounds_returns_unknown(self):
        assert norm_range(5.0, 5.0, 5.0) == UNKNOWN

    def test_norm_range_maps_correctly(self):
        assert norm_range(0.0, 0.0, 10.0) == pytest.approx(-1.0)
        assert norm_range(10.0, 0.0, 10.0) == pytest.approx(1.0)
        assert norm_range(5.0, 0.0, 10.0) == pytest.approx(0.0)


# ── encode_types ──────────────────────────────────────────────────────

class TestEncodeTypes:
    def test_size(self):
        mon = make_pokemon(types=["FIRE"])
        assert encode_types(mon).shape == (N_TYPES,)

    def test_single_type(self):
        mon = make_pokemon(types=["FIRE"])
        vec = encode_types(mon)
        fire_idx = GEN1_TYPES.index("FIRE")
        assert vec[fire_idx] == TRUE
        assert np.sum(vec == TRUE) == 1

    def test_dual_type(self):
        mon = make_pokemon(types=["WATER", "PSYCHIC"])
        vec = encode_types(mon)
        assert vec[GEN1_TYPES.index("WATER")] == TRUE
        assert vec[GEN1_TYPES.index("PSYCHIC")] == TRUE
        assert np.sum(vec == TRUE) == 2

    def test_unknown_returns_all_zeros(self):
        mon = make_pokemon(types=["FIRE"])
        vec = encode_types(mon, known=False)
        assert np.all(vec == UNKNOWN)

    def test_none_pokemon_returns_all_zeros(self):
        vec = encode_types(None)
        assert np.all(vec == UNKNOWN)

    def test_non_gen1_type_ignored(self):
        """Types not in Gen 1 (e.g. DARK, STEEL) are ignored without error."""
        mon = make_pokemon(types=["DARK"])  # not in GEN1_TYPES
        vec = encode_types(mon)
        assert np.all(vec == FALSE)

    def test_all_values_are_true_or_false(self):
        mon = make_pokemon(types=["ELECTRIC"])
        vec = encode_types(mon)
        assert all(v in (TRUE, FALSE) for v in vec)


# ── encode_status ─────────────────────────────────────────────────────

class TestEncodeStatus:
    def test_size(self):
        mon = make_pokemon()
        assert encode_status(mon).shape == (N_STATUS_CONDITIONS + 1,)

    def test_healthy_sets_index_5(self):
        mon = make_pokemon(status=None)
        vec = encode_status(mon)
        assert vec[N_STATUS_CONDITIONS] == TRUE
        assert np.sum(vec == TRUE) == 1

    @pytest.mark.parametrize("status_name,expected_idx", [
        ("BRN", 0), ("FRZ", 1), ("PAR", 2), ("PSN", 3), ("SLP", 4),
    ])
    def test_each_status_sets_correct_index(self, status_name, expected_idx):
        from unittest.mock import MagicMock
        status_mock = MagicMock()
        status_mock.name = status_name
        mon = make_pokemon(status=status_mock)
        vec = encode_status(mon)
        assert vec[expected_idx] == TRUE
        assert np.sum(vec == TRUE) == 1

    def test_unknown_returns_all_zeros(self):
        mon = make_pokemon()
        vec = encode_status(mon, known=False)
        assert np.all(vec == UNKNOWN)

    def test_healthy_vs_unknown_are_different(self):
        """This is the key disambiguation — healthy ≠ unknown."""
        mon = make_pokemon(status=None)
        healthy = encode_status(mon, known=True)
        unknown = encode_status(mon, known=False)
        assert not np.array_equal(healthy, unknown)


# ── encode_move ───────────────────────────────────────────────────────

class TestEncodeMove:
    def test_size(self):
        move = make_move()
        assert encode_move(move, None).shape == (MOVE_SIZE,)

    def test_unknown_returns_all_zeros(self):
        move = make_move()
        vec = encode_move(move, None, known=False)
        assert np.all(vec == UNKNOWN)

    def test_none_move_exists_is_false(self):
        vec = encode_move(None, None, known=True)
        assert vec[0] == FALSE  # exists = FALSE

    def test_existing_move_exists_is_true(self):
        move = make_move()
        vec = encode_move(move, None, known=True)
        assert vec[0] == TRUE  # exists = TRUE

    def test_disabled_flag(self):
        move = make_move()
        enabled = encode_move(move, None, is_disabled=False)
        disabled = encode_move(move, None, is_disabled=True)
        assert enabled[1] == FALSE
        assert disabled[1] == TRUE

    def test_all_values_in_bounds(self):
        move = make_move("thunderbolt", base_power=95, type_name="ELECTRIC",
                         category="SPECIAL", accuracy=100.0)
        vec = encode_move(move, None)
        assert np.all(vec >= -1.0), f"Values below -1: {vec[vec < -1.0]}"
        assert np.all(vec <= 1.0), f"Values above 1: {vec[vec > 1.0]}"

    def test_always_hits_move_accuracy(self):
        """Moves with accuracy=True (always hits) encode as TRUE."""
        move = make_move(accuracy=True)
        vec = encode_move(move, None)
        # accuracy is at index: 1(exists)+1(disabled)+3(cat)+15(type)+1(priority) = 21
        acc_idx = 1 + 1 + 3 + N_TYPES + 1
        assert vec[acc_idx] == TRUE

    def test_pp_fraction_unknown_when_no_pp_data(self):
        move = make_move(current_pp=None, max_pp=None)
        vec = encode_move(move, None)
        # pp_fraction index: 1+1+3+15+1+1+2+1+1 = 26
        pp_idx = 1 + 1 + 3 + N_TYPES + 1 + 1 + 2 + 1 + 1
        assert vec[pp_idx] == UNKNOWN

    def test_physical_category_encoding(self):
        move = make_move(category="PHYSICAL")
        vec = encode_move(move, None)
        cat_start = 2  # after exists + disabled
        assert vec[cat_start] == TRUE      # physical
        assert vec[cat_start + 1] == FALSE  # special
        assert vec[cat_start + 2] == FALSE  # status

    def test_special_category_encoding(self):
        move = make_move(category="SPECIAL")
        vec = encode_move(move, None)
        cat_start = 2
        assert vec[cat_start] == FALSE
        assert vec[cat_start + 1] == TRUE
        assert vec[cat_start + 2] == FALSE


# ── encode_moves ──────────────────────────────────────────────────────

class TestEncodeMoves:
    def test_size(self):
        moves = [make_move() for _ in range(4)]
        vec = encode_moves(moves, None)
        assert vec.shape == (MOVES_SIZE,)

    def test_fewer_than_4_moves_pads_correctly_known(self):
        """Our Pokémon with 2 moves: slots 2-3 are FALSE (confirmed empty)."""
        moves = [make_move(), make_move()]
        vec = encode_moves(moves, None, unknown_empty_slots=False)
        slot3 = vec[2 * MOVE_SIZE: 3 * MOVE_SIZE]
        slot4 = vec[3 * MOVE_SIZE: 4 * MOVE_SIZE]
        # exists flag (index 0) should be FALSE for empty slots
        assert slot3[0] == FALSE
        assert slot4[0] == FALSE

    def test_fewer_than_4_moves_pads_unknown_for_opponent(self):
        """Opponent with 2 known moves: slots 2-3 are UNKNOWN."""
        moves = [make_move(), make_move()]
        vec = encode_moves(moves, None, unknown_empty_slots=True)
        slot3 = vec[2 * MOVE_SIZE: 3 * MOVE_SIZE]
        assert np.all(slot3 == UNKNOWN)

    def test_disabled_move_flagged(self):
        m1 = make_move("tackle")
        m2 = make_move("growl")
        vec = encode_moves([m1, m2], None, disabled_moves={m2})
        slot2 = vec[MOVE_SIZE: 2 * MOVE_SIZE]
        assert slot2[1] == TRUE  # is_disabled

    def test_four_moves_fills_all_slots(self):
        moves = [make_move(f"move{i}") for i in range(4)]
        vec = encode_moves(moves, None)
        for slot in range(N_MOVE_SLOTS):
            assert vec[slot * MOVE_SIZE] == TRUE  # all exist


# ── encode_boosts ─────────────────────────────────────────────────────

class TestEncodeBoosts:
    def test_size(self):
        mon = make_pokemon()
        assert encode_boosts(mon).shape == (6,)

    def test_zero_boosts_encode_as_zero(self):
        mon = make_pokemon(boosts={"atk": 0, "def": 0, "spe": 0, "spa": 0,
                                   "accuracy": 0, "evasion": 0})
        vec = encode_boosts(mon)
        assert np.all(vec == 0.0)

    def test_plus_6_atk_boost_encodes_as_1(self):
        mon = make_pokemon(boosts={"atk": 6, "def": 0, "spe": 0, "spa": 0,
                                   "accuracy": 0, "evasion": 0})
        vec = encode_boosts(mon)
        assert vec[0] == pytest.approx(1.0)  # atk is index 0

    def test_minus_6_def_boost_encodes_as_minus_1(self):
        mon = make_pokemon(boosts={"atk": 0, "def": -6, "spe": 0, "spa": 0,
                                   "accuracy": 0, "evasion": 0})
        vec = encode_boosts(mon)
        assert vec[1] == pytest.approx(-1.0)  # def is index 1


# ── encode_volatile ───────────────────────────────────────────────────

class TestEncodeVolatile:
    def test_size(self):
        mon = make_pokemon(effects={})
        assert encode_volatile(mon).shape == (VOLATILE_SIZE,)

    def test_no_effects_all_false(self):
        mon = make_pokemon(effects={})
        vec = encode_volatile(mon)
        assert np.all(vec == FALSE)

    def test_confusion_detected(self):
        from unittest.mock import MagicMock
        effect = MagicMock()
        effect.name = "CONFUSION"
        mon = make_pokemon(effects={effect: 1})
        vec = encode_volatile(mon)
        assert vec[0] == TRUE  # CONFUSION is index 0

    def test_disable_not_in_volatile(self):
        """DISABLE is excluded — encoded per-move instead."""
        from environment.obs_builder import GEN1_VOLATILE_EFFECTS
        assert "DISABLE" not in GEN1_VOLATILE_EFFECTS

    def test_flinch_not_in_volatile(self):
        """FLINCH is excluded — intra-turn, not observable at decision time."""
        from environment.obs_builder import GEN1_VOLATILE_EFFECTS
        assert "FLINCH" not in GEN1_VOLATILE_EFFECTS


# ── encode_base_pokemon ───────────────────────────────────────────────

class TestEncodeBasePokemon:
    def test_size(self):
        mon = make_pokemon()
        vec = encode_base_pokemon(mon, None)
        assert vec.shape == (BASE_POKEMON_SIZE,)

    def test_unknown_returns_all_zeros(self):
        mon = make_pokemon()
        vec = encode_base_pokemon(mon, None, known=False)
        assert np.all(vec == UNKNOWN)

    def test_none_returns_all_zeros(self):
        vec = encode_base_pokemon(None, None)
        assert np.all(vec == UNKNOWN)

    def test_values_in_bounds(self):
        mon = make_pokemon(types=["FIRE"],
                           stats={"hp": 300, "atk": 200, "def": 180, "spa": 190, "spe": 210},
                           moves={"flamethrower": make_move("flamethrower", base_power=90,
                                                            type_name="FIRE", category="SPECIAL")})
        vec = encode_base_pokemon(mon, None)
        assert np.all(vec >= -1.0), f"Below -1: {vec[vec < -1.0]}"
        assert np.all(vec <= 1.0), f"Above 1: {vec[vec > 1.0]}"


# ── encode_active_extra ───────────────────────────────────────────────

class TestEncodeActiveExtra:
    def test_size(self):
        mon = make_pokemon()
        vec = encode_active_extra(mon)
        assert vec.shape == (ACTIVE_EXTRA_SIZE,)

    def test_unknown_returns_all_zeros(self):
        mon = make_pokemon()
        vec = encode_active_extra(mon, known=False)
        assert np.all(vec == UNKNOWN)

    def test_must_recharge_flag(self):
        mon = make_pokemon()
        mon.must_recharge = True
        vec_recharge = encode_active_extra(mon)

        mon2 = make_pokemon()
        mon2.must_recharge = False
        vec_normal = encode_active_extra(mon2)

        recharge_idx = 6 + 9  # after N_BOOSTS + VOLATILE_SIZE
        assert vec_recharge[recharge_idx] == TRUE
        assert vec_normal[recharge_idx] == FALSE

    def test_values_in_bounds(self):
        mon = make_pokemon(
            boosts={"atk": 2, "def": -1, "spe": 0, "spa": 3, "accuracy": 0, "evasion": 0}
        )
        vec = encode_active_extra(mon)
        assert np.all(vec >= -1.0)
        assert np.all(vec <= 1.0)


# ── OBS_SIZE constant ─────────────────────────────────────────────────

class TestObsSize:
    def test_obs_size_constant_is_correct(self):
        """Verify the OBS_SIZE constant matches the bottom-up calculation."""
        expected = (
            ACTIVE_POKEMON_SIZE
            + N_BENCH_SLOTS * BENCH_POKEMON_SIZE
            + ACTIVE_POKEMON_SIZE
            + N_BENCH_SLOTS * BENCH_POKEMON_SIZE
            + 1
        )
        assert OBS_SIZE == expected
        assert OBS_SIZE == 2671

    def test_move_size_constant_is_correct(self):
        """Verify MOVE_SIZE = 47."""
        assert MOVE_SIZE == 47

    def test_base_pokemon_size_constant_is_correct(self):
        """Verify BASE_POKEMON_SIZE = 216."""
        assert BASE_POKEMON_SIZE == 216

    def test_active_extra_size_constant_is_correct(self):
        """Verify ACTIVE_EXTRA_SIZE = 34."""
        assert ACTIVE_EXTRA_SIZE == 34
