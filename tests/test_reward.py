"""
Unit and integration tests for environment/reward.py.

Tests use make_battle() to construct minimal mock battle states.
No server connection required.
"""

from __future__ import annotations

import pytest

from environment.reward import ShapedReward, SparseReward
from tests.conftest import make_battle, make_pokemon


# ── SparseReward ──────────────────────────────────────────────────────

class TestSparseReward:
    def setup_method(self):
        self.reward = SparseReward()

    def test_win_returns_positive(self):
        battle = make_battle(won=True, lost=False)
        assert self.reward.reward(battle, battle) == pytest.approx(1.0)

    def test_loss_returns_negative(self):
        battle = make_battle(won=False, lost=True)
        assert self.reward.reward(battle, battle) == pytest.approx(-1.0)

    def test_ongoing_returns_zero(self):
        battle = make_battle(won=False, lost=False)
        assert self.reward.reward(battle, battle) == pytest.approx(0.0)

    def test_custom_win_value(self):
        reward = SparseReward(win_value=2.0, loss_value=-2.0)
        battle = make_battle(won=True)
        assert reward.reward(battle, battle) == pytest.approx(2.0)

    def test_win_takes_priority_over_ongoing(self):
        """Won=True overrides ongoing, regardless of other state."""
        battle = make_battle(won=True, lost=False)
        prev = make_battle(won=False, lost=False)
        assert self.reward.reward(battle, prev) == pytest.approx(1.0)


# ── ShapedReward ──────────────────────────────────────────────────────

class TestShapedReward:
    def setup_method(self):
        self.reward = ShapedReward(
            win_value=1.0,
            loss_value=-1.0,
            faint_weight=0.15,
            hp_weight=0.01,
        )

    def test_win_terminal(self):
        battle = make_battle(won=True)
        assert self.reward.reward(battle, battle) == pytest.approx(1.0)

    def test_loss_terminal(self):
        battle = make_battle(lost=True)
        assert self.reward.reward(battle, battle) == pytest.approx(-1.0)

    def test_no_change_returns_zero(self):
        """Identical battle states produce zero shaped reward."""
        mon = make_pokemon(current_hp_fraction=1.0, fainted=False)
        opp = make_pokemon(current_hp_fraction=1.0, fainted=False)
        battle = make_battle(
            active=mon,
            opp_active=opp,
            team={"mon": mon},
            opp_team={"opp": opp},
        )
        assert self.reward.reward(battle, battle) == pytest.approx(0.0)

    def test_opponent_fainted_positive_reward(self):
        """KOing an opponent gives +faint_weight."""
        opp_alive = make_pokemon(current_hp_fraction=0.5, fainted=False)
        opp_fainted = make_pokemon(current_hp_fraction=0.0, fainted=True)
        our_mon = make_pokemon(current_hp_fraction=1.0, fainted=False)

        prev = make_battle(
            active=our_mon, opp_active=opp_alive,
            team={"mon": our_mon}, opp_team={"opp": opp_alive},
        )
        curr = make_battle(
            active=our_mon, opp_active=opp_fainted,
            team={"mon": our_mon}, opp_team={"opp": opp_fainted},
        )

        r = self.reward.reward(curr, prev)
        # faint_weight (0.15) + hp_delta (opp 0.5→0.0, so +0.005) = 0.155
        assert r == pytest.approx(0.155, abs=1e-6)

    def test_our_pokemon_fainted_negative_reward(self):
        """Losing one of our Pokémon gives -faint_weight."""
        our_alive = make_pokemon(current_hp_fraction=0.5, fainted=False)
        our_fainted = make_pokemon(current_hp_fraction=0.0, fainted=True)
        opp_mon = make_pokemon(current_hp_fraction=1.0, fainted=False)

        prev = make_battle(
            active=our_alive, opp_active=opp_mon,
            team={"mon": our_alive}, opp_team={"opp": opp_mon},
        )
        curr = make_battle(
            active=our_fainted, opp_active=opp_mon,
            team={"mon": our_fainted}, opp_team={"opp": opp_mon},
        )

        r = self.reward.reward(curr, prev)
        # -faint_weight (-0.15) + hp_delta (ours 0.5→0.0, so -0.005) = -0.155
        assert r == pytest.approx(-0.155, abs=1e-6)

    def test_hp_advantage_positive(self):
        """Gaining HP advantage (relative) gives positive reward."""
        our_mon = make_pokemon(current_hp_fraction=1.0, fainted=False)
        opp_prev = make_pokemon(current_hp_fraction=1.0, fainted=False)
        opp_curr = make_pokemon(current_hp_fraction=0.5, fainted=False)

        prev = make_battle(team={"mon": our_mon}, opp_team={"opp": opp_prev})
        curr = make_battle(team={"mon": our_mon}, opp_team={"opp": opp_curr})

        r = self.reward.reward(curr, prev)
        # HP delta = (1.0 - 0.5) - (1.0 - 1.0) = 0.5 net gain → 0.5 * 0.01 = 0.005
        assert r == pytest.approx(0.005, abs=1e-6)

    def test_hp_advantage_negative(self):
        """Losing HP relative to opponent gives negative reward."""
        our_prev = make_pokemon(current_hp_fraction=1.0, fainted=False)
        our_curr = make_pokemon(current_hp_fraction=0.5, fainted=False)
        opp_mon = make_pokemon(current_hp_fraction=1.0, fainted=False)

        prev = make_battle(team={"mon": our_prev}, opp_team={"opp": opp_mon})
        curr = make_battle(team={"mon": our_curr}, opp_team={"opp": opp_mon})

        r = self.reward.reward(curr, prev)
        assert r < 0.0

    def test_custom_weights(self):
        reward = ShapedReward(faint_weight=1.0, hp_weight=0.0)
        opp_alive = make_pokemon(fainted=False)
        opp_fainted = make_pokemon(fainted=True)
        our_mon = make_pokemon(fainted=False)

        prev = make_battle(team={"mon": our_mon}, opp_team={"opp": opp_alive})
        curr = make_battle(team={"mon": our_mon}, opp_team={"opp": opp_fainted})

        assert reward.reward(curr, prev) == pytest.approx(1.0)

    def test_team_hp_helper(self):
        mon_a = make_pokemon(current_hp_fraction=0.8, fainted=False)
        mon_b = make_pokemon(current_hp_fraction=0.6, fainted=False)
        opp = make_pokemon(current_hp_fraction=0.5, fainted=False)
        battle = make_battle(team={"a": mon_a, "b": mon_b}, opp_team={"opp": opp})

        hp = ShapedReward._team_hp(battle)
        # (0.8 + 0.6) - 0.5 = 0.9
        assert hp == pytest.approx(0.9, abs=1e-6)

    def test_fainted_count_helper_ours(self):
        alive = make_pokemon(fainted=False)
        dead = make_pokemon(fainted=True)
        battle = make_battle(team={"a": alive, "b": dead})
        assert ShapedReward._fainted_count(battle, ours=True) == 1

    def test_fainted_count_helper_opponent(self):
        opp1 = make_pokemon(fainted=True)
        opp2 = make_pokemon(fainted=True)
        battle = make_battle(opp_team={"a": opp1, "b": opp2})
        assert ShapedReward._fainted_count(battle, ours=False) == 2
