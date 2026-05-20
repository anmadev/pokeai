"""
Reward functions for the Pokémon RL agent.

Two implementations are provided and can be swapped via config:

  SparseReward:
    +1 on win, -1 on loss, 0 otherwise.
    Clean signal, slow convergence. Good for establishing a baseline.

  ShapedReward:
    Intermediate rewards based on HP advantage and KO events.
    Faster convergence but risks reward hacking — the agent may
    learn to optimize proxies (e.g., maximizing opponent damage
    taken) rather than actually winning.

    Shaped components:
      ± hp_weight    ← HP advantage/disadvantage each step
      + faint_weight ← reward for KOing an opponent Pokémon
      - faint_weight ← penalty for losing one of ours
      ± win_weight   ← terminal win/loss signal

Both classes expose the same interface: reward(current, previous)
so they are interchangeable in PokemonEnv.
"""

from dataclasses import dataclass

from poke_env.battle.abstract_battle import AbstractBattle


@dataclass
class SparseReward:
    """Win = +1, Loss = -1, everything else = 0."""

    win_value: float = 1.0
    loss_value: float = -1.0

    def reward(
        self,
        current_battle: AbstractBattle,
        previous_battle: AbstractBattle,
    ) -> float:
        if current_battle.won:
            return self.win_value
        if current_battle.lost:
            return self.loss_value
        return 0.0


@dataclass
class ShapedReward:
    """
    Dense reward combining HP advantage, KO events, and terminal signal.

    Weights are exposed as dataclass fields so they can be logged
    to wandb as hyperparameters and tuned systematically.
    """

    win_value: float = 1.0
    loss_value: float = -1.0
    faint_weight: float = 0.15   # per KO event
    hp_weight: float = 0.01      # per step HP delta

    def reward(
        self,
        current_battle: AbstractBattle,
        previous_battle: AbstractBattle,
    ) -> float:
        # Terminal signal
        if current_battle.won:
            return self.win_value
        if current_battle.lost:
            return self.loss_value

        score = 0.0

        # HP advantage: positive if we're healthier relative to last step
        curr_hp = self._team_hp(current_battle)
        prev_hp = self._team_hp(previous_battle)
        score += self.hp_weight * (curr_hp - prev_hp)

        # KO events: count fainted Pokémon changes since last step
        curr_our_fainted = self._fainted_count(current_battle, ours=True)
        prev_our_fainted = self._fainted_count(previous_battle, ours=True)
        curr_opp_fainted = self._fainted_count(current_battle, ours=False)
        prev_opp_fainted = self._fainted_count(previous_battle, ours=False)

        score += self.faint_weight * (curr_opp_fainted - prev_opp_fainted)
        score -= self.faint_weight * (curr_our_fainted - prev_our_fainted)

        return score

    @staticmethod
    def _team_hp(battle: AbstractBattle) -> float:
        """
        Net HP advantage: sum of our HP fractions minus opponent's.
        Range: roughly [-6, 6].
        """
        our_hp = sum(p.current_hp_fraction for p in battle.team.values())
        opp_hp = sum(
            p.current_hp_fraction for p in battle.opponent_team.values()
        )
        return our_hp - opp_hp

    @staticmethod
    def _fainted_count(battle: AbstractBattle, ours: bool) -> int:
        team = battle.team if ours else battle.opponent_team
        return sum(1 for p in team.values() if p.fainted)
