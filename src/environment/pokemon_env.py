"""
Single-agent Gymnasium wrapper around poke-env's PokeEnv (PettingZoo API).

Architecture:
  _Gen1PokeEnv owns two internal _EnvPlayer agents that battle each other:
    agent1 - our RL agent (actions from gym.step)
    agent2 - opponent (actions from OpponentPolicy)

  PokemonEnv wraps _Gen1PokeEnv and exposes the standard gym.Env interface
  that Stable-Baselines3 expects. All async/WebSocket complexity is handled
  by PokeEnv internally.

Username note:
  The local Showdown server (--no-security) mangles usernames that start
  with underscores or contain special characters. We pass explicit
  AccountConfiguration objects with clean alphanumeric names to avoid
  login assertion failures.
"""

from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from poke_env import LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.battle import Battle
from poke_env.environment.env import PokeEnv
from poke_env.player.battle_order import (
    BattleOrder,
    ForfeitBattleOrder,
    SingleBattleOrder,
)
from poke_env.ps_client import AccountConfiguration

from environment.obs_builder import OBS_SIZE, build_observation
from environment.reward import SparseReward

N_MOVES     = 4
N_SWITCHES  = 5
ACTION_SIZE = N_MOVES + N_SWITCHES  # 9


# ── Action ↔ order conversion ────────────────────────────────────────

def action_to_order(
    action: int,
    battle: AbstractBattle,
    fake: bool = False,
    strict: bool = False,
) -> BattleOrder:
    """
    Maps integer action (0-8) to a BattleOrder.
      0-3 → use move slot 0-3
      4-8 → switch to bench slot 0-4
    Falls back to a random valid order on invalid or out-of-range actions.
    """
    if isinstance(battle, Battle):
        if action < N_MOVES:
            moves = list(battle.available_moves)
            if action < len(moves):
                return SingleBattleOrder(moves[action])
        else:
            switch_idx = action - N_MOVES
            switches = list(battle.available_switches)
            if switch_idx < len(switches):
                return SingleBattleOrder(switches[switch_idx])

        # Fallback: any valid move or switch
        if battle.available_moves:
            return SingleBattleOrder(random.choice(battle.available_moves))
        if battle.available_switches:
            return SingleBattleOrder(random.choice(battle.available_switches))

    return ForfeitBattleOrder()


def order_to_action(
    order: BattleOrder,
    battle: AbstractBattle,
    fake: bool = False,
    strict: bool = False,
) -> int:
    """Reverse mapping - required by PokeEnv interface."""
    if not isinstance(battle, Battle):
        return 0
    if isinstance(order, SingleBattleOrder):
        target = order.order
        if hasattr(target, "base_power"):  # Move
            moves = list(battle.available_moves)
            if target in moves:
                return moves.index(target)
        else:  # Pokémon (switch)
            switches = list(battle.available_switches)
            if target in switches:
                return N_MOVES + switches.index(target)
    return 0


# ── Opponent policy ──────────────────────────────────────────────────

class RandomOpponentPolicy:
    """
    Picks a random valid action for agent2.
    This is the default opponent policy - no separate Player instance
    required, no additional server connection opened.
    """

    def choose_action(self, battle: AbstractBattle) -> int:
        if not isinstance(battle, Battle):
            return 0
        valid: list[int] = (
            list(range(len(battle.available_moves)))
            + [N_MOVES + i for i in range(len(battle.available_switches))]
        )
        return random.choice(valid) if valid else 0


# ── Concrete PokeEnv subclass ────────────────────────────────────────

class _Gen1PokeEnv(PokeEnv):
    """
    Minimal concrete subclass of PokeEnv for Gen 1 single battles.
    Implements the four abstract methods required by the base class.
    All game logic (obs, reward) lives in PokemonEnv above this layer.
    """

    def calc_reward(self, battle: AbstractBattle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=0.15,
            hp_value=0.01,
            victory_value=1.0,
        )

    def embed_battle(self, battle: AbstractBattle) -> np.ndarray:
        return build_observation(battle)

    @staticmethod
    def action_to_order(
        action: int, battle: Any, fake: bool = False, strict: bool = False
    ) -> BattleOrder:
        return action_to_order(action, battle, fake=fake, strict=strict)

    @staticmethod
    def order_to_action(
        order: BattleOrder, battle: Any, fake: bool = False, strict: bool = False
    ) -> int:
        return order_to_action(order, battle, fake=fake, strict=strict)

    @staticmethod
    def get_action_mask(battle: Any) -> list[int]:
        if not isinstance(battle, Battle):
            return [1] * ACTION_SIZE
        mask = [0] * ACTION_SIZE
        for i in range(min(len(battle.available_moves), N_MOVES)):
            mask[i] = 1
        for i in range(min(len(battle.available_switches), N_SWITCHES)):
            mask[N_MOVES + i] = 1
        return mask

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        return ACTION_SIZE


# ── Single-agent Gymnasium wrapper ───────────────────────────────────

class PokemonEnv(gym.Env):
    """
    Single-agent Gymnasium environment for Gen 1 Pokémon battles.

    agent1 = our RL agent   → actions come from gym.step()
    agent2 = opponent       → actions come from OpponentPolicy.choose_action()

    Both agents are internal to _Gen1PokeEnv - no additional Player
    instances connect to the server.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_policy=None,
        reward_fn=None,
        server_configuration: ServerConfiguration = LocalhostServerConfiguration,
        battle_format: str = "gen1randombattle",
    ):
        super().__init__()

        self._reward_fn      = reward_fn or SparseReward()
        self._opponent_policy = opponent_policy or RandomOpponentPolicy()

        self._poke_env = _Gen1PokeEnv(
            # Clean alphanumeric usernames - avoids server username mangling
            account_configuration1=AccountConfiguration("RLAgent", None),
            account_configuration2=AccountConfiguration("RLOpponent", None),
            battle_format=battle_format,
            server_configuration=server_configuration,
            start_listening=True,
            choose_on_teampreview=False,
        )

        self._agent1_name: str | None = None
        self._agent2_name: str | None = None

    # ── Gym spaces ───────────────────────────────────────────────────

    @property
    def observation_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_SIZE,), dtype=np.float32
        )

    @property
    def action_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(ACTION_SIZE)

    # ── Gym interface ────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        observations, infos = self._poke_env.reset(seed=seed, options=options)

        self._agent1_name = self._poke_env.possible_agents[0]
        self._agent2_name = self._poke_env.possible_agents[1]

        obs = observations[self._agent1_name]["observation"]
        return obs, infos.get(self._agent1_name, {})

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._agent1_name is not None, "call reset() before step()"

        # agent2's action from opponent policy
        battle2 = self._poke_env.battle2
        agent2_action = self._opponent_policy.choose_action(battle2)

        observations, rewards, terminated, truncated, infos = \
            self._poke_env.step({
                self._agent1_name: action,
                self._agent2_name: agent2_action,
            })

        obs   = observations[self._agent1_name]["observation"]
        rew   = rewards[self._agent1_name]
        term  = terminated[self._agent1_name]
        trunc = truncated[self._agent1_name]
        info  = infos.get(self._agent1_name, {})

        return obs, rew, term, trunc, info

    def close(self):
        self._poke_env.close()
