"""
PPO training script for the Pokémon Showdown RL agent.

Curriculum:
  Stage 1 — vs RandomPlayer      (until win rate > 80%)
  Stage 2 — vs MaxDamagePlayer   (until win rate > 70%)
  Stage 3 — vs HeuristicPlayer   (ongoing)

Run:
  python scripts/train.py

Monitor:
  tensorboard --logdir logs/tensorboard
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import wandb

load_dotenv()

from agents.heuristic_agent import HeuristicPlayer
from agents.max_damage_agent import MaxDamagePlayer
from environment.pokemon_env import PokemonEnv, RandomOpponentPolicy
from environment.reward import ShapedReward, SparseReward
from utils.logger import get_logger

logger = get_logger("train")

# ── Config ────────────────────────────────────────────────────────────

CONFIG = {
    # Training
    "total_timesteps":  100_000,
    "checkpoint_freq":   10_000,
    # PPO hyperparameters
    "learning_rate":     3e-4,
    "n_steps":           2048,
    "batch_size":        64,
    "n_epochs":          10,
    "gamma":             0.99,
    "gae_lambda":        0.95,
    "clip_range":        0.2,
    # Environment
    "battle_format":     "gen1randombattle",
    "reward":            "sparse",   # "sparse" | "shaped"
    "opponent":          "random",   # "random" | "max_damage" | "heuristic"
    # Paths
    "model_dir":         "models",
    "run_name":          "ppo-v1-vs-random",
}

OPPONENT_MAP = {
    "random":     RandomPlayer,
    "max_damage": MaxDamagePlayer,
    "heuristic":  HeuristicPlayer,
}

REWARD_MAP = {
    "sparse":  SparseReward(),
    "shaped":  ShapedReward(),
}


# ── Callbacks ─────────────────────────────────────────────────────────

class WandbCallback(BaseCallback):
    """
    Logs training metrics and saves model checkpoints to wandb.

    Tracks:
      - ep_rew_mean  : mean episode reward over recent rollouts
      - ep_len_mean  : mean episode length (proxy for battle duration)
      - timestep     : absolute training step counter

    Checkpoints are saved locally under models/ and uploaded as
    wandb artifacts so any historical version is recoverable.
    """

    def __init__(self, check_freq: int, model_dir: str, reward: str, verbose: int = 0):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.model_dir  = Path(model_dir)
        self.reward     = reward
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        metrics: dict = {"timestep": self.num_timesteps}

        # SB3 Monitor wrapper writes ep info into ep_info_buffer
        buf = getattr(self.model, "ep_info_buffer", None)
        if buf:
            metrics["ep_rew_mean"] = sum(e["r"] for e in buf) / len(buf)
            metrics["ep_len_mean"] = sum(e["l"] for e in buf) / len(buf)

        wandb.log(metrics)
        logger.info(
            f"Step {self.num_timesteps} | "
            f"ep_rew_mean={metrics.get('ep_rew_mean', 'n/a'):.3f}"
            if "ep_rew_mean" in metrics
            else f"Step {self.num_timesteps}"
        )

        # Save checkpoint locally
        ckpt = self.model_dir / f"ppo_{self.reward}_step_{self.num_timesteps}"
        self.model.save(str(ckpt))

        # Upload as wandb artifact
        artifact = wandb.Artifact(
            name=f"model-step-{self.num_timesteps}",
            type="model",
        )
        artifact.add_file(str(ckpt) + ".zip")
        wandb.log_artifact(artifact)

        return True


# ── Training entry point ──────────────────────────────────────────────

def train():
    wandb.init(
        project="pokeai",
        name=CONFIG["run_name"],
        config=CONFIG,
    )

    server_config  = LocalhostServerConfiguration
    reward_fn      = REWARD_MAP[CONFIG["reward"]]

    # Environment — Monitor wraps it for SB3 episode tracking
    env = Monitor(
    PokemonEnv(
        opponent_policy=RandomOpponentPolicy(),  # swap class when curriculum advances
        reward_fn=reward_fn,
        server_configuration=server_config,
        battle_format=CONFIG["battle_format"],
    )
)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=CONFIG["learning_rate"],
        n_steps=CONFIG["n_steps"],
        batch_size=CONFIG["batch_size"],
        n_epochs=CONFIG["n_epochs"],
        gamma=CONFIG["gamma"],
        gae_lambda=CONFIG["gae_lambda"],
        clip_range=CONFIG["clip_range"],
        verbose=1,
        tensorboard_log="logs/tensorboard",
    )

    logger.info(
        f"Training: {CONFIG['total_timesteps']} timesteps | "
        f"opponent={CONFIG['opponent']} | reward={CONFIG['reward']}"
    )

    model.learn(
        total_timesteps=CONFIG["total_timesteps"],
        callback=WandbCallback(
            check_freq=CONFIG["checkpoint_freq"],
            model_dir=CONFIG["model_dir"],
            reward=CONFIG["reward"],
        ),
        progress_bar=True,
    )

    # Save final model
    final = Path(CONFIG["model_dir"]) / f"ppo_{CONFIG['reward']}_final"
    model.save(str(final))
    logger.info(f"Final model saved: {final}")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO agent for Pokémon Showdown")
    parser.add_argument(
        "--reward",
        choices=["sparse", "shaped"],
        default=CONFIG["reward"],
        help="Reward function (default: %(default)s)",
    )
    parser.add_argument(
        "--opponent",
        choices=["random", "max_damage", "heuristic"],
        default=CONFIG["opponent"],
        help="Opponent type (default: %(default)s)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=CONFIG["total_timesteps"],
        dest="total_timesteps",
        help="Total training timesteps (default: %(default)s)",
    )
    args = parser.parse_args()
    CONFIG["reward"] = args.reward
    CONFIG["opponent"] = args.opponent
    CONFIG["total_timesteps"] = args.total_timesteps
    train()
