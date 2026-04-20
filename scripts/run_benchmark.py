import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.max_damage_agent import MaxDamagePlayer
from src.agents.heuristic_agent import HeuristicPlayer
from src.utils.benchmark import benchmark
from src.utils.logger import get_logger, log_benchmark_result

logger = get_logger("run_benchmark")


async def main():
    server_config = LocalhostServerConfiguration

    matchups = [
        (RandomPlayer,      RandomPlayer,      "sanity check"),
        (MaxDamagePlayer,   RandomPlayer,      "max damage vs random"),
        (HeuristicPlayer,   RandomPlayer,      "heuristic vs random"),
        (HeuristicPlayer,   MaxDamagePlayer,   "heuristic vs max damage"),
    ]

    all_results = []

    for player_a_class, player_b_class, description in matchups:
        logger.info(
            f"Running: {player_a_class.__name__} vs "
            f"{player_b_class.__name__} — {description}"
        )
        result = await benchmark(
            player_a_class=player_a_class,
            player_b_class=player_b_class,
            server_config=server_config,
            n_battles=100,
        )
        log_benchmark_result(result)
        all_results.append(result)

    print("\n========== FULL BENCHMARK SUMMARY ==========")
    print(f"{'Matchup':<50} {'A winrate':>10} {'B winrate':>10}")
    print("-" * 72)
    for r in all_results:
        matchup = f"{r['player_a']} vs {r['player_b']}"
        print(f"{matchup:<50} {r['player_a_winrate']:>9}% {r['player_b_winrate']:>9}%")


if __name__ == "__main__":
    asyncio.run(main())
