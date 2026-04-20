import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poke_env import LocalhostServerConfiguration
from poke_env.player import RandomPlayer

from src.agents.max_damage_agent import MaxDamagePlayer
from src.utils.benchmark import benchmark
from src.utils.logger import get_logger, log_benchmark_result

logger = get_logger("benchmark_runner")


async def main():
    server_config = LocalhostServerConfiguration

    matchups = [
        # (Player A,          Player B,      description)
        (RandomPlayer,     RandomPlayer,    "sanity check — should be ~50/50"),
        (MaxDamagePlayer,  RandomPlayer,    "first real benchmark"),
    ]

    all_results = []

    for player_a_class, player_b_class, description in matchups:
        logger.info(f"Running: {player_a_class.__name__} vs {player_b_class.__name__} — {description}")

        result = await benchmark(
            player_a_class=player_a_class,
            player_b_class=player_b_class,
            server_config=server_config,
            n_battles=100,
        )

        log_benchmark_result(result)   # writes to logs/benchmark_results.jsonl
        all_results.append(result)

    # Final summary table in console
    print("\n========== FULL BENCHMARK SUMMARY ==========")
    print(f"{'Matchup':<45} {'A wins':>8} {'B wins':>8}")
    print("-" * 63)
    for r in all_results:
        matchup = f"{r['player_a']} vs {r['player_b']}"
        print(f"{matchup:<45} {r['player_a_winrate']:>7}% {r['player_b_winrate']:>7}%")


if __name__ == "__main__":
    asyncio.run(main())
