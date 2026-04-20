import asyncio
from typing import Type

from poke_env.player import Player
from poke_env import ServerConfiguration


async def benchmark(
    player_a_class: Type[Player],
    player_b_class: Type[Player],
    server_config: ServerConfiguration,
    n_battles: int = 100,
    battle_format: str = "gen1randombattle",
) -> dict:
    """
    Runs n_battles between two player classes and returns a results dict.
    Both players are instantiated fresh for each benchmark call.
    """
    player_a = player_a_class(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=1,
    )
    player_b = player_b_class(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=1,
    )

    await player_a.battle_against(player_b, n_battles=n_battles)

    results = {
        "player_a": player_a_class.__name__,
        "player_b": player_b_class.__name__,
        "n_battles": n_battles,
        "player_a_wins": player_a.n_won_battles,
        "player_b_wins": player_b.n_won_battles,
        "player_a_winrate": round(player_a.n_won_battles / n_battles * 100, 1),
        "player_b_winrate": round(player_b.n_won_battles / n_battles * 100, 1),
    }

    print(f"\n--- {results['player_a']} vs {results['player_b']} ({n_battles} battles) ---")
    print(f"{results['player_a']:<25} {results['player_a_wins']:>4} wins ({results['player_a_winrate']}%)")
    print(f"{results['player_b']:<25} {results['player_b_wins']:>4} wins ({results['player_b_winrate']}%)")

    return results
