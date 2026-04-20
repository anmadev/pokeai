import asyncio
from typing import Type

import wandb
from poke_env import ServerConfiguration
from poke_env.player import Player


async def benchmark(
    player_a_class: Type[Player],
    player_b_class: Type[Player],
    server_config: ServerConfiguration,
    n_battles: int = 100,
    battle_format: str = "gen1randombattle",
    wandb_project: str = "pokeai",
) -> dict:
    """
    Runs n_battles between two player classes, prints results,
    logs them to wandb and returns the results dict.
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
        "battle_format": battle_format,
        "n_battles": n_battles,
        "player_a_wins": player_a.n_won_battles,
        "player_b_wins": player_b.n_won_battles,
        "player_a_winrate": round(player_a.n_won_battles / n_battles * 100, 1),
        "player_b_winrate": round(player_b.n_won_battles / n_battles * 100, 1),
    }

    print(f"\n--- {results['player_a']} vs {results['player_b']} ({n_battles} battles) ---")
    print(f"{results['player_a']:<25} {results['player_a_wins']:>4} wins  ({results['player_a_winrate']}%)")
    print(f"{results['player_b']:<25} {results['player_b_wins']:>4} wins  ({results['player_b_winrate']}%)")

    # --- wandb logging ---
    run = wandb.init(
        project=wandb_project,
        name=f"{player_a_class.__name__}_vs_{player_b_class.__name__}",
        config={
            "battle_format": battle_format,
            "n_battles": n_battles,
        },
        reinit=True,   # allows multiple runs in the same Python process
    )

    wandb.log({
        "player_a_wins": results["player_a_wins"],
        "player_b_wins": results["player_b_wins"],
        "player_a_winrate": results["player_a_winrate"],
        "player_b_winrate": results["player_b_winrate"],
    })

    # Summary values appear as the headline numbers on the wandb run page
    wandb.run.summary["player_a_winrate"] = results["player_a_winrate"]
    wandb.run.summary["player_b_winrate"] = results["player_b_winrate"]

    run.finish()

    return results
