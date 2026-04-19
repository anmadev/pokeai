import asyncio
import os

from poke_env import LocalhostServerConfiguration, ServerConfiguration
from poke_env.player import RandomPlayer


def get_server_config() -> ServerConfiguration:
    """
    Returns server config depending on environment.
    - When running locally (Poetry): connects to localhost:8000
    - When running in Docker (agent container): connects to showdown:8000
    """
    host = os.environ.get("SHOWDOWN_HOST", "localhost")
    port = int(os.environ.get("SHOWDOWN_PORT", 8000))

    if host == "localhost":
        return LocalhostServerConfiguration
    
    return ServerConfiguration(
        server_url=f"{host}:{port}",
        authentication_url="https://play.pokemonshowdown.com/action.php",
    )


async def main():
    server_config = get_server_config()

    # Two random agents that will fight each other
    player_1 = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=server_config,
        max_concurrent_battles=1,
    )

    player_2 = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=server_config,
        max_concurrent_battles=1,
    )

    n_battles = 20

    print(f"Starting {n_battles} battles between two random agents...")
    await player_1.battle_against(player_2, n_battles=n_battles)

    print("\n--- Results ---")
    print(f"Player 1 | Wins: {player_1.n_won_battles} / {n_battles}")
    print(f"Player 2 | Wins: {player_2.n_won_battles} / {n_battles}")


if __name__ == "__main__":
    asyncio.run(main())
