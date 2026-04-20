import asyncio
import os

from poke_env import LocalhostServerConfiguration, ServerConfiguration
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player import Player


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


def estimate_move_power(move: Move, opponent: Pokemon) -> float:
    """
    Estimates the effective power of a move against a specific opponent.

    Effective power = base_power × type_effectiveness_multiplier

    Notes:
    - Moves with base_power == 0 are non-damaging (status moves like Thunder Wave,
      Growl, Recover) — we assign them 0 so they're always picked last.
    - type_multiplier accounts for type effectiveness (0.5, 1, 2) based on the
      move's type vs the opponent's type(s). This is the key improvement over
      blindly picking the highest base power.
    - In Gen 1 there are no abilities and no held items, so this estimate is
      clean — no hidden multipliers to worry about yet.
    """
    if move.base_power == 0:
        return 0.0

    type_multiplier = opponent.damage_multiplier(move)
    return move.base_power * type_multiplier


class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            best_move = max(
                battle.available_moves,
                key=lambda move: estimate_move_power(
                    move, battle.opponent_active_pokemon
                ),
            )
            return self.create_order(best_move)

        # No moves available: must switch
        # For now: switch to the first available Pokémon
        # We will improve switching logic in the heuristic agent
        return self.choose_random_move(battle)


async def main():
    from poke_env.player import RandomPlayer

    server_config = get_server_config()

    max_damage_player = MaxDamagePlayer(
        battle_format="gen1randombattle",
        server_configuration=server_config,
        max_concurrent_battles=1,
    )

    random_player = RandomPlayer(
        battle_format="gen1randombattle",
        server_configuration=server_config,
        max_concurrent_battles=1,
    )

    n_battles = 100

    print(f"Starting {n_battles} battles: MaxDamage vs Random...")
    await max_damage_player.battle_against(random_player, n_battles=n_battles)

    print("\n--- Results ---")
    print(f"MaxDamage | Wins: {max_damage_player.n_won_battles} / {n_battles} ({max_damage_player.n_won_battles}%)")
    print(f"Random    | Wins: {random_player.n_won_battles} / {n_battles} ({random_player.n_won_battles}%)")


if __name__ == "__main__":
    asyncio.run(main())
