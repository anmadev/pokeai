from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player import Player


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


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

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
