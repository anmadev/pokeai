from poke_env.player import Player

from src.engine.damage_calc import best_move_by_damage


# -------------------------------------------------------------------
# Agent
# -------------------------------------------------------------------

class MaxDamagePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            result = best_move_by_damage(
                battle.active_pokemon,
                battle.opponent_active_pokemon,
                battle.available_moves,
            )
            if result is not None:
                best_move, _ = result
                return self.create_order(best_move)

        # No damaging moves or no moves available: fall back to random
        return self.choose_random_move(battle)
