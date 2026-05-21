"""
Shared fixtures for all test modules.

All fixtures use Mock objects to simulate poke-env structures.
No server connection is required for any test in this suite.

Fixture design principle:
  Build the minimum object that satisfies the function under test.
  poke-env objects have many attributes — we only mock what is accessed.
"""

from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock, PropertyMock

import pytest

from poke_env.battle.pokemon_type import PokemonType


# ── Type helpers ─────────────────────────────────────────────────────

def make_type(name: str) -> MagicMock:
    t = MagicMock(spec=PokemonType)
    t.name = name.upper()
    return t


# ── Move factory ─────────────────────────────────────────────────────

def make_move(
    move_id: str = "tackle",
    base_power: int = 40,
    type_name: str = "NORMAL",
    category: str = "PHYSICAL",
    accuracy: float | bool = 100.0,
    priority: int = 0,
    n_hit: tuple = (1, 1),
    crit_ratio: int = 1,
    current_pp: int = 35,
    max_pp: int = 35,
    drain: float = 0.0,
    heal: float = 0.0,
    recoil: float = 0.0,
    status=None,
    boosts: Optional[dict] = None,
    self_boost: Optional[dict] = None,
) -> MagicMock:
    move = MagicMock()
    move.id = move_id
    move.base_power = base_power
    move.type = make_type(type_name)
    move.category = MagicMock()
    move.category.name = category.upper()
    move.accuracy = accuracy
    move.priority = priority
    move.n_hit = n_hit
    move.crit_ratio = crit_ratio
    move.current_pp = current_pp
    move.max_pp = max_pp
    move.drain = drain
    move.heal = heal
    move.recoil = recoil
    move.status = status
    move.boosts = boosts or {}
    move.self_boost = self_boost or {}
    move.ignore_defensive = False
    return move


# ── Pokémon factory ───────────────────────────────────────────────────

def make_pokemon(
    species: str = "pikachu",
    types: list[str] = None,
    stats: Optional[dict] = None,
    base_stats: Optional[dict] = None,
    current_hp_fraction: float = 1.0,
    max_hp: int = 263,
    status=None,
    status_counter: int = 0,
    moves: Optional[dict] = None,
    boosts: Optional[dict] = None,
    effects: Optional[dict] = None,
    fainted: bool = False,
    level: int = 80,
) -> MagicMock:
    mon = MagicMock()
    mon.species = species
    mon.level = level
    mon.fainted = fainted

    type_list = [make_type(t) for t in (types or ["NORMAL"])]
    mon.types = type_list

    default_stats = {"hp": 263, "atk": 200, "def": 180, "spa": 190, "spe": 210}
    mon.stats = stats or default_stats

    mon.base_stats = base_stats or {"hp": 45, "atk": 49, "def": 49, "spa": 65, "spe": 45}
    mon.current_hp_fraction = current_hp_fraction
    mon.max_hp = max_hp
    mon.status = status
    mon.status_counter = status_counter
    mon.moves = moves or {}
    mon.boosts = boosts or {k: 0 for k in ["atk", "def", "spe", "spa", "accuracy", "evasion"]}
    mon.effects = effects or {}
    mon.must_recharge = False
    mon.first_turn = False
    mon.last_move = None
    mon.preparing_move = None
    mon.damage_multiplier = MagicMock(return_value=1.0)
    return mon


# ── Battle factory ────────────────────────────────────────────────────

def make_battle(
    active: MagicMock = None,
    opp_active: MagicMock = None,
    team: Optional[dict] = None,
    opp_team: Optional[dict] = None,
    available_moves: Optional[list] = None,
    available_switches: Optional[list] = None,
    turn: int = 1,
    won: bool = False,
    lost: bool = False,
) -> MagicMock:
    battle = MagicMock()

    active = active or make_pokemon("bulbasaur", types=["GRASS", "POISON"])
    opp_active = opp_active or make_pokemon("charmander", types=["FIRE"])

    battle.active_pokemon = active
    battle.opponent_active_pokemon = opp_active
    battle.team = team or {"bulbasaur": active}
    battle.opponent_team = opp_team or {"charmander": opp_active}
    battle.available_moves = available_moves or []
    battle.available_switches = available_switches or []
    battle.turn = turn
    battle.won = won
    battle.lost = lost
    return battle


# ── Pytest fixtures ───────────────────────────────────────────────────

@pytest.fixture
def basic_move():
    return make_move()


@pytest.fixture
def pikachu():
    return make_pokemon(
        species="pikachu",
        types=["ELECTRIC"],
        stats={"hp": 263, "atk": 195, "def": 158, "spa": 195, "spe": 278},
        max_hp=263,
    )


@pytest.fixture
def starmie():
    return make_pokemon(
        species="starmie",
        types=["WATER", "PSYCHIC"],
        stats={"hp": 293, "atk": 230, "def": 230, "spa": 278, "spe": 295},
        max_hp=293,
    )


@pytest.fixture
def gengar():
    return make_pokemon(
        species="gengar",
        types=["GHOST", "POISON"],
        stats={"hp": 261, "atk": 230, "def": 195, "spa": 278, "spe": 278},
        max_hp=261,
    )


@pytest.fixture
def simple_battle(pikachu, starmie):
    thunderbolt = make_move("thunderbolt", base_power=95, type_name="ELECTRIC", category="SPECIAL")
    return make_battle(
        active=pikachu,
        opp_active=starmie,
        available_moves=[thunderbolt],
        turn=1,
    )
