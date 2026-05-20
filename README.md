# pokeai

Reinforcement learning agent for Pokémon Showdown Gen 1 random battles,
built from scratch using poke-env, Stable-Baselines3, and (upcoming) RLlib.

The project trains agents of increasing sophistication — from rule-based
heuristics to PPO with self-play and league training — and benchmarks them
against each other to measure what each approach actually contributes.

---

## Results

### Heuristic Baselines (1000 battles each)

| Agent A | Agent B | A win rate | B win rate | Notes |
|---|---|---|---|---|
| MaxDamage | Random | ~98.4% | ~1.5% | Type-effective moves dominate Gen 1 |
| Heuristic | Random | ~97.6% | ~2.1% | Switching logic adds marginal value |
| Heuristic | MaxDamage | ~47.7% | ~50.9% | Switching costs turns in offense-dominant Gen 1 |

**Key finding:** Gen 1 is offense-dominant enough that a pure max-damage
strategy outperforms a survival-aware heuristic.

### RL Training (PPO, Stage 1 — vs Random)

| Metric | Value |
|---|---|
| Total timesteps | 100,000 |
| Final ep_rew_mean | 1.23 (shaped reward) |
| Final ep_len_mean | 48.8 turns |
| Entropy loss | −2.19 → −0.96 (policy converged) |
| Opponent | RandomOpponentPolicy |

Full training run: [wandb.ai/anmastdev-freelance/pokeai](https://wandb.ai/anmastdev-freelance/pokeai)

---

## Architecture

### Reward

#### Shaped (the one used so far)
```
r = win/loss terminal (±1.0)
  + HP advantage delta per step (×0.01)
  + opponent KO events (×0.15)
  − own KO events (×0.15)
```

#### Sparse

Win = +1, Loss = -1, everything else = 0.

### Observation Vector (2623 floats)

Encodes the full battle state under partial information:

| Section | Dims | Content |
|---|---|---|
| Our active Pokémon | 246 | Types, stats, HP, status, moves ×4, boosts, volatile effects |
| Our bench ×5 | 1065 | Same minus active-only features |
| Opponent active | 246 | Same, unknown move slots encoded as 0.0 |
| Opponent bench ×5 | 1065 | Revealed features only |
| Turn | 1 | Normalized turn counter |

All values normalized to [−1, 1]. Unknown information encoded as 0.0
(distinct from known-false at −1.0 — the network can learn the difference).

### Action Space

9 discrete actions: move slots 0–3, switch slots 0–4.
Invalid actions fall back to a random valid move during training.

---

## Agents

| Agent | File | Description |
|---|---|---|
| `RandomPlayer` | poke-env built-in | Uniform random valid action |
| `MaxDamagePlayer` | `src/agents/max_damage_agent.py` | Highest type-effective damage move |
| `HeuristicPlayer` | `src/agents/heuristic_agent.py` | Switching logic, survival awareness, OHKO detection |
| `RLPlayer` | `src/agents/rl_agent.py` | Loads trained PPO checkpoint, runs inference |

---

## Project Structure

```
pokeai/
├── src/
│   ├── agents/          # All player implementations
│   ├── engine/          # Gen 1 damage calculator
│   ├── environment/     # Gym env, obs builder, reward, opponent policies
│   └── utils/           # Logging, benchmarking
├── scripts/
│   ├── run_benchmark.py # Heuristic agent benchmarks
│   ├── train.py         # PPO training
│   └── evaluate.py      # Evaluate trained model vs all agents
├── models/              # Saved checkpoints (not versioned)
├── logs/                # JSONL results + TensorBoard
└── notebooks/           # Analysis and visualisation
```

---

## Quickstart

### Requirements

- Python 3.11+
- Node.js (for Pokémon Showdown server)
- Poetry

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/pokeai.git
cd pokeai
poetry install

# Start local Showdown server
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown && node pokemon-showdown start --no-security
```

### Run benchmarks

```bash
python -m scripts.run_benchmark
```

### Train

```bash
# Stage 1: vs random opponent
python -m scripts.train

# Evaluate
python -m scripts.evaluate --model models/ppo_final --n_battles 100
```

---

## Roadmap

- [x] Random agent baseline
- [x] MaxDamage heuristic + type effectiveness
- [x] Gen 1 damage calculator (min/max range, multi-hit, OHKO)
- [x] Survival-aware heuristic (switching, speed, OHKO detection)
- [x] PPO training — Stage 1 vs random
- [ ] Evaluate Stage 1 model vs all heuristic agents
- [ ] PPO training — Stage 2 frozen opponent self-play (SB3)
- [ ] RLlib migration — league training, PFSP
- [ ] MovesetPrior v2 (species-conditioned unknown move slots)
- [ ] Species one-hot encoding
- [ ] Action masking

---

## Technical Notes

**Why Gen 1?**
No abilities, no held items, no weather, no special/physical split by
category. The state space is small enough to reason about while still
producing non-trivial strategic depth. Results are interpretable.

**Why the async/sync bridge matters**
poke-env communicates with Showdown via WebSocket (asyncio). SB3 expects
a synchronous `gym.Env`. PokeEnv bridges these using two thread-safe queues
per agent — battle states flow from the async thread to the sync training
loop, actions flow back. This is the core infrastructure challenge of the
project.

**Observation encoding convention**
`TRUE = 1.0`, `FALSE = −1.0`, `UNKNOWN = 0.0`. The network can distinguish
"I know this is false" from "I have no information about this" — relevant
for opponent Pokémon with unrevealed moves.

---

## Stack

| Component | Library |
|---|---|
| Battle environment | [poke-env](https://github.com/hsahovic/poke-env) |
| RL algorithm | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) — PPO |
| Experiment tracking | [Weights & Biases](https://wandb.ai/anmastdev-freelance/pokeai) |

---

## License

MIT
