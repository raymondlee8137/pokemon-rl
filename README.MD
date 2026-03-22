# Reinforcement Learning Agent for Pokémon Red

## Motivation

I originally watched Peter Whidden's [Training AI to Play Pokemon with Reinforcement Learning](https://www.youtube.com/watch?v=DcYLT37ImBY&t=967s&pp=ygUNcGV0ZXIgd2hpZGRlbg%3D%3D) and felt very passionately about his experiment. I've played Pokémon since I was a kid and have always found myself intrigued by autonomy. Seeing Peter Whidden's experiment inspired me to attempt the project myself.

The aim of this project is to improve upon Whidden's model and address some direct issues he mentioned. One key difference is that my model uses RAM-based observations rather than Whidden's CNN approach. This is both a practical choice given hardware constraints and a deliberate architectural decision. 

1. Removing the shortcut
- Whidden explicitly mentions that his model starts after obtaining the Pokedex because the back tracking required was difficult for his model to do. My initial plan is to tackle this by using more dense rewards and the hierarchical approach described below.
2. A hierarchical approach
- Whidden uses a flat PPO model. That means one model needs to learn all aspects of the game, including both high-level decision making and low-level execution. I aim to apply a meta-controller that can coordinate sub-policies.

## Why is Pokémon challenging?

Pokémon Red provides the full complexity profile of real autonomy problems: partial observability (opponent stats unknown in battle), long time horizons (thousands of steps between meaningful milestones), mixed objectives (explore, fight, heal, progress story), and adversarial elements (trainer and wild battles).

## Architecture

The system uses the **Options Framework** (Sutton, Precup & Singh, 1999) with three layers:

```
┌─────────────────────────────────────────────────┐
│               Meta-Controller                   │
│       Reads game state → selects sub-policy     │
└──────────┬────────────────┬────────────┬────────┘
           │                │            │
           ▼                ▼            ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Navigation  │ │    Battle    │ │     Menu     │
│    Policy    │ │    Policy    │ │    Policy    │
│  (Learned)   │ │  (Learned)   │ │   (Rules)    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       ▼                ▼                ▼
┌─────────────────────────────────────────────────┐
│            PyBoy Emulator (RAM I/O)             │
│       16 parallel instances via SubprocVecEnv   │
└─────────────────────────────────────────────────┘
```

**Observations** are RAM-based, not pixel-based. The agent reads structured game state directly from memory addresses (player position, map ID, party stats, battle flags, event flags).

## Tech Stack

| Component | Tool |
|---|---|
| Emulator | [PyBoy](https://github.com/Baekalfen/PyBoy) |
| RL Framework | [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) (PPO) |
| Environment API | [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) |
| Parallelism | `SubprocVecEnv` (16 environments) |
| Experiment Tracking | TensorBoard |
| RAM Reference | [pret/pokered](https://github.com/pret/pokered) disassembly |

## Project Phases

### Phase 0: Foundations — Proof of Concept

**Objective:** Build the environment, training infrastructure, and reward system. Prove the agent can learn meaningful behavior.

**Status:** Complete. Agent achieves **100% starter Pokémon pickup** on non-deterministic evaluation after ~10M timesteps.

Key outcomes:
- Built Gymnasium-compatible environment wrapper with RAM-based observations
- Implemented reward shaping (exploration, events, levels, step penalty)
- Discovered and fixed reward magnitude issues (A-button spam from over-weighted event rewards)
- Tuned hyperparameters for Pokémon's long-horizon dynamics — full-episode rollouts (`n_steps = max_steps`) proved critical
- Built comprehensive logging: per-episode reward components, action distributions, rolling statistics, checkpoint metadata

See [`phase_0_setup/README.md`](phase_0_setup/README.md) for detailed findings, TensorBoard analysis, and hyperparameter evolution.

### Phase 1: Flat Baseline Agent

**Objective:** Train a single flat PPO agent as the experimental baseline. Extend the agent's reach from Pallet Town through Route 1 to Viridian City.

**Key tasks:**
- Increase episode length (`max_steps`) to give the agent runway beyond starter pickup
- Refine reward shaping — balance exploration vs. event vs. level rewards
- Add intermediate event flags between Pallet Town and Pewter City for denser signal
- Establish baseline metrics for rigorous comparison against hierarchical approach
- Improve visualization method when testing models after training

**Success criteria:** Agent navigates Pallet Town → Route 1 → Viridian City → Viridian Forest → Pewter City with >60% reliability across 10 evaluation runs.

### Phase 2: Hierarchical Architecture

**Objective:** Implement the Options Framework — the core contribution of this project.

**Meta-Controller** classifies game state from RAM into modes:

| Game Mode | ID | Description |
|---|---|---|
| OVERWORLD | 0 | Walking around, no menus |
| BATTLE | 1 | In a Pokémon battle |
| MENU | 2 | In a menu (start, shop, PC) |
| TRANSITION | 3 | Map transition / loading |

**Sub-policies** are trained independently via curriculum learning before integrated fine-tuning:
- **Navigation Policy:** Overworld movement, exploration, pathfinding. Trained on save states at various map locations.
- **Battle Policy:** Move selection under partial observability. The inital plan is to pre-train using a standalone Python battle simulator so it learns general type-effectiveness principles rather than memorizing specific encounters.
- **Menu Policy:** Rule-based state machine. Deterministic, not learned.

**Target:** Agent reaches and defeats Brock (Boulder Badge) with >60% reliability.

## Stretch Goals

- **Vision-Based Observations** — Replace RAM reading with CNN over game frames, adding a perception layer
- **Extend to Misty (Cerulean Gym)** — Proves hierarchical architecture generalizes beyond one gym
- **Curriculum Learning** — Auto-adjust task difficulty (start closer to Brock, gradually move back to Pallet Town)
- **Sim-to-Sim Transfer** — Train on Red, evaluate on Blue (same engine, different wild Pokémon distribution)

## Getting Started

### Prerequisites

- Python 3.10+
- A Pokémon Red ROM file (`pokemon_red.gb`) — not included in this repo

### Installation

```bash
git clone <repo-url>
cd pokemon-rl
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Training (Phase 0)

```bash
cd phase_0_setup
python train_rl.py
```

Monitor training:

```bash
tensorboard --logdir=tb_logs
```

### Configuration

Key training parameters are at the top of `train_rl.py`:

| Parameter | Default | Description |
|---|---|---|
| `NUM_ENVS` | 16 | Parallel emulator instances |
| `MAX_STEPS_PER_EP` | 32,768 | Actions per episode |
| `EPISODES_PER_ENV` | 50 | Episodes per environment |
| `EMULATION_SPEED` | 0 | 0 = unlimited (headless) |
| `VISIBLE_ENV` | False | Show one env with SDL2 window |

## Acknowledgments & References

This project builds on the work of several open-source projects and research:

**Inspiration:**
- [**PWhiddy/PokemonRedExperiments**](https://github.com/PWhiddy/PokemonRedExperiments) by Peter Whidden — The project that demonstrated RL agents can learn to play Pokémon Red. Whidden's approach uses a single flat PPO agent with vision-based observations (CNN over stacked screen frames) and KNN-based exploration bonuses over frame embeddings. This project takes a deliberately different architectural approach: RAM-based observations and hierarchical policy decomposition. Whidden's reward shaping strategies were a valuable reference.

**Game Data:**
- [**pret/pokered**](https://github.com/pret/pokered) — The Pokémon Red disassembly project. Provided the RAM address map essential for building the observation space: player coordinates, map IDs, battle flags, party data, and event flags. This project would not be possible without the pret community's reverse engineering work.

**Tools:**
- [**PyBoy**](https://github.com/Baekalfen/PyBoy) — Game Boy emulator with Python API, enabling programmatic control and RAM access
- [**Stable-Baselines3**](https://github.com/DLR-RM/stable-baselines3) — PPO implementation and vectorized environment support
- [**Gymnasium**](https://github.com/Farama-Foundation/Gymnasium) — Standard RL environment interface

**Academic References:**
- Schulman et al. (2017) — "Proximal Policy Optimization Algorithms"
- Sutton, Precup & Singh (1999) — "Between MDPs and Semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning" (Options Framework)
- Bacon, Harb & Precup (2017) — "The Option-Critic Architecture"

## License

This project is for educational and portfolio purposes. Pokémon is a trademark of Nintendo/Game Freak/The Pokémon Company. No ROM files are included in this repository.