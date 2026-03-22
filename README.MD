# Hierarchical RL Agent for Pokémon Red

A hierarchical reinforcement learning system that plays Pokémon Red using specialized sub-policies coordinated by a meta-controller.

**Author:** Raymond

---

## Motivation

I originally watched Peter Whidden's Youtube video "Training AI to Play Pokemon with Reinforcement Learning" and found it fascinating. As someone who's played Pokémon since I was a kid and has a growing interest in RL, I decided to attempt the project myself.

Most RL Pokémon projects use a single flat agent with reward shaping to brute-force progress. This project takes a fundamentally different approach.

One of Peter Whidden's recommended next steps forward was building a hierarchical approach. I've decided to attempt this with a meta-controller coordinating specialized sub-policies, this system separates high-level decision-making from low-level execution.

### Why is Pokémon challenging?

Pokémon Red provides the full complexity profile of real autonomy problems: partial observability (opponent stats unknown in battle), long time horizons (thousands of steps between meaningful milestones), mixed objectives (explore, fight, heal, progress story), and adversarial elements (trainer and wild battles). A gridworld would be faster to train but wouldn't force the architectural decisions that matter in production systems.

## Architecture

The system uses the **Options Framework** (Sutton, Precup & Singh, 1999) with three layers:

```
┌─────────────────────────────────────────────┐
│              Meta-Controller                │
│   Reads game state → selects sub-policy     │
│   (Behavior Planner analog)                 │
└──────┬──────────┬──────────┬────────────────┘
       │          │          │
       ▼          ▼          ▼
  ┌─────────┐ ┌────────┐ ┌────────┐
  │  Nav    │ │ Battle │ │  Menu  │
  │ Policy  │ │ Policy │ │ Policy │
  │(Learned)│ │(Learned)│ │(Rules) │
  └─────────┘ └────────┘ └────────┘
       │          │          │
       ▼          ▼          ▼
  ┌─────────────────────────────────────────┐
  │         PyBoy Emulator (RAM I/O)        │
  │     16 parallel instances via           │
  │     SubprocVecEnv                       │
  └─────────────────────────────────────────┘
```

**Observations** are RAM-based, not pixel-based. The agent reads structured game state directly from memory addresses — player position, map ID, party stats, battle flags, event flags. This is mostly due to hardware constraints. My setup would struggle to handle a vision-based approach. However, vision-based observations are a documented stretch goal.

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

### Phase 0: Foundations — Proof of Concept ✅

**Objective:** Build the environment, training infrastructure, and reward system. Prove the agent can learn meaningful behavior.

**Status:** Complete. Agent achieves **100% starter Pokémon pickup** on deterministic evaluation after ~10M timesteps.

Key outcomes:
- Built Gymnasium-compatible environment wrapper with RAM-based observations
- Implemented dense reward shaping (exploration, events, levels, step penalty)
- Discovered and fixed reward magnitude issues (A-button spam from over-weighted event rewards)
- Tuned hyperparameters for Pokémon's long-horizon dynamics — full-episode rollouts (`n_steps = max_steps`) proved critical
- Built comprehensive logging: per-episode reward components, action distributions, rolling statistics, checkpoint metadata

See [`phase_0_setup/README.md`](phase_0_setup/README.md) for detailed findings, TensorBoard analysis, and hyperparameter evolution.

### Phase 1: Flat Baseline Agent 🔄

**Objective:** Train a single flat PPO agent as the experimental baseline. Extend the agent's reach from Pallet Town through Route 1 to Viridian City.

**Key tasks:**
- Increase episode length (`max_steps`) to give the agent runway beyond starter pickup
- Refine reward shaping — balance exploration vs. event vs. level rewards
- Add intermediate event flags between Pallet Town and Pewter City for denser signal
- Establish baseline metrics for rigorous comparison against hierarchical approach

**Success criteria:** Agent navigates Pallet Town → Route 1 → Viridian City → Viridian Forest → Pewter City with >60% reliability across 10 evaluation runs.

### Phase 2: Hierarchical Architecture 📋

**Objective:** Implement the Options Framework — the core contribution of this project.

**Meta-Controller** classifies game state from RAM into modes:

| Game Mode | ID | Description |
|---|---|---|
| OVERWORLD | 0 | Walking around, no menus |
| BATTLE | 1 | In a Pokémon battle |
| MENU | 2 | In a menu (start, shop, PC) |
| DIALOGUE | 3 | Talking to NPC |
| TRANSITION | 4 | Map transition / loading |

**Sub-policies** are trained independently via curriculum learning before integrated fine-tuning:
- **Navigation Policy:** Overworld movement, exploration, pathfinding. Trained on save states at various map locations.
- **Battle Policy:** Move selection under partial observability. Pre-trained using a standalone Python battle simulator so it learns general type-effectiveness principles rather than memorizing specific encounters.
- **Menu Policy:** Rule-based state machine. Deterministic, not learned.

**Target:** Agent reaches and defeats Brock (Boulder Badge) with >60% reliability.

## Stretch Goals

- **Vision-Based Observations** — Replace RAM reading with CNN over game frames, adding a perception layer
- **Extend to Misty (Cerulean Gym)** — Proves hierarchical architecture generalizes beyond one gym
- **Curriculum Learning** — Auto-adjust task difficulty (start closer to Brock, gradually move back to Pallet Town)
- **Sim-to-Sim Transfer** — Train on Red, evaluate on Blue (same engine, different wild Pokémon distribution)
- **Backtracking** — Tackle the challenge of needing to backtrack to get the Pokedex

## Getting Started

### Prerequisites

- Python 3.10+
- A Pokémon Red ROM file (`pokemon_red.gb`) — not included in this repo
- ~8GB RAM (16 parallel emulator instances)

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