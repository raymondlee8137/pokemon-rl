import os
import numpy as np
import multiprocessing as mp
from stable_baselines3 import PPO
from pokemon_env import PokemonRedEnv


# ── Config ────────────────────────────────────────────────────────────────────
NUM_EPISODES   = 10
MAX_EVAL_STEPS = 16384*2


def run_episode(args):

    ep_num, rom_path, state_path, model_path, window_type, emu_speed = args

    # Each subprocess loads the model independently
    model = PPO.load(model_path, device="cpu")

    env = PokemonRedEnv(
        rom_path=rom_path,
        state_path=state_path,
        max_steps=MAX_EVAL_STEPS,
        emulation_speed=emu_speed,
        window_type=window_type,
    )

    obs, info = env.reset()
    ep_reward = 0.0
    steps = 0
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        ep_reward += reward
        steps += 1
        done = terminated or truncated

    tiles       = info["tiles_visited"]
    events      = info.get("events_triggered", [])
    got_starter = "got_starter" in events

    env.close()
    return (ep_num, ep_reward, steps, tiles, got_starter, events)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rom_path   = os.path.join(script_dir, "..", "pokemon_red.gb")
    state_path = os.path.join(script_dir, "..", "assets", "savestates", "pallet_town.state")
    model_path = os.path.join(script_dir, "ppo_pokemon_phase0")

    for label, path in [("ROM", rom_path), ("state", state_path), ("model", model_path + ".zip")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at {path}")
            return

    print("=" * 70)
    print(f"Evaluating: {model_path}.zip")
    print(f"Episodes: {NUM_EPISODES}  |  Max steps/ep: {MAX_EVAL_STEPS}  |  Parallel")
    print("=" * 70)
    print(f"Running {NUM_EPISODES} episodes (headless)...")

    args_list = [
        (ep, rom_path, state_path, model_path, "null", 0)
        for ep in range(1, NUM_EPISODES + 1)
    ]

    with mp.Pool(processes=min(NUM_EPISODES, 16)) as pool:
        results = pool.map(run_episode, args_list)

    # Sort by episode number
    results.sort(key=lambda r: r[0])

    ep_rewards  = []
    ep_steps    = []
    ep_tiles    = []
    ep_starter  = []
    ep_events   = []

    print()
    for ep_num, reward, steps, tiles, got_starter, events in results:
        ep_rewards.append(reward)
        ep_steps.append(steps)
        ep_tiles.append(tiles)
        ep_starter.append(got_starter)
        ep_events.append(events)

        print(
            f"  Ep {ep_num:>2d}/{NUM_EPISODES} | "
            f"Reward: {reward:>8.1f} | "
            f"Steps: {steps:>5d} | "
            f"Tiles: {tiles:>4d} | "
            f"Starter: {'YES' if got_starter else 'no ':>3s} | "
            f"Events: {events}"
        )

    # ── Summary report ───────────────────────────────────────────────────
    starter_count = sum(ep_starter)
    print()
    print("=" * 70)
    print(f"Summary over {NUM_EPISODES} episodes:")
    print(f"  Reward  — avg: {np.mean(ep_rewards):>8.1f}  std: {np.std(ep_rewards):>7.1f}  "
          f"best: {max(ep_rewards):.1f}")
    print(f"  Steps   — avg: {np.mean(ep_steps):>8.1f}  std: {np.std(ep_steps):>7.1f}  "
          f"max: {max(ep_steps)}")
    print(f"  Tiles   — avg: {np.mean(ep_tiles):>8.1f}  std: {np.std(ep_tiles):>7.1f}  "
          f"best: {max(ep_tiles)}")
    print(f"  Starter chosen: {starter_count}/{NUM_EPISODES}")
    all_seen = sorted(set(e for events in ep_events for e in events))
    print(f"  Events seen (any ep): {all_seen}")
    print("=" * 70)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()