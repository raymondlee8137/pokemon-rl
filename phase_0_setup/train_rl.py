################################################################################
#
#
#   train_rl.py is used to train the model during phase 0 and log data
#
#
################################################################################

from __future__ import annotations

import json
import os
import time
from collections import deque

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from pokemon_env import PokemonRedEnv
from constants import ACTION_MAP

# Training Configurations
NUM_ENVS = 16                   # Number of agents running at the same time
MAX_STEPS_PER_EP = 32768        # Actions per episode
EPISODES_PER_ENV = 50
TOTAL_TIMESTEPS = NUM_ENVS * MAX_STEPS_PER_EP * EPISODES_PER_ENV
LOG_INTERVAL = int(NUM_ENVS * 0.5 * MAX_STEPS_PER_EP)

VISIBLE_ENV     = False         # True = env 0 gets an SDL2 window at WATCH_ENV_SPEED
RESUME_FROM     = None          # e.g. "checkpoints/ppo_pokemon_2000000_steps"
EMULATION_SPEED = 0             # 0 = unlimited (use for headless)
WATCH_ENV_SPEED = 10            # Speed for the visible env (1 = real-time, 10 = 10x)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TB_LOG_DIR = os.path.join(SCRIPT_DIR, "tb_logs")

# Reward component keys emitted by PokemonRedEnv
_COMPONENT_KEYS = ["reward_exploration", "reward_event", "reward_level", "penalty_step"]

# Dynamic action field names derived from ACTION_MAP
_ACTION_FIELDS = [f"{ACTION_MAP[i][0]}_pct" for i in range(len(ACTION_MAP))]


def make_env(rom_path: str, state_path: str, max_steps: int,
             emulation_speed: int, window_type: str = "null"):
    """Factory function for creating environments in subprocesses."""
    def _init():
        env = PokemonRedEnv(
            rom_path=rom_path,
            state_path=state_path,
            max_steps=max_steps,
            emulation_speed=emulation_speed,
            window_type=window_type,
        )
        return env
    return _init


class RewardLoggerCallback(BaseCallback):
    """
    Logs per-episode rewards, reward components, and rolling statistics.
    Writes to TensorBoard via SB3's logger, plus CSV files as secondary output.
    Tracks all environments (not just env 0) for episode completion.
    """

    def __init__(self, log_interval: int = LOG_INTERVAL, num_envs: int = NUM_ENVS):
        super().__init__()
        self.log_interval = log_interval
        self.num_envs = num_envs

        # Per-environment tracking
        self.episode_rewards = np.zeros(num_envs)
        self.episode_count = 0
        self.starter_picked_count = 0
        self.best_episode_reward = float("-inf")

        # Rolling statistics (last 100 episodes)
        self.episode_history: deque[dict] = deque(maxlen=100)

        # Episodes completed since last periodic log — used to compute
        # per-interval means for TensorBoard (avoids last-episode-wins overwrite)
        self._pending_episodes: list[dict] = []

        # Track which event flags have been seen across ALL episodes
        self.events_ever_seen: set[str] = set()

        # Action distribution tracking (reset each log interval)
        self.action_counts = np.zeros(len(ACTION_MAP), dtype=np.int64)

        # Per-env reward component accumulators (summed across steps within an episode)
        self.component_keys = _COMPONENT_KEYS
        self.episode_components: dict[str, np.ndarray] = {
            k: np.zeros(num_envs) for k in _COMPONENT_KEYS
        }

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        infos = self.locals["infos"]
        self.episode_rewards += rewards

        # Accumulate reward components per env
        for i in range(self.num_envs):
            for key in self.component_keys:
                self.episode_components[key][i] += infos[i].get(key, 0.0)

        # Track action counts
        for action in self.locals["actions"]:
            self.action_counts[action] += 1

        # Check for completed episodes across all environments
        dones = self.locals["dones"]
        for i in range(self.num_envs):
            if dones[i]:
                self._on_episode_end(i, infos[i])

        # Periodic logging (rolling stats + action distribution)
        if self.num_timesteps % self.log_interval == 0:
            self._log_periodic()

        return True

    def _log_periodic(self) -> None:
        """Log rolling statistics and action distribution at each log interval."""
        # Action distribution
        total_actions = self.action_counts.sum()
        if total_actions > 0:
            pcts = self.action_counts / total_actions * 100
        else:
            pcts = np.zeros(len(ACTION_MAP))

        action_str = " ".join(
            f"{ACTION_MAP[i][0].capitalize()}={pcts[i]:.0f}%"
            for i in range(len(ACTION_MAP))
        )

        # Log action distribution to TensorBoard
        for i in range(len(ACTION_MAP)):
            self.logger.record(f"actions/{_ACTION_FIELDS[i]}", pcts[i])

        # Log event progress
        self.logger.record("progress/unique_events", len(self.events_ever_seen))

        # Flush accumulated episode metrics to TensorBoard as interval means
        if self._pending_episodes:
            n = len(self._pending_episodes)
            self.logger.record("episode/reward", sum(e["reward"] for e in self._pending_episodes) / n)
            self.logger.record("episode/tiles_visited", sum(e["tiles"] for e in self._pending_episodes) / n)
            self.logger.record("episode/party_count", sum(e["party_count"] for e in self._pending_episodes) / n)
            self.logger.record("episode/total_levels", sum(e["levels"] for e in self._pending_episodes) / n)
            self.logger.record("episode/length", sum(e["length"] for e in self._pending_episodes) / n)

            # Mean reward components across the interval
            for key in self.component_keys:
                tb_key = key.replace("reward_", "").replace("penalty_", "")
                self.logger.record(
                    f"reward_components/{tb_key}",
                    sum(e["components"][key] for e in self._pending_episodes) / n,
                )
            self._pending_episodes.clear()

        # Rolling stats from episode history
        if self.episode_history:
            rewards_arr = [ep["reward"] for ep in self.episode_history]
            tiles_arr = [ep["tiles"] for ep in self.episode_history]

            mean_reward = float(np.mean(rewards_arr))
            mean_tiles = float(np.mean(tiles_arr))
            max_tiles = max(tiles_arr)
            starter_count = sum(1 for ep in self.episode_history if ep["party_count"] > 0)
            starter_rate = starter_count / len(self.episode_history)

            self.logger.record("rollout/mean_reward", mean_reward)
            self.logger.record("rollout/mean_tiles", mean_tiles)
            self.logger.record("rollout/max_tiles", max_tiles)
            self.logger.record("rollout/starter_pick_rate", starter_rate)

            print(
                f"  Step {self.num_timesteps:>7d}/{TOTAL_TIMESTEPS} | "
                f"Episodes: {self.episode_count:>4d} | "
                f"Roll100 Reward: {mean_reward:>7.1f} | "
                f"Roll100 Tiles: {mean_tiles:>5.1f} (max {max_tiles}) | "
                f"Starter: {starter_rate:.0%} | "
                f"Actions: {action_str}"
            )
        else:
            print(
                f"  Step {self.num_timesteps:>7d}/{TOTAL_TIMESTEPS} | "
                f"Episodes: {self.episode_count:>4d} | "
                f"(no completed episodes yet) | "
                f"Actions: {action_str}"
            )

        self.action_counts[:] = 0

    def _on_episode_end(self, env_idx: int, info: dict) -> None:
        """Handle a single environment's episode completion."""
        self.episode_count += 1
        ep_reward = self.episode_rewards[env_idx]
        tiles = info.get("tiles_visited", 0)
        events = info.get("events_triggered", [])
        party_count = info.get("party_count", 0)
        total_levels = info.get("total_levels", 0)
        ep_length = info.get("current_step", 0)

        # Collect accumulated reward components for this episode
        ep_components = {k: float(self.episode_components[k][env_idx]) for k in self.component_keys}

        # Track events and starter picks
        self.events_ever_seen.update(events)
        if party_count > 0:
            self.starter_picked_count += 1

        # Track best episode
        if ep_reward > self.best_episode_reward:
            self.best_episode_reward = ep_reward

        # Build episode record
        ep_record = {
            "reward": ep_reward,
            "tiles": tiles,
            "levels": total_levels,
            "party_count": party_count,
            "length": ep_length,
            "components": ep_components,
        }
        self.episode_history.append(ep_record)
        self._pending_episodes.append(ep_record)

        # Print per-episode summary
        print(
            f"  Ep {self.episode_count:>4d} (env{env_idx}) | "
            f"Reward: {ep_reward:>7.1f} | "
            f"Tiles: {tiles:>4d} | "
            f"Party: {party_count} | "
            f"TeamLvl: {total_levels} | "
            f"Events: {events}"
        )

        # Reset this env's accumulators
        self.episode_rewards[env_idx] = 0.0
        for k in self.component_keys:
            self.episode_components[k][env_idx] = 0.0


class MetadataCheckpointCallback(CheckpointCallback):
    """Extends CheckpointCallback to write a JSON metadata sidecar alongside each checkpoint."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "ppo_pokemon",
                 reward_logger: RewardLoggerCallback | None = None, **kwargs):
        super().__init__(save_freq=save_freq, save_path=save_path,
                         name_prefix=name_prefix, **kwargs)
        self.reward_logger = reward_logger

    def _on_step(self) -> bool:
        result = super()._on_step()

        # Write metadata sidecar if a checkpoint was just saved
        if self.n_calls % self.save_freq == 0 and self.reward_logger is not None:
            meta_path = os.path.join(
                self.save_path,
                f"checkpoint_meta_{self.num_timesteps}.json",
            )
            rl = self.reward_logger
            history = rl.episode_history

            meta = {
                "timestep": self.num_timesteps,
                "episode_count": rl.episode_count,
                "rolling_mean_reward": float(np.mean([e["reward"] for e in history])) if history else None,
                "rolling_mean_tiles": float(np.mean([e["tiles"] for e in history])) if history else None,
                "best_episode_reward": rl.best_episode_reward,
                "events_ever_seen": sorted(rl.events_ever_seen),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        return result


def main():
    
    rom_path = os.path.join(SCRIPT_DIR, "..", "pokemon_red.gb")
    state_path = os.path.join(SCRIPT_DIR, "..", "assets", "savestates", "pallet_town.state")

    # Verify files exist
    if not os.path.exists(rom_path):
        print(f"ERROR: ROM not found at {rom_path}")
        return
    if not os.path.exists(state_path):
        print(f"ERROR: Save state not found at {state_path}")
        return

    # Parallel Environments
    # Total env count is always NUM_ENVS. If VISIBLE_ENV is enabled, slot 0
    # gets an SDL2 window; the remaining slots are headless
    if VISIBLE_ENV:
        env_fns = [
            make_env(rom_path, state_path, MAX_STEPS_PER_EP,
                     WATCH_ENV_SPEED, window_type="SDL2"),
        ] + [
            make_env(rom_path, state_path, MAX_STEPS_PER_EP, EMULATION_SPEED)
            for _ in range(1, NUM_ENVS)
        ]
    else:
        env_fns = [
            make_env(rom_path, state_path, MAX_STEPS_PER_EP, EMULATION_SPEED)
            for _ in range(NUM_ENVS)
        ]
    env = SubprocVecEnv(env_fns)

    # Output directories
    checkpoint_dir = os.path.join(SCRIPT_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(TB_LOG_DIR, exist_ok=True)

    # PPO Agent
    if RESUME_FROM is not None:
        resume_path = os.path.join(SCRIPT_DIR, RESUME_FROM)
        print(f"Resuming from checkpoint: {resume_path}")
        model = PPO.load(resume_path, env=env)
        model.tensorboard_log = TB_LOG_DIR
    else:
        model = PPO(
            "MlpPolicy",
            env,
            device="cpu",
            verbose=0,
            n_steps=MAX_STEPS_PER_EP,        # Full-episode rollouts per env before update
            batch_size=512,                 # Minibatch size for PPO updates
            n_epochs=3,                     # PPO epochs per update
            learning_rate=2.5e-4,
            gamma=0.997,                    # High discount for long horizon task
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.02,                  # Entropy bonus
            tensorboard_log=TB_LOG_DIR,
        )

    # ── Train ────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"Starting PPO training:")
    print(f"  Total timesteps:  {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs:    {env.num_envs}")
    print(f"  Episode length:   {MAX_STEPS_PER_EP} steps")
    print(f"  Rollout (n_steps): {model.n_steps} per env ({model.n_steps * env.num_envs} total)")
    print(f"  Gamma:            {model.gamma}")
    print(f"  Emulation speed:  {'unlimited' if EMULATION_SPEED == 0 else f'{EMULATION_SPEED}x'}"
          + (f" (env 0 visible at {WATCH_ENV_SPEED}x)" if VISIBLE_ENV else ""))
    print(f"  Checkpoint dir:   {checkpoint_dir}")
    print(f"  TensorBoard dir:  {TB_LOG_DIR}")
    print(f"  Resume from:      {RESUME_FROM or 'N/A (fresh start)'}")
    print("=" * 70)

    train_start = time.time()

    reward_callback = RewardLoggerCallback(
        log_interval=LOG_INTERVAL, num_envs=env.num_envs,
    )
    
    # Saves once per full rollout across all vec envs
    checkpoint_callback = MetadataCheckpointCallback(
        save_freq=MAX_STEPS_PER_EP,
        save_path=checkpoint_dir,
        name_prefix="ppo_pokemon",
        reward_logger=reward_callback,
    )
    callback = CallbackList([checkpoint_callback, reward_callback])

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    # ── Save ─────────────────────────────────────────────────────────────
    save_path = os.path.join(SCRIPT_DIR, "ppo_pokemon_phase0")
    model.save(save_path)
    print(f"\n{'=' * 70}")
    print(f"Model saved to: {save_path}")
    print(f"Final stats:")
    print(f"  Episodes completed: {reward_callback.episode_count}")
    print(f"  Best episode reward: {reward_callback.best_episode_reward:.1f}")
    if reward_callback.episode_history:
        hist = reward_callback.episode_history
        print(f"  Rolling100 mean reward: {np.mean([e['reward'] for e in hist]):.1f}")
        print(f"  Rolling100 mean tiles:  {np.mean([e['tiles'] for e in hist]):.1f}")
    print(f"  Starter picked: {reward_callback.starter_picked_count}/{reward_callback.episode_count} episodes")
    print(f"  Events ever triggered: {sorted(reward_callback.events_ever_seen)}")

    elapsed = int(time.time() - train_start)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    print(f"  Total training time: {h}h {m}m {s}s")
    print(f"  TensorBoard logs: {TB_LOG_DIR}")
    print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    main()
