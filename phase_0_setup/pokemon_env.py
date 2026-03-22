################################################################################
#
#
#   pokemon_env.py is the Gymnasium environment wrapper for Pokémon Red
#
#
################################################################################

from dataclasses import dataclass

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pyboy import PyBoy
from constants import (
    PLAYER_X_ADDR, PLAYER_Y_ADDR, MAP_ID_ADDR, BATTLE_FLAG_ADDR,
    PARTY_COUNT_ADDR, PARTY1_HP_ADDR, PARTY1_LEVEL_ADDR, PARTY1_MAXHP_ADDR,
    PARTY_MON_STRIDE,
    FRAMES_PER_STEP, DPAD_HOLD_FRAMES, ACTION_HOLD_FRAMES,
    REWARD_NEW_TILE, REWARD_LEVEL_GAINED, REWARD_STEP_PENALTY,
    ACTION_MAP, EVENT_FLAGS, EVENT_REWARDS,
)

@dataclass
class GameState:
    player_x:      int
    player_y:      int
    map_id:        int
    battle_flag:   int
    party_count:   int
    party1_hp:     int
    party1_max_hp: int
    party1_level:  int
    total_levels:  int
    event_flags:   dict   # {name: bool}

    @property
    def tile_key(self) -> tuple:
        return (self.map_id, self.player_x, self.player_y)

# Observation Schema
# Each entry: (field_name, min, max)
# To add a new observation field: add a row here and a value in _get_observation()

_OBS_SCHEMA = [
    ("player_x",      0,     255),
    ("player_y",      0,     255),
    ("map_id",        0,     255),
    ("battle_flag",   0,     1),
    ("party_count",   0,     6),
    ("party1_hp",     0,     65535),
    ("party1_max_hp", 0,     65535),
    ("party1_level",  0,     100),
] + [(name, 0, 1) for name in EVENT_FLAGS]

class PokemonRedEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, rom_path: str, state_path: str, max_steps: int = 2000,
                 emulation_speed: int = 1, window_type: str = "SDL2") -> None:
        super().__init__()

        self.rom_path = rom_path
        self.state_path = state_path
        self.max_steps = max_steps
        self.emulation_speed = emulation_speed  # 0 = unlimited, 1 = real-time, N = Nx
        self.window_type = window_type          # "SDL2" for visual, "null" for headless

        self.action_space = spaces.Discrete(8)

        # Observation: 8 base values + one entry per active event flag
        self.observation_space = spaces.Box(
            low=np.array( [lo for _, lo, _ in _OBS_SCHEMA], dtype=np.float32),
            high=np.array([hi for _, _, hi in _OBS_SCHEMA], dtype=np.float32),
            dtype=np.float32,
        )

        # Emulator
        self.pyboy = None

        # Tracking state
        self.visited_tiles = set()
        self.current_step = 0
        self.prev_state = None         # GameState from the previous step
        self.triggered_events = set()  # Event flags already rewarded this episode
        self._last_reward_components: dict[str, float] = {
            "reward_exploration": 0.0,
            "reward_event": 0.0,
            "reward_level": 0.0,
            "penalty_step": 0.0,
        }

    def reset(self, seed=None, options=None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        # Create emulator only on first call
        if self.pyboy is None:
            self.pyboy = PyBoy(self.rom_path, window=self.window_type)
            self.pyboy.set_emulation_speed(self.emulation_speed)

        # Load save state
        with open(self.state_path, "rb") as f:
            self.pyboy.load_state(f)

        # Tick a few frames to let the game settle after state load
        for _ in range(10):
            self.pyboy.tick()

        # Reset tracking
        self.visited_tiles = set()
        self.current_step = 0
        self.triggered_events = set()
        self._last_reward_components = {
            "reward_exploration": 0.0,
            "reward_event": 0.0,
            "reward_level": 0.0,
            "penalty_step": 0.0,
        }

        curr_state = self._read_game_state()

        # Snapshot which event flags are already set in the save state
        # so we don't reward pre-existing flags
        for name, is_set in curr_state.event_flags.items():
            if is_set:
                self.triggered_events.add(name)

        # Record starting tile and set initial prev_state
        self.visited_tiles.add(curr_state.tile_key)
        self.prev_state = curr_state

        obs = self._get_observation(curr_state)
        info = self._get_info(curr_state)
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        button_name, is_directional = ACTION_MAP[action]
        hold_frames = DPAD_HOLD_FRAMES if is_directional else ACTION_HOLD_FRAMES
        remaining_frames = FRAMES_PER_STEP - hold_frames

        # Hold button for the appropriate number of frames
        self.pyboy.button_press(button_name)
        for _ in range(hold_frames):
            self.pyboy.tick()
        self.pyboy.button_release(button_name)

        # Tick remaining frames to complete the step
        for _ in range(remaining_frames):
            self.pyboy.tick()

        self.current_step += 1

        # Compute reward
        curr_state = self._read_game_state()
        reward = self._compute_reward(self.prev_state, curr_state)
        self.prev_state = curr_state

        # Check termination
        terminated = False
        truncated = self.current_step >= self.max_steps

        obs = self._get_observation(curr_state)
        info = self._get_info(curr_state)

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self.pyboy is not None:
            self.pyboy.stop()
            self.pyboy = None

    # RAM Reading Helpers 

    # Util function to read single byte from memory
    def _read_memory(self, address: int) -> int:
        return self.pyboy.memory[address]

    # Util function to read 16-bit big-endian value from memory
    def _read_memory_16(self, address: int) -> int:
        high = self.pyboy.memory[address]
        low = self.pyboy.memory[address + 1]
        return (high << 8) | low

    # Util function to read single bit event flags
    def _read_event_flag(self, flag_name: str) -> bool:
        addr, bit = EVENT_FLAGS[flag_name]
        return bool((self._read_memory(addr) >> bit) & 1)

    # Read all RAM-derived states at once and return snapshot
    def _read_game_state(self) -> GameState:
        
        player_x    = self._read_memory(PLAYER_X_ADDR)
        player_y    = self._read_memory(PLAYER_Y_ADDR)
        map_id      = self._read_memory(MAP_ID_ADDR)
        battle_flag = self._read_memory(BATTLE_FLAG_ADDR)
        party_count = self._read_memory(PARTY_COUNT_ADDR)

        # Party Pokémon 1 stats (default to 0 if no Pokémon yet)
        if party_count > 0:
            party1_hp     = self._read_memory_16(PARTY1_HP_ADDR)
            party1_max_hp = self._read_memory_16(PARTY1_MAXHP_ADDR)
            party1_level  = self._read_memory(PARTY1_LEVEL_ADDR)
        else:
            party1_hp = party1_max_hp = party1_level = 0

        # Party Pokémon data is spaced PARTY_MON_STRIDE bytes apart
        total_levels = sum(
            self._read_memory(PARTY1_LEVEL_ADDR + i * PARTY_MON_STRIDE)
            for i in range(min(party_count, 6))
        )

        event_flags = {name: self._read_event_flag(name) for name in EVENT_FLAGS}

        return GameState(
            player_x=player_x, player_y=player_y, map_id=map_id,
            battle_flag=battle_flag, party_count=party_count,
            party1_hp=party1_hp, party1_max_hp=party1_max_hp,
            party1_level=party1_level, total_levels=total_levels,
            event_flags=event_flags,
        )

    # Reward Computation Helpers

    # Sum of all rewards
    def _compute_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        exploration = self._exploration_reward(prev_state, curr_state)
        event = self._event_reward(curr_state)
        level = self._level_reward(prev_state, curr_state)
        step_pen = self._step_penalty()
        self._last_reward_components = {
            "reward_exploration": exploration,
            "reward_event": event,
            "reward_level": level,
            "penalty_step": step_pen,
        }
        return exploration + event + level + step_pen

    def _exploration_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        if (curr_state.battle_flag == 0
                and curr_state.tile_key != prev_state.tile_key
                and curr_state.tile_key not in self.visited_tiles):
            self.visited_tiles.add(curr_state.tile_key)
            return REWARD_NEW_TILE
        return 0.0

    def _event_reward(self, curr_state: GameState) -> float:
        reward = 0.0
        for name, reward_value in EVENT_REWARDS.items():
            if name not in EVENT_FLAGS:
                continue
            if name not in self.triggered_events and curr_state.event_flags[name]:
                self.triggered_events.add(name)
                reward += reward_value
        return reward

    def _level_reward(self, prev_state: GameState, curr_state: GameState) -> float:
        levels_gained = curr_state.total_levels - prev_state.total_levels
        if levels_gained > 0:
            return REWARD_LEVEL_GAINED * levels_gained
        return 0.0

    def _step_penalty(self) -> float:
        return REWARD_STEP_PENALTY

    # Observation and Info

    def _get_observation(self, state: GameState) -> np.ndarray:
        base = [
            state.player_x, state.player_y, state.map_id, min(state.battle_flag, 1),
            state.party_count, state.party1_hp, state.party1_max_hp, state.party1_level,
        ]
        # Event flags as binary observations
        flag_values = [float(state.event_flags[name]) for name in EVENT_FLAGS]
        return np.array(base + flag_values, dtype=np.float32)

    def _get_info(self, state: GameState) -> dict:
        return {
            "tiles_visited": len(self.visited_tiles),
            "current_step": self.current_step,
            "map_id": state.map_id,
            "player_x": state.player_x,
            "player_y": state.player_y,
            "party_count": state.party_count,
            "total_levels": state.total_levels,
            "events_triggered": sorted(self.triggered_events),
            **self._last_reward_components,
        }

# Run python phase_0_setup/pokemon_env.py to make sure env is functioning
if __name__ == "__main__":
    import os

    TEST_STEPS = 100

    rom_path = os.path.join(os.path.dirname(__file__), "..", "pokemon_red.gb")
    state_path = os.path.join(os.path.dirname(__file__), "..", "assets", "savestates", "pallet_town.state")

    env = PokemonRedEnv(rom_path=rom_path, state_path=state_path, max_steps=TEST_STEPS)

    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    # Run random actions so to watch it in the SDL2 window
    total_reward = 0
    for step in range(TEST_STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        # Print cumulative total reward every 10 steps
        if (step + 1) % 10 == 0:
            print(f"Step {step + 1:>4d} | Total Reward: {total_reward:>6.1f} | "
                  f"Tiles: {info['tiles_visited']:>3d} | "
                  f"Map: {info['map_id']} | Pos: ({info['player_x']}, {info['player_y']}) | "
                  f"Events: {info['events_triggered']}")

        if terminated or truncated:
            break

    print(f"\nDone! Total reward: {total_reward:.1f}, Tiles visited: {info['tiles_visited']}")
    print(f"Events triggered: {info['events_triggered']}")
    env.close()
