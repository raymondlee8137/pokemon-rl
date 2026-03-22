# ── RAM Addresses (Pokémon Red) ──────────────────────────────────────────────
PLAYER_X_ADDR    = 0xD362
PLAYER_Y_ADDR    = 0xD361
MAP_ID_ADDR      = 0xD35E
BATTLE_FLAG_ADDR = 0xD057
PARTY_COUNT_ADDR = 0xD163

# Party Pokémon 1 data
PARTY1_HP_ADDR    = 0xD16C   # 2 bytes, current HP (big-endian)
PARTY1_LEVEL_ADDR = 0xD18C
PARTY1_MAXHP_ADDR = 0xD18D   # 2 bytes, max HP (big-endian)
PARTY_MON_STRIDE  = 0x2C     # Bytes between party Pokémon data slots

# Badge bitmask (bit 0 = Boulder Badge)
BADGES_ADDR = 0xD356

# ── Event Flags (wEventFlags base = 0xD747) ─────────────────────────────────
# Derived from pret/pokered event_constants.asm
# Each flag is a single bit: address = 0xD747 + (bit_index // 8),
#                             bit position = bit_index % 8
#
# Starter sequence (in order):
#   1. Player walks into Route 1 grass → Oak appears and stops you
#   2. EVENT_FOLLOWED_OAK_INTO_LAB  (bit 0)  → 0xD747 bit 0
#   3. EVENT_FOLLOWED_OAK_INTO_LAB_2 (bit 32) → 0xD74B bit 0
#   4. EVENT_OAK_ASKED_TO_CHOOSE_MON (bit 33) → 0xD74B bit 1
#   5. EVENT_GOT_STARTER             (bit 34) → 0xD74B bit 2
#   6. EVENT_BATTLED_RIVAL_IN_OAKS_LAB (bit 35) → 0xD74B bit 3
#   7. EVENT_GOT_POKEDEX             (bit 37) → 0xD74B bit 5
EVENT_FLAGS = {
    "followed_oak":      (0xD747, 0),  # EVENT_FOLLOWED_OAK_INTO_LAB
    # "followed_oak_2":    (0xD74B, 0),  # EVENT_FOLLOWED_OAK_INTO_LAB_2
    # "oak_asked_choose":  (0xD74B, 1),  # EVENT_OAK_ASKED_TO_CHOOSE_MON
    "got_starter":       (0xD74B, 2),  # EVENT_GOT_STARTER
    # "battled_rival":     (0xD74B, 3),  # EVENT_BATTLED_RIVAL_IN_OAKS_LAB
    "got_pokedex":       (0xD74B, 5),  # EVENT_GOT_POKEDEX
}

# ── Button Timing ────────────────────────────────────────────────────────────
FRAMES_PER_STEP    = 24
DPAD_HOLD_FRAMES   = 8   # Directional: hold long enough to commit movement
ACTION_HOLD_FRAMES = 2   # A/B/Start/Select: quick tap is sufficient

# ── Reward Values ────────────────────────────────────────────────────────────
REWARD_NEW_TILE      = 0.1
REWARD_LEVEL_GAINED  = 1.0
REWARD_STEP_PENALTY  = -0.001

# Event flag rewards — breadcrumbs toward the starter
# These form a progression: each step in the sequence is worth more
EVENT_REWARDS = {
    "followed_oak":      5.0,   # Oak stops you on Route 1
    # "followed_oak_2":    15.0,   # You followed Oak back to the lab
    # "oak_asked_choose":  25.0,   # Oak asks you to choose a Pokémon
    "got_starter":      20.0,   # You picked a starter!
    "battled_rival":    5.0,   # Beat the rival battle
    "got_pokedex":       100.0,   # Got the Pokédex
}

# ── Action Mapping ───────────────────────────────────────────────────────────
# Index → (button_name, is_directional)
ACTION_MAP = {
    0: ("up",     True),
    1: ("down",   True),
    2: ("left",   True),
    3: ("right",  True),
    4: ("a",      False),
    5: ("b",      False),
    6: ("start",  False),
    7: ("select", False),
}
