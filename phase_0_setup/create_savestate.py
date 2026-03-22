################################################################################
#
#
#   create_savestate.py creates new save states for testing
#
#
################################################################################

import os
import threading
from pyboy import PyBoy

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROM_PATH = os.path.join(SCRIPT_DIR, "..", "pokemon_red.gb")
SAVE_DIR = os.path.join(SCRIPT_DIR, "..", "assets", "savestates")
SAVE_PATH = os.path.join(SAVE_DIR, "pallet_town.state")

pyboy = PyBoy(ROM_PATH, window="SDL2")

print("Play through the title screen and intro.")
print("Once you're standing in Pallet Town, press Enter in this terminal.")

save_requested = threading.Event()

def wait_for_enter():
    input("Press Enter to save state...")
    save_requested.set()

t = threading.Thread(target=wait_for_enter, daemon=True)
t.start()

while not save_requested.is_set():
    pyboy.tick()

os.makedirs(SAVE_DIR, exist_ok=True)
with open(SAVE_PATH, "wb") as f:
    pyboy.save_state(f)

print(f"Save state created: {SAVE_PATH}")
pyboy.stop()