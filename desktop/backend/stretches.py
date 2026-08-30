"""Stretch-break catalogue and picker — ported verbatim from posture_app.py."""
import random
from typing import List, Tuple

# Each entry: (name, target group, one-sentence instruction, icon key).
# Grouped by what a desk/typing session actually strains.
STRETCHES = [
    # -- neck --
    ("Chin tucks", "neck",
     "Pull your chin straight back like you're making a double chin, hold, then release.", "chin_tuck"),
    ("Neck side bend", "neck",
     "Tilt one ear toward its shoulder until you feel a gentle stretch, then switch sides.", "neck_tilt"),
    ("Neck rotation", "neck",
     "Slowly turn your head to look over one shoulder, then the other.", "neck_rotate"),
    ("Chin-to-chest stretch", "neck",
     "Lower your chin toward your chest and feel the stretch down the back of your neck.", "chin_chest"),
    # -- shoulders / upper back --
    ("Shoulder rolls", "shoulders",
     "Roll both shoulders backward in big, slow circles.", "shoulder_roll"),
    ("Shoulder blade squeeze", "shoulders",
     "Pull your shoulder blades together like you're pinching a pencil between them.", "blade_squeeze"),
    ("Cross-body shoulder stretch", "shoulders",
     "Pull one arm across your chest with the other hand, then switch sides.", "cross_arm"),
    ("Upper back stretch", "shoulders",
     "Clasp your hands, round your upper back, and push your hands away from you.", "back_round"),
    ("Seated cat-cow", "shoulders",
     "Arch and round your upper spine slowly while seated, following your breath.", "cat_cow"),
    # -- eyes --
    ("Look far away", "eyes",
     "Focus on the furthest thing you can see and let your eyes reset.", "eye_far"),
    ("20-20-20 blink reset", "eyes",
     "Close your eyes gently for a count of five, then blink slowly ten times.", "eye_blink"),
    ("Eye circles", "eyes",
     "Without moving your head, roll your eyes slowly in a full circle, then reverse.", "eye_circle"),
    # -- wrists --
    ("Wrist flex & extend", "wrists",
     "Straighten one arm and gently pull your fingers back, then push them down.", "wrist_flex"),
    ("Wrist circles", "wrists",
     "Rotate both wrists in slow circles, then reverse direction.", "wrist_circle"),
    ("Finger spread & fist", "wrists",
     "Spread your fingers as wide as you can, then close into a soft fist. Repeat.", "finger_fist"),
]

# Every guided break is a fixed 30s: 3 exercises, 10s each, drawn at random
# from the STRETCHES catalogue.
BREAK_DURATION_S = 30.0
BREAK_EXERCISES_N = 3


def pick_stretches() -> List[Tuple[str, str, str, str]]:
    """Random 3-exercise subset of the STRETCHES catalogue for one break."""
    return random.sample(STRETCHES, min(BREAK_EXERCISES_N, len(STRETCHES)))
