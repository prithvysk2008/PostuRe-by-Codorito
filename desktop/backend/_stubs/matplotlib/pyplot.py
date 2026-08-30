"""Stub for mediapipe.python.solutions.drawing_utils's unused plot_landmarks()
helper — see the package __init__ for the full explanation. If anything ever
actually calls into this, fail loudly and clearly rather than silently
no-op, so a real future need for matplotlib doesn't go unnoticed.
"""


def __getattr__(name):
    raise RuntimeError(
        f"matplotlib.pyplot.{name} was called, but this build substitutes a "
        "stub matplotlib (see build_pyinstaller.sh) on the assumption that "
        "nothing in this app's actual code path needs it. If you're seeing "
        "this, something now calls mediapipe's plot_landmarks() (or another "
        "matplotlib-dependent path) — remove the --paths override in "
        "build_pyinstaller.sh to bundle the real matplotlib instead."
    )
