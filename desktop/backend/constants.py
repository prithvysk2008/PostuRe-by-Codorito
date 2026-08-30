"""
Design tokens and app-wide constants — ported verbatim from posture_app.py.
No Streamlit dependency; safe to import from anywhere.
"""
import os
import sys

BG_0 = "#0B1E33"      # --ink
BG_1 = "#0E2743"
SURFACE = "#12314F"   # --ink-2
SURFACE_2 = "#173A5C"
LINE = "#1F3F5E"
TEXT = "#EAF3F2"       # --paper
MUTED = "#7C97AA"      # --graphite
ACCENT = "#4FD8C4"     # --drafting

C_GOOD = "#35E6A6"
C_WATCH = "#FF8A3D"    # --hazard
C_BAD = "#FF4757"      # --critical
C_IDLE = "#4FD8C4"

STATUS_COLORS = {"GOOD": C_GOOD, "WATCH": C_WATCH, "BAD": C_BAD, "IDLE": C_IDLE}
STATUS_TEXT = {
    "GOOD": "ALIGNED",
    "WATCH": "DRIFTING",
    "BAD": "SLOUCHED",
    "IDLE": "NO SUBJECT",
}

# Thresholds
T_GOOD = 80.0
T_WATCH = 60.0

# Capture
CAP_W, CAP_H = 960, 540
TARGET_FPS = 20.0

# Calibration
CAL_WARMUP_S = 2.0
CAL_CAPTURE_S = 5.0
CAL_MIN_SAMPLES = 22

def _db_file() -> str:
    """Where posture_data.db (SQLite) lives.

    - Running from source (dev, or `streamlit run` on the original app):
      this file is desktop/backend/constants.py, so three levels up is the
      repo root — the SAME posture_data.db the Streamlit app reads/writes,
      so session history isn't silently forked between the two frontends.
    - Running as a PyInstaller-frozen executable (a distributed .app has no
      "repo root" on the end user's machine — sys.frozen is PyInstaller's
      own flag for this): use the standard per-user app-support directory
      instead, which is the only sensible writable, persistent location for
      a real installed app.
    """
    if getattr(sys, "frozen", False):
        base = os.path.join(os.path.expanduser("~"), "Library", "Application Support", "PostuRe")
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, "posture_data.db")
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(repo_root, "posture_data.db")


DB_FILE = _db_file()
