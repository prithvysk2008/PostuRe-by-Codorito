"""
PostuRe:  —  Posture & Fatigue Detection
=========================================
A 100% on-device, offline posture + fatigue coach built on MediaPipe Pose,
MediaPipe Face Mesh, OpenCV, NumPy and Streamlit.

No cloud. No API keys. No database. Nothing leaves the laptop.

Run with:   streamlit run posture_app.py
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import time
import wave as wave_lib
from collections import deque
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import streamlit as st

# ----------------------------------------------------------------------------
# Page config must be the first Streamlit call.
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="PostuRe:",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Heavy imports guarded so a broken install shows a fixable message, not a
# 40-line traceback in front of judges.
# ----------------------------------------------------------------------------
IMPORT_ERROR = None
try:
    import cv2
    import mediapipe as mp
except Exception as exc:  # pragma: no cover
    IMPORT_ERROR = exc
    cv2 = None
    mp = None


# ============================================================================
# DESIGN TOKENS
# Direction: "clinical telemetry" — a spine monitor, not a wellness widget.
# Deep slate blues, one periwinkle brand accent kept away from the status
# palette so colour always means exactly one thing: how your spine is doing.
# ============================================================================
BG_0 = "#080B12"
BG_1 = "#0D1320"
SURFACE = "#121A29"
SURFACE_2 = "#182234"
LINE = "#233248"
TEXT = "#E9EFF9"
MUTED = "#7E8FA8"
ACCENT = "#6E8BFF"

C_GOOD = "#35E6A6"
C_WATCH = "#FFB547"
C_BAD = "#FF5C7A"
C_IDLE = "#6E8BFF"

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

DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posture_data.json")


# ============================================================================
# METRIC SPECIFICATION
# Every metric is scale-invariant (normalised by shoulder width or frame
# height) and scored as a *deviation from your own calibrated baseline*.
# ============================================================================
class MetricSpec:
    __slots__ = ("weight", "direction", "tol", "span", "tip")

    def __init__(self, weight, direction, tol, span, tip):
        self.weight = weight        # contribution to the 0-100 score
        self.direction = direction  # "dec" = falling is bad, "inc" = rising is bad, "abs" = either
        self.tol = tol              # minimum dead-zone (natural fidgeting)
        self.span = span            # deviation beyond tolerance that costs full weight
        self.tip = tip              # coaching line shown when this metric dominates


METRICS: Dict[str, MetricSpec] = {
    # Vertical gap between ear-line and shoulder-line, in shoulder widths.
    "neck": MetricSpec(0.28, "dec", 0.020, 0.115, "Lift the crown of your head — your neck is collapsing."),
    # 2D craniovertebral angle (ear→shoulder vs horizontal).
    "cva": MetricSpec(0.18, "dec", 2.0, 14.0, "Craniovertebral angle dropping — tuck your chin back."),
    # Nose below the ear-line = looking down at the keyboard.
    "pitch": MetricSpec(0.18, "inc", 0.025, 0.130, "Head tilted down — raise your screen to eye level."),
    # Shoulders sinking in the frame = sliding down the chair.
    "drop": MetricSpec(0.14, "inc", 0.012, 0.070, "You're sinking into the chair — sit back into the backrest."),
    # Face appears larger relative to shoulders = head craning toward screen.
    "face": MetricSpec(0.14, "inc", 0.012, 0.070, "You're creeping toward the screen — push your chair in instead."),
    # Shoulder line rotated = leaning on one arm.
    "tilt": MetricSpec(0.08, "abs", 2.5, 12.0, "One shoulder is dropping — even out your weight."),
}
METRIC_KEYS = list(METRICS.keys())

STRETCHES = [
    ("Chin tucks", "Pull your chin straight back, hold 2s, release. Repeat slowly.", 10),
    ("Shoulder rolls", "Roll both shoulders backward in big, slow circles.", 10),
    ("Look far away", "Focus on the furthest thing you can see. Let your eyes reset.", 10),
]


# ============================================================================
# SMALL HELPERS
# ============================================================================
def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def fmt_clock(seconds: float) -> str:
    seconds = int(max(0, seconds))
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def status_for(score: float) -> str:
    if score >= T_GOOD:
        return "GOOD"
    if score >= T_WATCH:
        return "WATCH"
    return "BAD"


# ============================================================================
# STYLESHEET
# System font stack only — no CDN fonts, because the offline claim is real.
# Numerals are monospace with tabular figures: this is an instrument panel.
# ============================================================================
CSS = """
<style>
:root{
  --bg0:#080B12; --bg1:#0D1320; --surface:#121A29; --surface2:#182234;
  --line:#233248; --text:#E9EFF9; --muted:#7E8FA8; --accent:#6E8BFF;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, "Cascadia Mono", "Consolas", "Roboto Mono", monospace;
}
html, body, [data-testid="stAppViewContainer"]{
  background: radial-gradient(1200px 700px at 20% -10%, #14203a 0%, var(--bg1) 45%, var(--bg0) 100%);
  color: var(--text);
  font-family: var(--sans);
}
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stToolbar"]{ right: 8px; }
.block-container{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer{ visibility: hidden; }

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0E1626 0%, #0A0F1A 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .stMarkdown p{ color: var(--muted); font-size: 0.86rem; }

/* ---------- masthead ---------- */
.pr-mast{ display:flex; align-items:flex-end; gap:18px; margin-bottom:6px; flex-wrap:wrap; }
.pr-wordmark{
  font-family: var(--mono); font-size: 2.5rem; font-weight: 700;
  letter-spacing:-0.03em; line-height:1; color: var(--text); margin:0;
}
.pr-wordmark span{ color: var(--accent); }
.pr-tag{
  font-family: var(--mono); font-size:.68rem; letter-spacing:.22em; text-transform:uppercase;
  color: var(--muted); padding-bottom:.35rem;
}
.pr-rule{ height:1px; background:linear-gradient(90deg,var(--line),transparent); margin:10px 0 20px; }

/* ---------- ambient screen-edge alert ---------- */
.pr-ambient{
  position: fixed; inset: 0; pointer-events: none; z-index: 9998;
  transition: box-shadow .7s cubic-bezier(.4,0,.2,1), border-color .7s ease;
}

/* ---------- video stage ---------- */
.pr-stage{
  position: relative; border-radius: 20px; overflow: hidden;
  border: 1px solid var(--line); background: #05070C;
  transition: box-shadow .45s ease, border-color .45s ease;
}
.pr-stage img{ display:block; width:100%; height:auto; }
.pr-badge{
  position:absolute; top:14px; left:14px; padding:6px 14px; border-radius:999px;
  font-family: var(--mono); font-size:.68rem; letter-spacing:.18em; font-weight:600;
  border:1px solid; backdrop-filter: blur(8px);
}
.pr-fps{
  position:absolute; bottom:12px; right:14px; font-family:var(--mono); font-size:.62rem;
  letter-spacing:.12em; color:rgba(233,239,249,.45); background:rgba(5,7,12,.45);
  padding:3px 9px; border-radius:6px;
}

/* ---------- metric cards ---------- */
.pr-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.pr-card{
  background: linear-gradient(160deg, var(--surface2) 0%, var(--surface) 100%);
  border:1px solid var(--line); border-radius:16px; padding:14px 16px;
  box-shadow: 0 10px 26px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.03);
}
.pr-card.wide{ grid-column: span 2; }
.pr-label{
  font-family:var(--mono); font-size:.6rem; letter-spacing:.2em; text-transform:uppercase;
  color:var(--muted); margin-bottom:6px;
}
.pr-value{
  font-family:var(--mono); font-size:2.0rem; font-weight:700; line-height:1.05;
  font-variant-numeric: tabular-nums; letter-spacing:-0.02em;
}
.pr-value .u{ font-size:.85rem; font-weight:500; color:var(--muted); margin-left:4px; letter-spacing:0; }
.pr-sub{ font-size:.74rem; color:var(--muted); margin-top:5px; line-height:1.35; }

.pr-bar{ height:5px; border-radius:99px; background:#0A1120; margin-top:10px; overflow:hidden; }
.pr-bar i{ display:block; height:100%; border-radius:99px; transition:width .35s ease; }

/* ---------- telemetry strip (the signature element) ---------- */
.pr-strip{
  background:linear-gradient(160deg,var(--surface2),var(--surface));
  border:1px solid var(--line); border-radius:16px; padding:14px 16px 8px;
  box-shadow:0 10px 26px rgba(0,0,0,.42);
}
.pr-strip svg{ display:block; width:100%; height:88px; }
.pr-legend{
  display:flex; gap:16px; font-family:var(--mono); font-size:.58rem; letter-spacing:.14em;
  color:var(--muted); text-transform:uppercase; margin-top:4px; flex-wrap:wrap;
}
.pr-legend b{ color:var(--text); font-weight:600; }

/* ---------- coaching / alert banner ---------- */
.pr-banner{
  border-radius:14px; padding:13px 18px; border:1px solid; display:flex; gap:12px;
  align-items:center; font-size:.92rem; margin-bottom:12px;
}
.pr-banner .k{
  font-family:var(--mono); font-size:.6rem; letter-spacing:.2em; text-transform:uppercase;
  padding:3px 9px; border-radius:6px; white-space:nowrap;
}

/* ---------- stretch break overlay ---------- */
.pr-break{
  border-radius:20px; padding:26px 28px; text-align:center;
  background:linear-gradient(160deg,#152B45,#0E1B2C);
  border:1px solid #2B4568;
  box-shadow:0 18px 50px rgba(0,0,0,.55);
}
.pr-break .n{ font-family:var(--mono); font-size:3.4rem; font-weight:700; color:#6E8BFF; line-height:1; }
.pr-break h3{ margin:10px 0 4px; font-size:1.25rem; color:var(--text); }
.pr-break p{ color:var(--muted); font-size:.92rem; margin:0; }

/* ---------- snapshot ---------- */
.pr-snap{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.pr-snap figure{ margin:0; }
.pr-snap img{ width:100%; border-radius:14px; border:1px solid var(--line); display:block; }
.pr-snap figcaption{
  font-family:var(--mono); font-size:.6rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); margin-top:8px;
}

/* ---------- summary ---------- */
.pr-hero{
  background:linear-gradient(150deg,#16233A 0%,#0C131F 100%);
  border:1px solid var(--line); border-radius:24px; padding:34px 32px;
  box-shadow:0 20px 60px rgba(0,0,0,.5); text-align:center;
}
.pr-hero .big{
  font-family:var(--mono); font-size:5.2rem; font-weight:700; line-height:1;
  letter-spacing:-.04em; font-variant-numeric:tabular-nums;
}
.pr-hero .cap{
  font-family:var(--mono); font-size:.66rem; letter-spacing:.24em; text-transform:uppercase; color:var(--muted);
}
.pr-hero h2{ margin:12px 0 2px; font-size:1.5rem; }

/* ---------- buttons ---------- */
.stButton > button{
  width:100%; border-radius:12px; border:1px solid var(--line);
  background:linear-gradient(160deg,var(--surface2),var(--surface)); color:var(--text);
  font-weight:600; padding:.6rem 1rem; transition:all .18s ease;
}
.stButton > button:hover{ border-color:var(--accent); color:#fff; transform:translateY(-1px); }
.stButton > button[kind="primary"]{
  background:linear-gradient(160deg,#6E8BFF,#4A63D8); border-color:#6E8BFF; color:#fff;
}
div[data-testid="stMetricValue"]{ font-family:var(--mono); }
@media (prefers-reduced-motion: reduce){
  .pr-ambient,.pr-stage,.pr-bar i{ transition:none !important; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


# ============================================================================
# OFFLINE AUDIO — chimes are synthesised with NumPy at runtime.
# No sound files to ship, nothing to download.
# ============================================================================
@st.cache_data(show_spinner=False)
def make_chime(freqs: Tuple[float, ...], duration: float = 0.75, volume: float = 0.22) -> str:
    """Return a base64 WAV of a soft, exponentially-decaying chord."""
    sr = 22050
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    tone = np.zeros_like(t)
    for i, f in enumerate(freqs):
        delay = i * 0.12
        env = np.exp(-3.2 * np.maximum(t - delay, 0.0)) * (t >= delay)
        tone += np.sin(2 * np.pi * f * (t - delay)) * env
    # gentle attack so it never clicks
    attack = np.minimum(t / 0.03, 1.0)
    tone *= attack
    peak = float(np.max(np.abs(tone))) or 1.0
    audio = np.int16(np.clip(tone / peak * volume, -1, 1) * 32767)

    buf = io.BytesIO()
    with wave_lib.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


CHIMES = {
    "watch": (587.33, 493.88),                 # descending — "you're drifting"
    "bad": (523.25, 440.00, 392.00),           # deeper fall — "you've slumped"
    "predict": (659.25, 783.99),               # rising — "heads up, before it happens"
    "break": (523.25, 659.25, 783.99),         # major triad — "time to move"
    "recover": (783.99, 1046.50),              # bright — "nice save"
}


def play_chime(placeholder, kind: str, token: int) -> None:
    b64 = make_chime(CHIMES[kind])
    placeholder.markdown(
        f'<audio autoplay="true" data-k="{kind}-{token}">'
        f'<source src="data:audio/wav;base64,{b64}" type="audio/wav"></audio>',
        unsafe_allow_html=True,
    )


# ============================================================================
# LOCAL PERSISTENCE — a single JSON file next to the script. Still offline.
# ============================================================================
def load_store() -> dict:
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"last_date": None, "daily_streak": 0, "best_streak": 0, "sessions": []}


def save_store(store: dict) -> None:
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(store, f, indent=2)
    except Exception:
        pass


def register_day(store: dict) -> dict:
    today = date.today().isoformat()
    last = store.get("last_date")
    if last == today:
        return store
    if last:
        try:
            gap = (date.today() - date.fromisoformat(last)).days
        except Exception:
            gap = 99
    else:
        gap = 99
    store["daily_streak"] = store.get("daily_streak", 0) + 1 if gap == 1 else 1
    store["last_date"] = today
    store["best_streak"] = max(store.get("best_streak", 0), store["daily_streak"])
    return store


# ============================================================================
# MODELS + CAMERA (cached so a Streamlit rerun never re-initialises them)
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_pose(model_complexity: int):
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@st.cache_resource(show_spinner=False)
def load_face():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def get_camera(index: int):
    """Keep one VideoCapture alive in session_state across Streamlit reruns."""
    cap = st.session_state.get("cap")
    if cap is not None and cap.isOpened():
        return cap
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():  # retry with the default backend
        cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
    cap.set(cv2.CAP_PROP_FPS, 30)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    st.session_state.cap = cap
    return cap


def release_camera() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None


# ============================================================================
# GEOMETRY — pose landmarks to normalised posture metrics
# ============================================================================
L_NOSE, L_EYE_L, L_EYE_R = 0, 2, 5
L_EAR_L, L_EAR_R = 7, 8
L_SH_L, L_SH_R = 11, 12
L_HIP_L, L_HIP_R = 23, 24

EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def extract_metrics(lms, w: int, h: int) -> Optional[dict]:
    """Turn 33 pose landmarks into six scale-invariant posture metrics."""
    def P(i):
        return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float64)

    def V(i):
        return lms[i].visibility

    if V(L_SH_L) < 0.4 or V(L_SH_R) < 0.4:
        return None

    sh_l, sh_r = P(L_SH_L), P(L_SH_R)
    shoulder_w = float(np.linalg.norm(sh_l - sh_r))
    if shoulder_w < 40:                      # too far away / bad detection
        return None
    sh_mid = (sh_l + sh_r) / 2.0

    # Head anchor: ears preferred (they track the tragus, the clinical C7 pair),
    # eyes as a fallback when the ears are hidden by hair or headphones.
    ears = []
    if V(L_EAR_L) > 0.3:
        ears.append(P(L_EAR_L))
    if V(L_EAR_R) > 0.3:
        ears.append(P(L_EAR_R))
    head = np.mean(ears, axis=0) if ears else (P(L_EYE_L) + P(L_EYE_R)) / 2.0

    nose = P(L_NOSE)
    eye_dist = float(np.linalg.norm(P(L_EYE_L) - P(L_EYE_R)))

    # Craniovertebral angle, 2D: angle between the shoulder→ear vector and the
    # horizontal. Measured on whichever side the camera sees more clearly.
    if V(L_EAR_L) >= V(L_EAR_R):
        ear_pt, sh_pt = P(L_EAR_L), sh_l
    else:
        ear_pt, sh_pt = P(L_EAR_R), sh_r
    dx = abs(float(ear_pt[0] - sh_pt[0]))
    dy = float(sh_pt[1] - ear_pt[1])
    cva = math.degrees(math.atan2(dy, dx + 1e-6))

    dxs = float(sh_r[0] - sh_l[0])
    dys = float(sh_r[1] - sh_l[1])
    tilt = math.degrees(math.atan2(dys, dxs if abs(dxs) > 1e-6 else 1e-6))
    if tilt > 90:
        tilt -= 180
    elif tilt < -90:
        tilt += 180

    return {
        "neck": float((sh_mid[1] - head[1]) / shoulder_w),
        "cva": float(clamp(cva, 0.0, 120.0)),
        "pitch": float((nose[1] - head[1]) / shoulder_w),
        "drop": float(sh_mid[1] / h),
        "face": float(eye_dist / shoulder_w),
        "tilt": float(tilt),
        # geometry kept for drawing only
        "_sh_l": sh_l, "_sh_r": sh_r, "_sh_mid": sh_mid,
        "_head": head, "_ear": ear_pt, "_ear_sh": sh_pt,
        "_shoulder_w": shoulder_w,
    }


# --- Face Mesh: EAR / MAR -----------------------------------------------------
EYE_R_IDX = [33, 160, 158, 133, 153, 144]
EYE_L_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_V = (13, 14)
MOUTH_H = (78, 308)


def _ear(pts: np.ndarray, idx: List[int]) -> float:
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    horiz = float(np.linalg.norm(p1 - p4))
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


def extract_face(landmarks, w: int, h: int) -> Optional[Tuple[float, float]]:
    pts = np.array([[l.x * w, l.y * h] for l in landmarks], dtype=np.float64)
    if pts.shape[0] < 400:
        return None
    ear = (_ear(pts, EYE_L_IDX) + _ear(pts, EYE_R_IDX)) / 2.0
    mh = float(np.linalg.norm(pts[MOUTH_H[0]] - pts[MOUTH_H[1]]))
    mv = float(np.linalg.norm(pts[MOUTH_V[0]] - pts[MOUTH_V[1]]))
    mar = mv / mh if mh > 1e-6 else 0.0
    return ear, mar


# ============================================================================
# THE ENGINE
# One object, kept in session_state, holding every piece of session state:
# calibration, scoring, streaks, prediction, fatigue and break scheduling.
# ============================================================================
class PostureEngine:
    def __init__(self):
        self.hard_reset()

    # -- lifecycle -----------------------------------------------------------
    def hard_reset(self):
        self.reset_calibration()
        self.session_start = None
        self.last_t = None
        self.paused_s = 0.0

        self.metrics_ema: Optional[dict] = None
        self.raw_score = 100.0
        self.score = 100.0
        self.status = "IDLE"
        self._cand = "IDLE"
        self._cand_n = 0
        self.parts: Dict[str, float] = {k: 0.0 for k in METRIC_KEYS}
        self.tip = ""

        self.history: deque = deque(maxlen=4000)      # (t, score)
        self.score_sum = 0.0
        self.score_n = 0
        self.time_in = {"GOOD": 0.0, "WATCH": 0.0, "BAD": 0.0}

        self.streak_s = 0.0
        self.best_streak_s = 0.0
        self.recoveries = 0
        self.bonus = 0
        self._bad_since: Optional[float] = None
        self._await_recovery_until: Optional[float] = None

        self.trend = 0.0                              # points per minute
        self.eta: Optional[float] = None              # seconds until score hits 60
        self.predicting = False

        # fatigue
        self.ear = 0.0
        self.mar = 0.0
        self.ear_base = 0.27
        self.mar_base = 0.12
        self.eye_closed_since: Optional[float] = None
        self.blinks = 0
        self.blink_times: deque = deque(maxlen=4000)
        self.microsleeps = 0
        self.yawns = 0
        self.yawn_times: deque = deque(maxlen=200)
        self._yawn_since: Optional[float] = None
        self._yawn_flag = False
        self._ms_flag = False
        self.perclos_win: deque = deque(maxlen=2400)  # (t, closed?)
        self.fatigue = 0.0
        self.fatigue_label = "Fresh"
        self.fatigue_sum = 0.0
        self.fatigue_n = 0
        self.blink_rate = 0.0
        self.dry_eye = False

        # breaks
        self.next_break: Optional[float] = None
        self.break_until: Optional[float] = None
        self.breaks_taken = 0

        # frames
        self.baseline_jpg: Optional[str] = None
        self.current_jpg: Optional[str] = None
        self._snap_t = 0.0

        # alerts
        self.last_chime = {"watch": 0.0, "bad": 0.0, "predict": 0.0, "break": 0.0, "recover": 0.0}
        self.audio_token = 0
        self.pending_chime: Optional[str] = None

    def reset_calibration(self):
        self.baseline: Optional[dict] = None
        self.mad: Optional[dict] = None
        self.cal_started: Optional[float] = None
        self.cal_samples: List[dict] = []
        self.cal_ear: List[float] = []
        self.cal_mar: List[float] = []
        self.calibrated = False

    # -- calibration ---------------------------------------------------------
    def cal_phase(self, now: float) -> Tuple[str, float]:
        """Returns (phase, progress 0..1). Phases: warmup, capture."""
        if self.cal_started is None:
            self.cal_started = now
        el = now - self.cal_started
        if el < CAL_WARMUP_S:
            return "warmup", el / CAL_WARMUP_S
        return "capture", clamp((el - CAL_WARMUP_S) / CAL_CAPTURE_S, 0.0, 1.0)

    def add_calibration(self, m: Optional[dict], ear: Optional[float], mar: Optional[float]):
        if m is not None:
            self.cal_samples.append({k: m[k] for k in METRIC_KEYS})
        if ear:
            self.cal_ear.append(ear)
        if mar:
            self.cal_mar.append(mar)

    def finish_calibration(self) -> bool:
        if len(self.cal_samples) < CAL_MIN_SAMPLES:
            return False
        self.baseline, self.mad = {}, {}
        for k in METRIC_KEYS:
            v = np.array([s[k] for s in self.cal_samples], dtype=np.float64)
            med = float(np.median(v))
            self.baseline[k] = med
            # Robust spread → the dead-zone adapts to how still this person sits.
            self.mad[k] = float(np.median(np.abs(v - med)) * 1.4826)
        if len(self.cal_ear) >= 8:
            self.ear_base = float(clamp(np.median(self.cal_ear), 0.15, 0.45))
        if len(self.cal_mar) >= 8:
            self.mar_base = float(clamp(np.median(self.cal_mar), 0.02, 0.35))
        self.calibrated = True
        self.score = self.raw_score = 100.0
        self.status = "GOOD"
        self._cand, self._cand_n = "GOOD", 0
        return True

    @property
    def ear_thresh(self) -> float:
        return float(clamp(self.ear_base * 0.72, 0.11, 0.30))

    @property
    def mar_thresh(self) -> float:
        return float(max(self.mar_base * 2.4, 0.45))

    # -- scoring -------------------------------------------------------------
    def smooth_metrics(self, m: dict, alpha: float = 0.35) -> dict:
        if self.metrics_ema is None:
            self.metrics_ema = {k: m[k] for k in METRIC_KEYS}
        else:
            for k in METRIC_KEYS:
                self.metrics_ema[k] = alpha * m[k] + (1 - alpha) * self.metrics_ema[k]
        return self.metrics_ema

    def compute_score(self, sm: dict, sensitivity: float) -> float:
        total, parts = 0.0, {}
        for k, spec in METRICS.items():
            d = sm[k] - self.baseline[k]
            if spec.direction == "dec":
                dev = -d
            elif spec.direction == "inc":
                dev = d
            else:
                dev = abs(d)
            tol = max(self.mad[k] * 2.5, spec.tol)
            span = max(spec.span / max(sensitivity, 0.2), 1e-6)
            pen = clamp((dev - tol) / span, 0.0, 1.0)
            parts[k] = pen
            total += spec.weight * pen
        self.parts = parts
        worst = max(parts.items(), key=lambda kv: METRICS[kv[0]].weight * kv[1])
        self.tip = METRICS[worst[0]].tip if worst[1] > 0.18 else ""
        return float(clamp(100.0 * (1.0 - total), 0.0, 100.0))

    def update_posture(self, m: Optional[dict], now: float, sensitivity: float, on_break: bool):
        dt = 0.0 if self.last_t is None else clamp(now - self.last_t, 0.0, 0.5)
        self.last_t = now

        if m is None:
            self._set_status("IDLE")
            return

        sm = self.smooth_metrics(m)
        self.raw_score = self.compute_score(sm, sensitivity)
        # light EMA on the score itself: the gauge should glide, not twitch
        self.score = 0.25 * self.raw_score + 0.75 * self.score
        self.history.append((now, self.score))

        if on_break:
            self._set_status(status_for(self.score))
            return

        self.score_sum += self.score * max(dt, 1e-3)
        self.score_n += max(dt, 1e-3)
        st_now = status_for(self.score)
        self._set_status(st_now)
        if self.status in self.time_in:
            self.time_in[self.status] += dt

        # ---- streak + recovery bonus ----
        if self.status == "GOOD":
            self.streak_s += dt
            self.best_streak_s = max(self.best_streak_s, self.streak_s)
            if self._await_recovery_until and now <= self._await_recovery_until:
                self.recoveries += 1
                self.bonus += 25
                self._await_recovery_until = None
                self._queue_chime("recover", now, cooldown=8.0)
            elif self._await_recovery_until and now > self._await_recovery_until:
                self._await_recovery_until = None
            self._bad_since = None
        elif self.status == "BAD":
            if self._bad_since is None:
                self._bad_since = now
                # 12-second window to fix it and keep the streak alive
                self._await_recovery_until = now + 12.0
            if now - self._bad_since > 3.0:
                self.streak_s = 0.0

        # ---- predictive trend ----
        self._update_trend(now)

    def _set_status(self, new: str, needed: int = 7):
        """Hysteresis: a status must hold for several frames before it flips."""
        if new == self.status:
            self._cand_n = 0
            return
        if new == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = new, 1
        if self._cand_n >= needed:
            self.status = new
            self._cand_n = 0

    def _update_trend(self, now: float, window: float = 45.0):
        pts = [(t, s) for t, s in self.history if now - t <= window]
        if len(pts) < 20:
            self.trend, self.eta, self.predicting = 0.0, None, False
            return
        ts = np.array([p[0] for p in pts], dtype=np.float64)
        ss = np.array([p[1] for p in pts], dtype=np.float64)
        ts -= ts[0]
        try:
            slope = float(np.polyfit(ts, ss, 1)[0]) * 60.0   # points / minute
        except Exception:
            slope = 0.0
        self.trend = slope
        if slope < -6.0 and self.score > T_WATCH:
            self.eta = (self.score - T_WATCH) / (-slope) * 60.0
            self.predicting = self.eta < 75.0
        else:
            self.eta = None
            self.predicting = False

    # -- fatigue -------------------------------------------------------------
    def update_fatigue(self, ear: Optional[float], mar: Optional[float], now: float):
        if ear is not None:
            self.ear = ear
            closed = ear < self.ear_thresh
            self.perclos_win.append((now, closed))
            if closed:
                if self.eye_closed_since is None:
                    self.eye_closed_since = now
                elif now - self.eye_closed_since > 1.1:
                    # sustained closure = micro-sleep, counted once per event
                    if not getattr(self, "_ms_flag", False):
                        self.microsleeps += 1
                        self._ms_flag = True
            else:
                if self.eye_closed_since is not None:
                    dur = now - self.eye_closed_since
                    if 0.05 <= dur <= 0.7:
                        self.blinks += 1
                        self.blink_times.append(now)
                self.eye_closed_since = None
                self._ms_flag = False

        if mar is not None:
            self.mar = mar
            if mar > self.mar_thresh:
                if self._yawn_since is None:
                    self._yawn_since = now
                elif now - self._yawn_since > 0.8 and not getattr(self, "_yawn_flag", False):
                    self.yawns += 1
                    self.yawn_times.append(now)
                    self._yawn_flag = True
            else:
                self._yawn_since = None
                self._yawn_flag = False

        # blink rate over the last minute, scaled early in the session
        elapsed = max(now - (self.session_start or now), 1e-3)
        win = min(60.0, max(elapsed, 12.0))
        recent = [t for t in self.blink_times if now - t <= win]
        self.blink_rate = len(recent) / win * 60.0
        self.dry_eye = elapsed > 90 and self.blink_rate < 10.0

        # PERCLOS: fraction of the last 60s with the eyes closed
        pw = [(t, c) for t, c in self.perclos_win if now - t <= 60.0]
        perclos = (sum(1 for _, c in pw if c) / len(pw)) if pw else 0.0

        yawns_5m = len([t for t in self.yawn_times if now - t <= 300.0])
        f = 0.0
        f += clamp(perclos / 0.22, 0, 1) * 45.0
        f += clamp(self.microsleeps / 3.0, 0, 1) * 30.0
        f += clamp(yawns_5m / 3.0, 0, 1) * 20.0
        if self.dry_eye or self.blink_rate > 34:
            f += 5.0
        self.fatigue = 0.1 * clamp(f, 0, 100) + 0.9 * self.fatigue
        self.fatigue_sum += self.fatigue
        self.fatigue_n += 1
        self.fatigue_label = (
            "Fresh" if self.fatigue < 25 else
            "Mild" if self.fatigue < 50 else
            "Drowsy" if self.fatigue < 75 else "Critical"
        )
        return perclos

    # -- breaks --------------------------------------------------------------
    def schedule_breaks(self, now: float, interval_min: float):
        if self.next_break is None:
            self.next_break = now + interval_min * 60.0

    def check_break(self, now: float, interval_min: float) -> bool:
        if self.break_until is not None:
            if now >= self.break_until:
                self.break_until = None
                self.next_break = now + interval_min * 60.0
                self.breaks_taken += 1
                return False
            return True
        if self.next_break is not None and now >= self.next_break:
            self.break_until = now + 30.0
            self._queue_chime("break", now, cooldown=0.0)
            return True
        return False

    def start_break_now(self, now: float):
        self.break_until = now + 30.0
        self._queue_chime("break", now, cooldown=0.0)

    # -- alerts --------------------------------------------------------------
    def _queue_chime(self, kind: str, now: float, cooldown: float):
        if now - self.last_chime.get(kind, 0.0) >= cooldown:
            self.last_chime[kind] = now
            self.audio_token += 1
            self.pending_chime = kind

    def maybe_alert(self, now: float, audio_on: bool):
        if not audio_on:
            self.pending_chime = None
            return
        if self.status == "BAD":
            self._queue_chime("bad", now, cooldown=30.0)
        elif self.status == "WATCH":
            self._queue_chime("watch", now, cooldown=45.0)
        elif self.predicting:
            self._queue_chime("predict", now, cooldown=60.0)

    # -- summary -------------------------------------------------------------
    def avg_score(self) -> float:
        return self.score_sum / self.score_n if self.score_n > 0 else 0.0

    def spine_age(self) -> Tuple[int, str, str]:
        avg = self.avg_score()
        total = sum(self.time_in.values()) or 1.0
        red = self.time_in["BAD"] / total
        fat = (self.fatigue_sum / self.fatigue_n) if self.fatigue_n else 0.0
        age = 22.0
        age += (100.0 - avg) * 0.55
        age += red * 18.0
        age += (fat / 100.0) * 8.0
        age -= min(self.recoveries * 0.4, 4.0)
        age = int(round(clamp(age, 18, 79)))
        if age <= 26:
            return age, "Textbook", "Cervical loading stayed in a healthy range the whole session."
        if age <= 34:
            return age, "Holding up", "Mostly aligned, with short slips that you corrected."
        if age <= 46:
            return age, "Feeling the desk", "Sustained forward-head loading. Raise the screen and take breaks."
        if age <= 60:
            return age, "Overloaded", "Long stretches of slouch. This is the pattern that becomes chronic."
        return age, "Critical load", "Your neck spent most of the session under heavy strain."


# ============================================================================
# DRAWING — a custom skeleton, because MediaPipe's default red/green dots
# look like a tutorial screenshot, and this has to look good on a projector.
# ============================================================================
def draw_glow_line(img, p1, p2, color, thick=3, glow=11):
    overlay = img.copy()
    cv2.line(overlay, p1, p2, color, glow, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)
    cv2.line(img, p1, p2, color, thick, cv2.LINE_AA)


def draw_joint(img, p, color, r=5):
    overlay = img.copy()
    cv2.circle(overlay, p, r + 6, color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.circle(img, p, r, color, -1, cv2.LINE_AA)
    cv2.circle(img, p, max(r - 3, 1), (255, 255, 255), -1, cv2.LINE_AA)


def draw_skeleton(frame, lms, m: Optional[dict], color_hex: str, show_angle: bool = True):
    h, w = frame.shape[:2]
    color = hex_to_bgr(color_hex)
    accent = hex_to_bgr(ACCENT)

    def P(i):
        return (int(lms[i].x * w), int(lms[i].y * h))

    for a, b in EDGES:
        if lms[a].visibility > 0.45 and lms[b].visibility > 0.45:
            draw_glow_line(frame, P(a), P(b), color, 3, 11)
    for i in {p for e in EDGES for p in e}:
        if lms[i].visibility > 0.45:
            draw_joint(frame, P(i), color, 5)

    if m is None:
        return frame

    head = tuple(np.int32(m["_head"]))
    sh_mid = tuple(np.int32(m["_sh_mid"]))
    # the cervical vector — the line this whole app is about
    draw_glow_line(frame, head, sh_mid, color, 3, 13)
    draw_joint(frame, head, color, 6)

    if show_angle:
        ear = tuple(np.int32(m["_ear"]))
        shp = tuple(np.int32(m["_ear_sh"]))
        r = int(clamp(m["_shoulder_w"] * 0.30, 40, 110))
        cv2.line(frame, shp, (shp[0] + r + 18, shp[1]), accent, 1, cv2.LINE_AA)
        cv2.line(frame, shp, ear, accent, 2, cv2.LINE_AA)
        end = -float(m["cva"]) if ear[0] >= shp[0] else 180.0 + float(m["cva"])
        start = 0.0 if ear[0] >= shp[0] else 180.0
        try:
            cv2.ellipse(frame, shp, (r, r), 0, start, end, accent, 2, cv2.LINE_AA)
        except Exception:
            pass
        label = f"CVA {m['cva']:.0f}"
        lx = shp[0] + (16 if ear[0] >= shp[0] else -110)
        ly = max(shp[1] - r - 14, 22)
        cv2.putText(frame, label, (lx, ly), cv2.FONT_HERSHEY_DUPLEX, 0.62, accent, 1, cv2.LINE_AA)
    return frame


def draw_frame_hud(frame, text: str, color_hex: str):
    """A slim caption strip at the bottom of the video, used for phase text."""
    h, w = frame.shape[:2]
    if not text:
        return frame
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 46), (w, h), (8, 11, 18), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.line(frame, (0, h - 46), (w, h - 46), hex_to_bgr(color_hex), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (18, h - 17), cv2.FONT_HERSHEY_DUPLEX, 0.62,
                (233, 239, 249), 1, cv2.LINE_AA)
    return frame


def draw_calibration_ui(frame, phase: str, progress: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (8, 11, 18), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    cx, cy, r = w // 2, h // 2, int(min(w, h) * 0.20)
    cv2.circle(frame, (cx, cy), r, (40, 55, 80), 4, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360 * clamp(progress, 0, 1),
                hex_to_bgr(ACCENT), 6, cv2.LINE_AA)

    if phase == "warmup":
        big, small = "GET SET", "Sit tall. Shoulders down. Eyes on the screen."
    else:
        big, small = f"{max(0, int(CAL_CAPTURE_S * (1 - progress)) + 1)}", "Hold your best posture — this becomes your baseline."
    (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
    cv2.putText(frame, big, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX,
                1.5, (233, 239, 249), 2, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(small, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
    cv2.putText(frame, small, (cx - sw // 2, cy + r + 46), cv2.FONT_HERSHEY_DUPLEX,
                0.62, (126, 143, 168), 1, cv2.LINE_AA)
    return frame


def to_b64_jpeg(frame_bgr, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ============================================================================
# SHARE CARD — built with OpenCV so there are no font files to ship.
# ============================================================================
def build_share_card(engine: PostureEngine, minutes: float) -> bytes:
    W = H = 1080
    card = np.zeros((H, W, 3), dtype=np.uint8)
    top, bot = np.array(hex_to_bgr("#16233A"), np.float32), np.array(hex_to_bgr("#080B12"), np.float32)
    for y in range(H):
        card[y, :] = top + (bot - top) * (y / H)

    age, label, note = engine.spine_age()
    avg = engine.avg_score()
    color = hex_to_bgr(C_GOOD if age <= 30 else C_WATCH if age <= 46 else C_BAD)

    def text(s, x, y, scale, col, thick=1, font=cv2.FONT_HERSHEY_DUPLEX, center=False):
        if center:
            (tw, _), _ = cv2.getTextSize(s, font, scale, thick)
            x = x - tw // 2
        cv2.putText(card, s, (int(x), int(y)), font, scale, col, thick, cv2.LINE_AA)

    text("POSTURE:", 80, 110, 1.3, (233, 239, 249), 2)
    cv2.line(card, (80, 145), (1000, 145), hex_to_bgr(LINE), 2)
    text("SPINE AGE", W // 2, 300, 0.9, hex_to_bgr(MUTED), 1, center=True)

    big = str(age)
    (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 9.0, 14)
    cv2.putText(card, big, (W // 2 - bw // 2, 300 + bh + 60), cv2.FONT_HERSHEY_DUPLEX,
                9.0, color, 14, cv2.LINE_AA)
    text(label.upper(), W // 2, 560, 1.4, (233, 239, 249), 2, center=True)

    words, line, lines = note.split(), "", []
    for wd in words:
        t = (line + " " + wd).strip()
        if cv2.getTextSize(t, cv2.FONT_HERSHEY_DUPLEX, 0.75, 1)[0][0] > 820:
            lines.append(line)
            line = wd
        else:
            line = t
    lines.append(line)
    for i, ln in enumerate(lines[:3]):
        text(ln, W // 2, 615 + i * 38, 0.75, hex_to_bgr(MUTED), 1, center=True)

    stats = [
        ("AVG SCORE", f"{avg:.0f}"),
        ("SESSION", fmt_clock(minutes * 60)),
        ("BEST STREAK", fmt_clock(engine.best_streak_s)),
        ("SAVES", str(engine.recoveries)),
    ]
    x0, y0, bw2, bh2, gap = 80, 760, 220, 150, 20
    for i, (k, v) in enumerate(stats):
        x = x0 + i * (bw2 + gap)
        cv2.rectangle(card, (x, y0), (x + bw2, y0 + bh2), hex_to_bgr(SURFACE), -1)
        cv2.rectangle(card, (x, y0), (x + bw2, y0 + bh2), hex_to_bgr(LINE), 1)
        text(k, x + bw2 // 2, y0 + 45, 0.55, hex_to_bgr(MUTED), 1, center=True)
        text(v, x + bw2 // 2, y0 + 108, 1.5, (233, 239, 249), 2, center=True)

    text("100% on-device  ·  nothing left this laptop", W // 2, 1010, 0.68,
         hex_to_bgr(MUTED), 1, center=True)
    ok, buf = cv2.imencode(".png", card)
    return buf.tobytes() if ok else b""


# ============================================================================
# HTML RENDERERS
# ============================================================================
def render_video(ph, b64: str, color_hex: str, status: str, fps: float):
    if not b64:
        return
    glow = hex_to_rgba(color_hex, 0.42)
    inner = hex_to_rgba(color_hex, 0.10)
    ph.markdown(
        f'<div class="pr-stage" style="border-color:{hex_to_rgba(color_hex,0.55)};'
        f'box-shadow:0 0 0 1px {hex_to_rgba(color_hex,0.55)}, 0 0 48px {glow}, inset 0 0 70px {inner};">'
        f'<img src="data:image/jpeg;base64,{b64}"/>'
        f'<div class="pr-badge" style="background:{hex_to_rgba(color_hex,0.14)};color:{color_hex};'
        f'border-color:{hex_to_rgba(color_hex,0.45)}">{STATUS_TEXT.get(status,status)}</div>'
        f'<div class="pr-fps">{fps:.0f} FPS · ON-DEVICE</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def render_ambient(ph, color_hex: str, intensity: float):
    if intensity <= 0.01:
        ph.markdown('<div class="pr-ambient"></div>', unsafe_allow_html=True)
        return
    ph.markdown(
        f'<div class="pr-ambient" style="box-shadow: inset 0 0 {int(70+70*intensity)}px '
        f'{int(6+16*intensity)}px {hex_to_rgba(color_hex, 0.10 + 0.26*intensity)};"></div>',
        unsafe_allow_html=True,
    )


def _card(label, value, unit, sub, color, bar=None, wide=False):
    bar_html = ""
    if bar is not None:
        bar_html = (f'<div class="pr-bar"><i style="width:{clamp(bar,0,1)*100:.0f}%;'
                    f'background:{color}"></i></div>')
    u = f'<span class="u">{unit}</span>' if unit else ""
    return (f'<div class="pr-card{" wide" if wide else ""}">'
            f'<div class="pr-label">{label}</div>'
            f'<div class="pr-value" style="color:{color}">{value}{u}</div>'
            f'<div class="pr-sub">{sub}</div>{bar_html}</div>')


def render_cards(ph, e: PostureEngine, now: float, daily_streak: int, perclos: float):
    color = STATUS_COLORS.get(e.status, C_IDLE)

    if e.trend > 3:
        tr_v, tr_c, tr_s = "RISING", C_GOOD, "You're straightening up."
    elif e.trend < -6:
        tr_v, tr_c = "FALLING", C_BAD
        tr_s = (f"Slump predicted in ~{e.eta:.0f}s at this rate."
                if e.eta else "Posture is decaying steadily.")
    else:
        tr_v, tr_c, tr_s = "STABLE", MUTED, "Holding your baseline."

    fat_c = (C_GOOD if e.fatigue < 25 else C_WATCH if e.fatigue < 50 else C_BAD)
    fat_sub = f"{e.blink_rate:.0f} blinks/min · {e.yawns} yawns"
    if e.microsleeps:
        fat_sub += f" · {e.microsleeps} micro-sleep{'s' if e.microsleeps > 1 else ''}"
    if e.dry_eye:
        fat_sub += " · dry-eye risk"

    streak_sub = f"Best {fmt_clock(e.best_streak_s)}"
    if e.recoveries:
        streak_sub += f" · {e.recoveries} save{'s' if e.recoveries > 1 else ''} (+{e.bonus})"

    if e.break_until:
        nb = "now"
    elif e.next_break:
        nb = fmt_clock(max(0, e.next_break - now))
    else:
        nb = "—"

    html = '<div class="pr-grid">'
    html += _card("Posture score", f"{e.score:.0f}", "/100",
                  STATUS_TEXT.get(e.status, ""), color, bar=e.score / 100.0)
    html += _card("Fatigue", e.fatigue_label.upper(), "", fat_sub, fat_c, bar=e.fatigue / 100.0)
    html += _card("Streak", fmt_clock(e.streak_s), "", streak_sub, C_GOOD if e.streak_s > 0 else MUTED)
    html += _card("Trend", tr_v, "", tr_s, tr_c)
    html += _card("CVA", f"{e.metrics_ema['cva']:.0f}" if e.metrics_ema else "—", "°",
                  f"Baseline {e.baseline['cva']:.0f}°" if e.baseline else "Not calibrated", ACCENT)
    html += _card("Next break", nb, "", f"{e.breaks_taken} taken · day streak {daily_streak}", ACCENT)
    html += "</div>"
    ph.markdown(html, unsafe_allow_html=True)


def render_strip(ph, e: PostureEngine, now: float, window: float = 120.0):
    pts = [(t, s) for t, s in e.history if now - t <= window]
    W, H = 600.0, 100.0
    if len(pts) < 2:
        poly = f"0,{H/2:.0f} {W},{H/2:.0f}"
    else:
        t0 = pts[0][0]
        span = max(now - t0, 1e-3)
        poly = " ".join(f"{(t-t0)/span*W:.1f},{(100.0-s):.1f}" for t, s in pts)

    proj = ""
    if e.predicting and e.eta:
        x_now = W
        # project the decay forward into the "future" gutter on the right
        proj = (f'<line x1="{x_now}" y1="{100.0-e.score:.1f}" x2="{W}" y2="{100.0-T_WATCH:.1f}" '
                f'stroke="{C_BAD}" stroke-width="1.6" stroke-dasharray="4 3" opacity="0.9"/>')

    color = STATUS_COLORS.get(e.status, C_IDLE)
    trend_txt = f"{e.trend:+.0f} pts/min"
    eta_txt = f"slump in ~{e.eta:.0f}s" if (e.predicting and e.eta) else "no slump predicted"
    svg = f"""
<div class="pr-strip">
  <svg viewBox="0 0 {W:.0f} {H:.0f}" preserveAspectRatio="none">
    <rect x="0" y="0" width="{W:.0f}" height="20" fill="{hex_to_rgba(C_GOOD,0.10)}"/>
    <rect x="0" y="20" width="{W:.0f}" height="20" fill="{hex_to_rgba(C_WATCH,0.09)}"/>
    <rect x="0" y="40" width="{W:.0f}" height="60" fill="{hex_to_rgba(C_BAD,0.09)}"/>
    <line x1="0" y1="20" x2="{W:.0f}" y2="20" stroke="{hex_to_rgba(C_GOOD,0.45)}" stroke-width="0.8"/>
    <line x1="0" y1="40" x2="{W:.0f}" y2="40" stroke="{hex_to_rgba(C_WATCH,0.45)}" stroke-width="0.8"/>
    <polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2.2"
      stroke-linejoin="round" stroke-linecap="round" vector-effect="non-scaling-stroke"/>
    {proj}
  </svg>
  <div class="pr-legend">
    <span>Last {int(window)}s</span><span>Trend <b>{trend_txt}</b></span>
    <span>Forecast <b>{eta_txt}</b></span><span>Bands <b>80 / 60</b></span>
  </div>
</div>"""
    ph.markdown(svg, unsafe_allow_html=True)


def render_banner(ph, e: PostureEngine):
    if e.break_until:
        ph.empty()
        return
    if e.status == "IDLE":
        k, msg, c = "NO SUBJECT", "Step back into frame — I can't see your shoulders.", MUTED
    elif e.predicting and e.status != "BAD":
        k, msg, c = "PREDICTED", (f"You're drifting. At this rate you'll be slouching in "
                                  f"~{e.eta:.0f}s — reset now, before it costs you."), C_WATCH
    elif e.tip:
        k, msg, c = ("CORRECT" if e.status == "BAD" else "ADJUST"), e.tip, STATUS_COLORS[e.status]
    elif e.microsleeps and e.fatigue > 60:
        k, msg, c = "FATIGUE", "Eyes are closing for too long. Take a real break.", C_BAD
    else:
        k, msg, c = "ALIGNED", "Cervical loading is in a healthy range. Keep it here.", C_GOOD
    ph.markdown(
        f'<div class="pr-banner" style="background:{hex_to_rgba(c,0.08)};border-color:{hex_to_rgba(c,0.35)}">'
        f'<span class="k" style="background:{hex_to_rgba(c,0.16)};color:{c}">{k}</span>'
        f'<span style="color:#E9EFF9">{msg}</span></div>',
        unsafe_allow_html=True,
    )


def render_break(ph, e: PostureEngine, now: float):
    left = max(0.0, (e.break_until or now) - now)
    elapsed = 30.0 - left
    idx = min(int(elapsed // 10), len(STRETCHES) - 1)
    name, how, _ = STRETCHES[idx]
    ph.markdown(
        f'<div class="pr-break"><div class="n">{int(left)+1}</div>'
        f"<h3>{name}</h3><p>{how}</p>"
        f'<p style="margin-top:10px;font-size:.72rem;letter-spacing:.18em;color:#4E5F78">'
        f"MOVE {idx+1} OF {len(STRETCHES)} · SCORING PAUSED</p></div>",
        unsafe_allow_html=True,
    )


def render_snapshot(ph, e: PostureEngine):
    if not (e.baseline_jpg and e.current_jpg):
        ph.empty()
        return
    delta = e.score - 100.0
    dc = C_GOOD if e.score >= T_GOOD else C_WATCH if e.score >= T_WATCH else C_BAD
    ph.markdown(
        f'<div class="pr-snap">'
        f'<figure><img src="data:image/jpeg;base64,{e.baseline_jpg}"/>'
        f"<figcaption>Calibrated baseline · 100</figcaption></figure>"
        f'<figure><img src="data:image/jpeg;base64,{e.current_jpg}" '
        f'style="border-color:{hex_to_rgba(dc,0.55)}"/>'
        f'<figcaption>Now · <span style="color:{dc}">{e.score:.0f} '
        f"({delta:+.0f})</span></figcaption></figure></div>",
        unsafe_allow_html=True,
    )


# ============================================================================
# STATE + CONTROLS
# ============================================================================
def rerun():
    try:
        st.rerun()
    except AttributeError:                      # Streamlit < 1.27
        st.experimental_rerun()


def init_state():
    ss = st.session_state
    if "engine" not in ss:
        ss.engine = PostureEngine()
    ss.setdefault("running", False)
    ss.setdefault("summary", None)
    ss.setdefault("cap", None)
    ss.setdefault("store", load_store())
    ss.setdefault("share_png", None)


def start_session():
    e = st.session_state.engine
    e.hard_reset()
    e.session_start = time.time()
    st.session_state.summary = None
    st.session_state.share_png = None
    st.session_state.running = True


def stop_session():
    ss = st.session_state
    e = ss.engine
    ss.running = False
    if e.session_start and e.score_n > 3:
        dur = time.time() - e.session_start
        age, label, note = e.spine_age()
        ss.summary = {
            "duration": dur,
            "avg": e.avg_score(),
            "age": age, "label": label, "note": note,
            "time_in": dict(e.time_in),
            "best_streak": e.best_streak_s,
            "recoveries": e.recoveries,
            "bonus": e.bonus,
            "blinks": e.blinks,
            "yawns": e.yawns,
            "microsleeps": e.microsleeps,
            "blink_rate": e.blink_rate,
            "breaks": e.breaks_taken,
            "baseline_jpg": e.baseline_jpg,
            "current_jpg": e.current_jpg,
        }
        if dur >= 60:
            ss.store = register_day(ss.store)
            ss.store.setdefault("sessions", []).append({
                "at": datetime.now().isoformat(timespec="seconds"),
                "minutes": round(dur / 60, 1),
                "avg_score": round(e.avg_score(), 1),
                "spine_age": age,
            })
            ss.store["sessions"] = ss.store["sessions"][-60:]
            save_store(ss.store)
    release_camera()


def recalibrate():
    st.session_state.engine.reset_calibration()


def sidebar() -> dict:
    ss = st.session_state
    sb = st.sidebar
    sb.markdown(
        '<div style="font-family:ui-monospace,monospace;font-size:1.5rem;font-weight:700;'
        'letter-spacing:-.03em;color:#E9EFF9">POSTU<span style="color:#6E8BFF">Re:</span></div>'
        '<div style="font-family:ui-monospace,monospace;font-size:.6rem;letter-spacing:.22em;'
        'color:#7E8FA8;margin:4px 0 14px">SPINE TELEMETRY · OFFLINE</div>',
        unsafe_allow_html=True,
    )

    if ss.running:
        sb.button("Stop session", type="primary", on_click=stop_session)
        sb.button("Recalibrate baseline", on_click=recalibrate)
    else:
        sb.button("Start session", type="primary", on_click=start_session)

    sb.markdown("---")
    sb.markdown("**Coaching**")
    sensitivity = sb.slider("Sensitivity", 0.6, 1.8, 1.0, 0.1,
                            help="Higher means smaller deviations from your baseline cost you points.")
    break_min = sb.slider("Stretch break every (min)", 5, 45, 20, 1)

    sb.markdown("**Feedback**")
    ambient = sb.checkbox("Ambient screen glow", True)
    audio = sb.checkbox("Audio nudges", True)
    fatigue = sb.checkbox("Fatigue engine (eyes + yawns)", True)

    sb.markdown("**Overlay**")
    skeleton = sb.checkbox("Draw skeleton", True)
    show_angle = sb.checkbox("Show CVA angle", True)
    snapshot = sb.checkbox("Evolution snapshot", True)

    with sb.expander("Performance & camera"):
        cam_index = st.number_input("Camera index", 0, 5, 0, 1)
        complexity = st.select_slider("Pose model", options=[0, 1], value=1,
                                      format_func=lambda v: "Fast (0)" if v == 0 else "Accurate (1)")
        face_every = st.select_slider("Face mesh every N frames", options=[1, 2, 3], value=2)

    store = ss.store
    sb.markdown("---")
    sb.markdown(
        f'<div class="pr-label">DAY STREAK</div>'
        f'<div style="font-family:ui-monospace,monospace;font-size:1.8rem;font-weight:700;color:#35E6A6">'
        f'{store.get("daily_streak",0)}<span style="font-size:.8rem;color:#7E8FA8"> days</span></div>'
        f'<div style="font-size:.72rem;color:#7E8FA8">Best {store.get("best_streak",0)} · '
        f'{len(store.get("sessions",[]))} sessions logged locally</div>',
        unsafe_allow_html=True,
    )
    return dict(sensitivity=sensitivity, break_min=break_min, ambient=ambient, audio=audio,
                fatigue=fatigue, skeleton=skeleton, show_angle=show_angle, snapshot=snapshot,
                cam_index=int(cam_index), complexity=int(complexity), face_every=int(face_every))


def masthead():
    st.markdown(
        '<div class="pr-mast"><p class="pr-wordmark">PostuRe<span>:</span></p>'
        '<div class="pr-tag">Craniovertebral tracking · Fatigue detection · 100% on-device</div></div>'
        '<div class="pr-rule"></div>',
        unsafe_allow_html=True,
    )


# ============================================================================
# THE LIVE SESSION LOOP
# ============================================================================
def run_session(cfg: dict):
    ss = st.session_state
    e: PostureEngine = ss.engine

    cap = get_camera(cfg["cam_index"])
    if cap is None or not cap.isOpened():
        st.error(
            f"Camera {cfg['cam_index']} didn't open. Close Zoom/Meet/Teams and any other app "
            "using the webcam, then press Start again. If you have more than one camera, try "
            "another camera index in the sidebar."
        )
        ss.running = False
        return

    pose = load_pose(cfg["complexity"])
    face = load_face() if cfg["fatigue"] else None

    ambient_ph = st.empty()
    banner_ph = st.empty()
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        video_ph = st.empty()
        strip_ph = st.empty()
    with right:
        cards_ph = st.empty()
        break_ph = st.empty()

    snap_head = st.empty()
    snap_ph = st.empty()
    audio_ph = st.empty()

    if cfg["snapshot"]:
        snap_head.markdown(
            '<div class="pr-label" style="margin:18px 0 8px">POSTURE EVOLUTION · BASELINE VS NOW</div>',
            unsafe_allow_html=True,
        )

    frame_i = 0
    errors = 0
    fps = 0.0
    last_ambient: Optional[Tuple[str, int]] = None
    cal_fail_msg = ""

    while ss.running:
        t0 = time.time()
        try:
            ok, frame = cap.read()
            if not ok or frame is None:
                errors += 1
                if errors > 40:
                    st.error("Lost the camera feed. Press Stop, then Start again.")
                    break
                time.sleep(0.03)
                continue
            errors = 0

            frame = cv2.flip(frame, 1)
            if frame.shape[1] != CAP_W:
                frame = cv2.resize(frame, (CAP_W, int(frame.shape[0] * CAP_W / frame.shape[1])))
            h, w = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = pose.process(rgb)

            ear = mar = None
            if face is not None and frame_i % cfg["face_every"] == 0:
                fres = face.process(rgb)
                if fres.multi_face_landmarks:
                    got = extract_face(fres.multi_face_landmarks[0].landmark, w, h)
                    if got:
                        ear, mar = got

            now = time.time()
            lms = res.pose_landmarks.landmark if res.pose_landmarks else None
            m = extract_metrics(lms, w, h) if lms else None
            disp = frame.copy()

            # ---------------- calibration ----------------
            if not e.calibrated:
                phase, prog = e.cal_phase(now)
                if phase == "capture":
                    e.add_calibration(m, ear, mar)
                if lms is not None and cfg["skeleton"]:
                    draw_skeleton(disp, lms, m, ACCENT, show_angle=False)
                draw_calibration_ui(disp, phase, prog)
                if cal_fail_msg:
                    draw_frame_hud(disp, cal_fail_msg, C_WATCH)
                color, status = ACCENT, "IDLE"

                if phase == "capture" and prog >= 1.0:
                    if e.finish_calibration():
                        cal_fail_msg = ""
                        snap = frame.copy()
                        if lms is not None:
                            draw_skeleton(snap, lms, m, C_GOOD, cfg["show_angle"])
                        e.baseline_jpg = to_b64_jpeg(
                            cv2.resize(snap, (460, int(h * 460 / w))), 78)
                        e.schedule_breaks(now, cfg["break_min"])
                    else:
                        cal_fail_msg = "Couldn't see your shoulders clearly — restarting calibration."
                        e.reset_calibration()

                banner_ph.markdown(
                    '<div class="pr-banner" style="background:rgba(110,139,255,.08);'
                    'border-color:rgba(110,139,255,.35)">'
                    '<span class="k" style="background:rgba(110,139,255,.16);color:#6E8BFF">CALIBRATING</span>'
                    '<span style="color:#E9EFF9">Sit the way you want to sit for the next hour. '
                    'Everything after this is measured against this exact posture.</span></div>',
                    unsafe_allow_html=True,
                )
                cards_ph.empty()
                strip_ph.empty()

            # ---------------- live ----------------
            else:
                on_break = e.check_break(now, cfg["break_min"])
                e.update_posture(m, now, cfg["sensitivity"], on_break)
                perclos = e.update_fatigue(ear, mar, now) if cfg["fatigue"] else 0.0
                if not on_break:
                    e.maybe_alert(now, cfg["audio"])

                color = STATUS_COLORS.get(e.status, C_IDLE)
                status = e.status
                if lms is not None and cfg["skeleton"]:
                    draw_skeleton(disp, lms, m, color, cfg["show_angle"])

                if on_break:
                    ov = disp.copy()
                    cv2.rectangle(ov, (0, 0), (w, h), (8, 11, 18), -1)
                    cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)
                    draw_frame_hud(disp, "STRETCH BREAK · scoring paused", ACCENT)
                    render_break(break_ph, e, now)
                else:
                    break_ph.empty()
                    hud = f"SCORE {e.score:.0f}   STREAK {fmt_clock(e.streak_s)}   {e.fatigue_label.upper()}"
                    draw_frame_hud(disp, hud, color)

                render_cards(cards_ph, e, now, ss.store.get("daily_streak", 0), perclos)
                render_strip(strip_ph, e, now)
                render_banner(banner_ph, e)

            # ---------------- output ----------------
            b64 = to_b64_jpeg(disp, 82)
            dt = time.time() - t0
            fps = (1.0 / dt) if fps == 0 else 0.15 * (1.0 / max(dt, 1e-3)) + 0.85 * fps
            render_video(video_ph, b64, color, status, fps)

            if cfg["ambient"]:
                intensity = {"GOOD": 0.12, "WATCH": 0.55, "BAD": 1.0}.get(e.status, 0.0)
                if not e.calibrated:
                    intensity = 0.25
                key = (color, int(intensity * 10))
                if key != last_ambient:
                    render_ambient(ambient_ph, color, intensity)
                    last_ambient = key
            elif last_ambient is not None:
                ambient_ph.empty()
                last_ambient = None

            if cfg["snapshot"] and e.calibrated and now - e._snap_t > 1.5:
                e._snap_t = now
                e.current_jpg = to_b64_jpeg(cv2.resize(disp, (460, int(h * 460 / w))), 78)
                render_snapshot(snap_ph, e)

            if e.pending_chime:
                play_chime(audio_ph, e.pending_chime, e.audio_token)
                e.pending_chime = None

            frame_i += 1
            time.sleep(max(0.0, (1.0 / TARGET_FPS) - (time.time() - t0)))

        except Exception as exc:  # keep the demo alive no matter what
            errors += 1
            if errors > 25:
                st.exception(exc)
                break
            time.sleep(0.05)


# ============================================================================
# IDLE + SUMMARY SCREENS
# ============================================================================
def render_idle():
    store = st.session_state.store
    st.markdown(
        '<div class="pr-banner" style="background:rgba(110,139,255,.08);border-color:rgba(110,139,255,.35)">'
        '<span class="k" style="background:rgba(110,139,255,.16);color:#6E8BFF">READY</span>'
        '<span style="color:#E9EFF9">Press <b>Start session</b> in the sidebar. '
        'Five seconds of calibration, then the coaching is live.</span></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")
    blocks = [
        (c1, "01 · MEASURE",
         "Craniovertebral angle, neck collapse, head pitch, chair slide and shoulder tilt — "
         "all normalised by your shoulder width, so leaning closer to the camera doesn't fake a score."),
        (c2, "02 · PREDICT",
         "A 45-second regression on your score spots the slump forming and nudges you "
         "before it happens, instead of scolding you after."),
        (c3, "03 · REWARD",
         "Streaks, recovery saves for fixing a slouch fast, and one shareable Spine Age at the end. "
         "Ambient colour and soft chimes only — no pop-ups."),
    ]
    for col, title, body in blocks:
        with col:
            st.markdown(
                f'<div class="pr-card" style="height:100%"><div class="pr-label">{title}</div>'
                f'<div style="color:#E9EFF9;font-size:.92rem;line-height:1.5;margin-top:6px">{body}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    sessions = store.get("sessions", [])
    if sessions:
        st.markdown('<div class="pr-label">RECENT SESSIONS · STORED ON THIS MACHINE ONLY</div>',
                    unsafe_allow_html=True)
        rows = "".join(
            f'<tr><td style="padding:7px 14px 7px 0;color:#7E8FA8">{s["at"].replace("T"," ")}</td>'
            f'<td style="padding:7px 14px 7px 0">{s["minutes"]} min</td>'
            f'<td style="padding:7px 14px 7px 0">avg {s["avg_score"]}</td>'
            f'<td style="padding:7px 0;color:#6E8BFF">spine age {s["spine_age"]}</td></tr>'
            for s in reversed(sessions[-6:])
        )
        st.markdown(
            f'<table style="font-family:ui-monospace,monospace;font-size:.78rem;color:#E9EFF9;'
            f'border-collapse:collapse">{rows}</table>',
            unsafe_allow_html=True,
        )


def render_summary(s: dict):
    ss = st.session_state
    age_color = C_GOOD if s["age"] <= 30 else C_WATCH if s["age"] <= 46 else C_BAD
    st.markdown(
        f'<div class="pr-hero"><div class="cap">SPINE AGE</div>'
        f'<div class="big" style="color:{age_color}">{s["age"]}</div>'
        f'<h2>{s["label"]}</h2>'
        f'<p style="color:#7E8FA8;max-width:620px;margin:8px auto 0">{s["note"]}</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)

    total = sum(s["time_in"].values()) or 1.0
    cards = '<div class="pr-grid" style="grid-template-columns:repeat(4,1fr)">'
    cards += _card("Session", fmt_clock(s["duration"]), "", f"{s['breaks']} stretch breaks", ACCENT)
    cards += _card("Average score", f"{s['avg']:.0f}", "/100",
                   f"{s['time_in']['GOOD']/total*100:.0f}% aligned · "
                   f"{s['time_in']['BAD']/total*100:.0f}% slouched",
                   C_GOOD if s["avg"] >= T_GOOD else C_WATCH if s["avg"] >= T_WATCH else C_BAD,
                   bar=s["avg"] / 100.0)
    cards += _card("Best streak", fmt_clock(s["best_streak"]), "",
                   f"{s['recoveries']} recovery saves · +{s['bonus']} bonus", C_GOOD)
    cards += _card("Fatigue", f"{s['blinks']}", " blinks",
                   f"{s['blink_rate']:.0f}/min · {s['yawns']} yawns · "
                   f"{s['microsleeps']} micro-sleeps",
                   C_BAD if s["microsleeps"] else C_GOOD)
    cards += "</div>"
    st.markdown(cards, unsafe_allow_html=True)

    if s.get("baseline_jpg") and s.get("current_jpg"):
        st.markdown('<div class="pr-label" style="margin:22px 0 8px">'
                    'HOW YOU STARTED VS HOW YOU FINISHED</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="pr-snap">'
            f'<figure><img src="data:image/jpeg;base64,{s["baseline_jpg"]}"/>'
            f"<figcaption>Calibrated baseline</figcaption></figure>"
            f'<figure><img src="data:image/jpeg;base64,{s["current_jpg"]}"/>'
            f"<figcaption>Final frame</figcaption></figure></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 1])
    with c1:
        if ss.share_png is None:
            try:
                ss.share_png = build_share_card(ss.engine, s["duration"] / 60.0)
            except Exception:
                ss.share_png = b""
        if ss.share_png:
            st.download_button("Download share card", ss.share_png,
                               file_name=f"posture-spine-age-{s['age']}.png", mime="image/png")
    with c2:
        st.button("Start another session", type="primary", on_click=start_session)


# ============================================================================
# MAIN
# ============================================================================
def main():
    inject_css()

    if IMPORT_ERROR is not None:
        masthead()
        st.error(
            "MediaPipe or OpenCV failed to import, so the camera pipeline can't start.\n\n"
            f"`{type(IMPORT_ERROR).__name__}: {IMPORT_ERROR}`\n\n"
            "Fix it with:\n\n"
            "```\npip install \"mediapipe==0.10.14\" \"opencv-python==4.10.0.84\" "
            "\"numpy==1.26.4\" \"protobuf<5\"\n```\n\n"
            "MediaPipe needs Python 3.9–3.12. Check with `python --version`."
        )
        st.stop()

    init_state()
    cfg = sidebar()
    masthead()

    if st.session_state.running:
        run_session(cfg)
        # The loop only falls through if the feed died; keep the UI consistent.
        if not st.session_state.running:
            rerun()
    elif st.session_state.summary:
        render_summary(st.session_state.summary)
    else:
        render_idle()


if __name__ == "__main__":
    main()
