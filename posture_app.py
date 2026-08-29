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
import random
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
# Direction: "Drafting Table" — a technical instrument reading a live
# measurement, not a generic wellness dashboard. A real dark navy (not
# near-black), one structural teal accent, and a construction-safety orange
# for the WATCH state instead of a generic amber.
BG_0 = "#0B1E33"      # --ink
BG_1 = "#0E2743"
SURFACE = "#12314F"   # --ink-2
SURFACE_2 = "#173A5C"
LINE = "#1F3F5E"
TEXT = "#EAF3F2"       # --paper
MUTED = "#7C97AA"      # --graphite
ACCENT = "#4FD8C4"     # --drafting

C_GOOD = "#35E6A6"     # kept exactly as-is: the quiet, low-attention state
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
    "neck": MetricSpec(0.26, "dec", 0.020, 0.115, "Lift the crown of your head — your neck is collapsing."),
    # 2D craniovertebral angle (ear→shoulder vs horizontal). The clinical
    # measure this app is framed around, so it carries the most weight and
    # the tightest tolerance of any metric.
    "cva": MetricSpec(0.28, "dec", 1.4, 8.0, "Craniovertebral angle dropping — tuck your chin back."),
    # Nose below the ear-line = looking down at the keyboard.
    "pitch": MetricSpec(0.16, "inc", 0.025, 0.130, "Head tilted down — raise your screen to eye level."),
    # Shoulders sinking in the frame = sliding down the chair.
    "drop": MetricSpec(0.12, "inc", 0.012, 0.070, "You're sinking into the chair — sit back into the backrest."),
    # Face appears larger relative to shoulders = head craning toward screen.
    "face": MetricSpec(0.12, "inc", 0.012, 0.070, "You're creeping toward the screen — push your chair in instead."),
    # Shoulder line rotated = leaning on one arm.
    "tilt": MetricSpec(0.06, "abs", 2.5, 12.0, "One shoulder is dropping — even out your weight."),
}
METRIC_KEYS = list(METRICS.keys())

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


def break_duration(interval_min: float) -> float:
    """Break length scales with how long you've been sitting: a 5-minute
    check-in only needs ~20s, a full hour (or more) earns a real ~90s+ reset.
    """
    slope = (90.0 - 20.0) / (60.0 - 5.0)
    return float(max(20.0, 20.0 + (interval_min - 5.0) * slope))


def pick_stretches(duration_s: float) -> List[Tuple[str, str, str, str]]:
    """Randomised subset of STRETCHES sized to how long this break is."""
    if duration_s < 35:
        n = 1
    elif duration_s < 55:
        n = 2
    elif duration_s < 75:
        n = 3
    else:
        n = 4
    n = min(n, len(STRETCHES))
    return random.sample(STRETCHES, n)


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


def _gradient_canvas(w: int, h: int, top_hex: str, bot_hex: str) -> np.ndarray:
    top = np.array(hex_to_bgr(top_hex), np.float32)
    bot = np.array(hex_to_bgr(bot_hex), np.float32)
    t = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    grad = top[None, None, :] * (1 - t) + bot[None, None, :] * t
    return np.repeat(grad, w, axis=1).astype(np.uint8)


def _add_corner_glow(card: np.ndarray, w: int, h: int, hex_color: str) -> None:
    """A soft off-corner highlight so the card background reads as a real
    treatment rather than a flat fill or a plain top-to-bottom gradient."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - w * 0.18) ** 2 + (yy - h * 0.03) ** 2) / (max(w, h) * 0.85)
    glow = np.clip(1.0 - dist, 0.0, 1.0) ** 2
    tint = np.array(hex_to_bgr(hex_color), np.float32)
    add = glow[..., None] * tint[None, None, :] * 0.22
    card[:] = np.clip(card.astype(np.float32) + add, 0, 255).astype(np.uint8)


def _add_grid_texture(card: np.ndarray, spacing: int = 40, hex_color: str = "#7C97AA",
                       alpha: float = 0.05) -> None:
    """Faint blueprint/graph-paper ruling, matching the app's background texture."""
    tint = np.array(hex_to_bgr(hex_color), np.float32)
    card_f = card.astype(np.float32)
    card_f[::spacing, :, :] = card_f[::spacing, :, :] * (1 - alpha) + tint * alpha
    card_f[:, ::spacing, :] = card_f[:, ::spacing, :] * (1 - alpha) + tint * alpha
    card[:] = np.clip(card_f, 0, 255).astype(np.uint8)


def _gradient_bar(card: np.ndarray, x0: int, y0: int, x1: int, y1: int, hex_stops: List[str]) -> None:
    n = max(x1 - x0, 1)
    stops = np.array([hex_to_bgr(hx) for hx in hex_stops], dtype=np.float32)
    xs = np.linspace(0, 1, len(stops))
    t = np.linspace(0, 1, n)
    bar = np.stack([np.interp(t, xs, stops[:, c]) for c in range(3)], axis=1).astype(np.uint8)
    card[y0:y1, x0:x1] = bar[None, :, :]


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
  /* -- Drafting Table: an engineering instrument reading a live measurement -- */
  --ink:#0B1E33; --ink-2:#12314F; --paper:#EAF3F2; --graphite:#7C97AA;
  --drafting:#4FD8C4; --hazard:#FF8A3D; --critical:#FF4757;
  /* aliases so every rule below is wired to the new palette in one place */
  --bg0:var(--ink); --bg1:#0E2743; --surface:var(--ink-2); --surface2:#173A5C;
  --line:#1F3F5E; --text:var(--paper); --muted:var(--graphite); --accent:var(--drafting);
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --mono: ui-monospace, "SF Mono", SFMono-Regular, "Cascadia Mono", "Consolas", "Roboto Mono", monospace;
  --grid: repeating-linear-gradient(0deg, rgba(124,151,170,.05) 0, rgba(124,151,170,.05) 1px, transparent 1px, transparent 40px),
          repeating-linear-gradient(90deg, rgba(124,151,170,.05) 0, rgba(124,151,170,.05) 1px, transparent 1px, transparent 40px);
}
html, body, [data-testid="stAppViewContainer"]{
  background:
    var(--grid),
    radial-gradient(1200px 700px at 20% -10%, #163256 0%, var(--bg1) 45%, var(--bg0) 100%);
  background-color: var(--bg0);
  color: var(--text);
  font-family: var(--sans);
}
[data-testid="stHeader"]{ background: transparent; }
[data-testid="stToolbar"]{ right: 8px; }
.block-container{ padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer{ visibility: hidden; }

[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #0E2138 0%, #0A1929 100%);
  border-right: 1px solid var(--line);
}
[data-testid="stSidebar"] .stMarkdown p{ color: var(--muted); font-size: 0.86rem; }
[data-testid="stWidgetLabel"] p{
  font-family: var(--mono); font-size:.76rem; letter-spacing:.03em; color: var(--text);
}
[data-testid="stSidebar"] .stMarkdown p strong{
  font-family: var(--mono); font-size:.68rem; letter-spacing:.16em; text-transform:uppercase;
  color: var(--muted); font-weight:600;
}
[data-testid="stExpander"] summary{ font-family: var(--mono); font-size:.78rem; }

/* the Streamlit theme's base font is monospace (so native chrome — buttons,
   widget labels — reads as instrument-panel text); actual sentences inside
   our own markdown blocks (coaching tips, tooltips, card body copy) opt back
   into sans-serif here since they have no class of their own to hang a rule on. */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] div:not([class]),
[data-testid="stMarkdownContainer"] span:not([class]){
  font-family: var(--sans);
}

/* ---------- corner registration marks — a few, not on everything ---------- */
.pr-ticks{ position:relative; }
.pr-ticks::before, .pr-ticks::after{
  content:""; position:absolute; width:13px; height:13px; pointer-events:none;
  border-color: var(--graphite); opacity:.5;
}
.pr-ticks::before{ top:9px; left:9px; border-top:2px solid; border-left:2px solid; }
.pr-ticks::after{ bottom:9px; right:9px; border-bottom:2px solid; border-right:2px solid; }

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
  position: relative; border-radius: 10px; overflow: hidden;
  border: 1px solid var(--line);
  background: var(--grid), #04101F;
  transition: box-shadow .45s ease, border-color .45s ease;
}
.pr-stage img{ display:block; width:100%; height:auto; }
.pr-badge{
  position:absolute; top:14px; left:14px; padding:5px 13px; border-radius:3px;
  font-family: var(--mono); font-size:.68rem; letter-spacing:.18em; font-weight:600;
  border:1px solid; backdrop-filter: blur(8px);
}
.pr-fps{
  position:absolute; bottom:12px; right:14px; font-family:var(--mono); font-size:.62rem;
  letter-spacing:.12em; color:rgba(234,243,242,.45); background:rgba(4,16,31,.55);
  padding:3px 9px; border-radius:3px;
}

/* ---------- metric cards ---------- */
.pr-grid{ display:grid; grid-template-columns:repeat(2,1fr); gap:12px; }
.pr-card{
  background: var(--surface);
  border:1px solid var(--line); border-radius:10px; padding:14px 16px;
  box-shadow: 0 6px 16px rgba(0,0,0,.35);
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

.pr-bar{ height:5px; border-radius:2px; background:#081726; margin-top:10px; overflow:hidden; }
.pr-bar i{ display:block; height:100%; border-radius:2px; transition:width .35s ease; }

/* ---------- telemetry strip (secondary readout — the gauge is the signature element) ---------- */
.pr-strip{
  background: var(--surface);
  border:1px solid var(--line); border-radius:10px; padding:14px 16px 8px;
  box-shadow:0 6px 16px rgba(0,0,0,.35);
}
.pr-strip svg{ display:block; width:100%; height:88px; }
.pr-legend{
  display:flex; gap:16px; font-family:var(--mono); font-size:.58rem; letter-spacing:.14em;
  color:var(--muted); text-transform:uppercase; margin-top:4px; flex-wrap:wrap;
}
.pr-legend b{ color:var(--text); font-weight:600; }

/* ---------- coaching / alert banner ---------- */
.pr-banner{
  border-radius:8px; padding:13px 18px; border:1px solid; display:flex; gap:12px;
  align-items:center; font-size:.92rem; margin-bottom:12px;
}
.pr-banner .k{
  font-family:var(--mono); font-size:.6rem; letter-spacing:.2em; text-transform:uppercase;
  padding:3px 9px; border-radius:3px; white-space:nowrap;
}

/* ---------- stretch break overlay ---------- */
.pr-break{
  border-radius:14px; padding:26px 28px; text-align:center;
  background: var(--surface);
  border:1px solid var(--line);
  box-shadow:0 14px 40px rgba(0,0,0,.5);
}
.pr-break .n{ font-family:var(--mono); font-size:3.4rem; font-weight:700; color:var(--accent); line-height:1; }
.pr-break h3{ margin:10px 0 4px; font-size:1.25rem; color:var(--text); font-family:var(--mono); font-weight:600; }
.pr-break p{ color:var(--muted); font-size:.92rem; margin:0; font-family:var(--sans); }

/* ---------- snapshot ---------- */
.pr-snap{ display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.pr-snap figure{ margin:0; }
.pr-snap img{ width:100%; border-radius:8px; border:1px solid var(--line); display:block; }
.pr-snap figcaption{
  font-family:var(--mono); font-size:.6rem; letter-spacing:.18em; text-transform:uppercase;
  color:var(--muted); margin-top:8px;
}

/* ---------- summary ---------- */
.pr-hero{
  background: var(--surface);
  border:1px solid var(--line); border-radius:14px; padding:34px 32px;
  box-shadow:0 16px 46px rgba(0,0,0,.45); text-align:center;
}
.pr-hero .big{
  font-family:var(--mono); font-size:5.2rem; font-weight:700; line-height:1;
  letter-spacing:-.04em; font-variant-numeric:tabular-nums;
}
.pr-hero .cap{
  font-family:var(--mono); font-size:.66rem; letter-spacing:.24em; text-transform:uppercase; color:var(--muted);
}
.pr-hero h2{
  margin:12px 0 2px; font-size:1.3rem; font-family:var(--mono); font-weight:600;
  letter-spacing:-.01em; color:var(--text);
}

/* ---------- buttons ---------- */
.stButton > button{
  width:100%; border-radius:6px; border:1px solid var(--line);
  background: var(--surface); color:var(--text);
  font-family: var(--mono); font-size:.82rem; letter-spacing:.04em; font-weight:600;
  padding:.6rem 1rem; transition:all .18s ease;
}
.stButton > button:hover{ border-color:var(--accent); color:var(--accent); transform:translateY(-1px); }
.stButton > button[kind="primary"]{
  background:var(--accent); border-color:var(--accent); color:var(--ink);
}
.stButton > button[kind="primary"]:hover{ color:var(--ink); opacity:.9; }
div[data-testid="stMetricValue"]{ font-family:var(--mono); }

/* ---------- card label + tooltip ---------- */
.pr-label-row{ display:flex; align-items:center; gap:5px; margin-bottom:6px; }
.pr-label-row .pr-label{ margin-bottom:0; }
.pr-tip{ position:relative; display:inline-flex; }
.pr-tip-ic{
  width:14px; height:14px; border-radius:50%; border:1px solid var(--muted);
  color:var(--muted); font-family:var(--mono); font-size:.55rem; line-height:12px;
  text-align:center; cursor:help; user-select:none;
}
.pr-tip-box{
  visibility:hidden; opacity:0; position:absolute; bottom:130%; left:0; z-index:20;
  width:220px; background:#081726; border:1px solid var(--line); border-radius:6px;
  padding:9px 11px; font-size:.72rem; line-height:1.4; color:var(--text);
  font-family:var(--sans); font-weight:400; letter-spacing:normal; text-transform:none;
  box-shadow:0 12px 30px rgba(0,0,0,.5);
}
/* No transition: live stat tiles re-render their whole HTML block every
   video frame, which recreates this node many times a second. A CSS
   transition never gets to finish before the next replacement resets it,
   so hovering during a live session shows nothing — an instant toggle
   still reads as a stable tooltip since every frame paints it the same way. */
.pr-tip:hover .pr-tip-box{ visibility:visible; opacity:1; }

/* ---------- session history ---------- */
.pr-hist-strip{
  display:grid; grid-template-columns:repeat(4,1fr); gap:12px; margin-bottom:14px;
}
.pr-hist-stat{
  background: var(--surface); border:1px solid var(--line);
  border-radius:8px; padding:12px 14px; text-align:center;
}
.pr-hist-stat b{
  display:block; font-family:var(--mono); font-size:1.5rem; font-weight:700; color:var(--text);
}
.pr-hist-stat span{
  font-size:.68rem; color:var(--muted); letter-spacing:.06em; text-transform:uppercase;
}
.pr-hist-row{
  display:grid; grid-template-columns:104px 64px 1fr 44px 72px; align-items:center; gap:12px;
  padding:9px 4px; border-bottom:1px solid var(--line);
}
.pr-hist-row:last-child{ border-bottom:none; }
.pr-hist-date{ font-size:.82rem; color:var(--text); }
.pr-hist-mins{ font-size:.74rem; color:var(--muted); }
.pr-hist-bar{ height:5px; border-radius:2px; background:#081726; overflow:hidden; }
.pr-hist-bar i{ display:block; height:100%; border-radius:2px; }
.pr-hist-score{ font-family:var(--mono); font-weight:700; font-size:.86rem; text-align:right; }
.pr-hist-age{ font-size:.68rem; color:var(--muted); text-align:right; }

/* ---------- spine age explanation ---------- */
.pr-explain{ font-size:.8rem; color:var(--muted); max-width:520px; margin:10px auto 0; line-height:1.5; }

/* ---------- signature element: vertical measurement gauge ----------
   A ruled tick-mark scale with a moving indicator, styled like an
   engineering angle gauge / caliper readout. Appears live (CVA card) and
   on the end-of-session summary (Spine Age), always in --drafting. */
.pr-gauge{ display:inline-flex; align-items:center; gap:10px; }
.pr-gauge-track{
  position:relative; width:16px; border-radius:2px;
  background:
    repeating-linear-gradient(180deg, var(--graphite) 0, var(--graphite) 1px, transparent 1px, transparent 9px),
    rgba(124,151,170,.07);
  border:1px solid var(--line);
}
.pr-gauge-fill{
  position:absolute; left:1px; right:1px; bottom:1px;
  background:linear-gradient(180deg, var(--accent), rgba(79,216,196,.18));
  border-radius:0 0 1px 1px;
}
.pr-gauge-marker{
  position:absolute; left:-6px; width:28px; height:2px; background:var(--paper);
  box-shadow:0 0 7px rgba(79,216,196,.85);
}
.pr-gauge-labels{
  display:flex; flex-direction:column; justify-content:space-between;
  font-family:var(--mono); font-size:.6rem; letter-spacing:.06em; color:var(--muted);
}
.pr-gauge-labels span:last-child{ align-self:flex-end; }

/* ---------- stretch break icons + completion moment ---------- */
.pr-stretch-fig{ width:110px; height:110px; margin:4px auto 0; display:block; }
.pr-stretch-fig .part{ transform-box:fill-box; }
@keyframes pr-tilt{
  0%,100%{ transform:rotate(var(--r0,0deg)); }
  50%{ transform:rotate(var(--r1,0deg)); }
}
@keyframes pr-slide{
  0%,100%{ transform:translate(var(--x0,0px),var(--y0,0px)); }
  50%{ transform:translate(var(--x1,0px),var(--y1,0px)); }
}
@keyframes pr-scale{
  0%,100%{ transform:scale(var(--s0,1)); }
  50%{ transform:scale(var(--s1,1)); }
}
@keyframes pr-spin{
  from{ transform:rotate(0deg); }
  to{ transform:rotate(360deg); }
}
@keyframes pr-check-pop{
  0%{ transform:scale(0); opacity:0; }
  60%{ transform:scale(1.15); opacity:1; }
  100%{ transform:scale(1); opacity:1; }
}
.pr-check-mark{
  display:inline-flex; align-items:center; gap:6px; margin-top:8px; padding:5px 12px;
  border-radius:4px; background:rgba(53,230,166,.14); border:1px solid rgba(53,230,166,.4);
  color:#35E6A6; font-family:var(--mono); font-size:.68rem; letter-spacing:.08em;
  animation:pr-check-pop .5s ease-out;
}
.pr-check-glyph{ font-weight:700; }

/* ---------- summary reveal ---------- */
@keyframes pr-reveal{
  0%{ opacity:0; transform:scale(.85) translateY(6px); }
  100%{ opacity:1; transform:scale(1) translateY(0); }
}
.pr-hero .big{ animation:pr-reveal .6s cubic-bezier(.2,.8,.2,1) both; }

@media (prefers-reduced-motion: reduce){
  .pr-ambient,.pr-stage,.pr-bar i{ transition:none !important; }
  .pr-stretch-fig .part{ animation:none !important; }
  .pr-check-mark{ animation:none !important; }
  .pr-hero .big{ animation:none !important; }
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
def make_chime(notes: Tuple[Tuple[float, float, float], ...], volume: float = 0.22,
                decay: float = 3.2) -> str:
    """Return a base64 WAV built from a short sequence of percussive notes.

    Each note is (freq, start_offset_s, note_len_s) and gets its own fast
    attack + exponential decay, so a multi-note phrase reads as distinct taps
    or a falling run rather than one blurred chord.
    """
    sr = 22050
    total = max(start + length for _, start, length in notes) + 0.12
    t = np.linspace(0.0, total, int(sr * total), endpoint=False)
    tone = np.zeros_like(t)
    for freq, start, length in notes:
        local = t - start
        env = np.exp(-decay * np.maximum(local, 0.0)) * (local >= 0.0)
        attack = np.clip(local / 0.012, 0.0, 1.0)  # gentle attack so it never clicks
        tone += np.sin(2 * np.pi * freq * local) * env * attack
    peak = float(np.max(np.abs(tone))) or 1.0
    audio = np.int16(np.clip(tone / peak * volume, -1, 1) * 32767)

    buf = io.BytesIO()
    with wave_lib.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return base64.b64encode(buf.getvalue()).decode("ascii")


# "watch"/"bad"/"predict" are short attention-grabbing phrases meant to cut
# through background noise without a screen glance. "break"/"recover" stay
# closer to a single soft, positive tone since they're not urgent.
CHIME_SPECS = {
    "watch": {  # falling two-note phrase — "you're drifting"
        "notes": ((659.25, 0.00, 0.16), (523.25, 0.15, 0.24)),
        "volume": 0.30, "decay": 9.0,
    },
    "bad": {  # firm falling run — "you've slumped"
        "notes": ((587.33, 0.00, 0.14), (493.88, 0.14, 0.14), (392.00, 0.28, 0.32)),
        "volume": 0.34, "decay": 10.0,
    },
    "predict": {  # two soft knocks then a rise — "heads up, before it happens"
        "notes": ((659.25, 0.00, 0.13), (659.25, 0.16, 0.13), (783.99, 0.32, 0.26)),
        "volume": 0.28, "decay": 9.5,
    },
    "break": {  # bright ascending triad — "time to move"
        "notes": ((523.25, 0.00, 0.40), (659.25, 0.12, 0.40), (783.99, 0.24, 0.55)),
        "volume": 0.20, "decay": 3.2,
    },
    "recover": {  # bright two-note lift — "nice save"
        "notes": ((783.99, 0.00, 0.40), (1046.50, 0.12, 0.55)),
        "volume": 0.20, "decay": 3.2,
    },
}


def play_chime(placeholder, kind: str, token: int) -> None:
    spec = CHIME_SPECS[kind]
    b64 = make_chime(spec["notes"], spec["volume"], spec["decay"])
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
        if st.session_state.get("cap_index") == index:
            return cap
        # a different index was requested — release the old handle first
        try:
            cap.release()
        except Exception:
            pass
        st.session_state.cap = None

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
    st.session_state.cap_index = index
    return cap


def release_camera() -> None:
    cap = st.session_state.get("cap")
    if cap is not None:
        try:
            cap.release()
        except Exception:
            pass
    st.session_state.cap = None
    st.session_state.cap_index = None


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
    # horizontal. Averaged across both ears when both are clearly visible —
    # using a single side flickers between frames as visibility scores jitter.
    def _cva_for(ear_pt, sh_pt) -> float:
        dx = abs(float(ear_pt[0] - sh_pt[0]))
        dy = float(sh_pt[1] - ear_pt[1])
        return math.degrees(math.atan2(dy, dx + 1e-6))

    if V(L_EAR_L) > 0.3 and V(L_EAR_R) > 0.3:
        ear_pt, sh_pt = P(L_EAR_L), sh_l  # used below only for the on-screen angle overlay
        cva = (_cva_for(P(L_EAR_L), sh_l) + _cva_for(P(L_EAR_R), sh_r)) / 2.0
    elif V(L_EAR_R) > 0.3:
        ear_pt, sh_pt = P(L_EAR_R), sh_r
        cva = _cva_for(ear_pt, sh_pt)
    elif V(L_EAR_L) > 0.3:
        ear_pt, sh_pt = P(L_EAR_L), sh_l
        cva = _cva_for(ear_pt, sh_pt)
    else:
        # neither ear is confidently visible — fall back to the eye midpoint
        ear_pt, sh_pt = head, sh_mid
        cva = _cva_for(ear_pt, sh_pt)

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
        "_framing": frame_quality(lms, w, h),
    }


def frame_quality(lms, w: int, h: int) -> Optional[str]:
    """Flags framing that would make every downstream metric unreliable.

    Every metric here is normalised by shoulder width, so bad framing doesn't
    produce an obviously wrong score — it produces a quietly untrustworthy
    one. Returns a short, specific fix, or None for a normally-framed shot.
    """
    def P(i):
        return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float64)

    def V(i):
        return lms[i].visibility

    if V(L_SH_L) < 0.4 or V(L_SH_R) < 0.4:
        return None  # extract_metrics already bails out on this case

    sh_l, sh_r = P(L_SH_L), P(L_SH_R)
    shoulder_w = float(np.linalg.norm(sh_l - sh_r))
    frac = shoulder_w / w
    top_y = min(sh_l[1], sh_r[1])
    mid_y = (sh_l[1] + sh_r[1]) / 2.0

    if frac < 0.16:
        return "Move closer — your shoulders should fill about a third of the frame."
    if frac > 0.65:
        return "Move back a little — you're too close to the camera."
    if sh_l[0] < 0.03 * w or sh_r[0] > 0.97 * w:
        return "Center yourself in the frame — a shoulder is getting cut off."
    if top_y < 0.10 * h:
        return "Lower the camera a little — you're too near the top of the frame."
    if mid_y > 0.92 * h:
        return "Raise the camera a little — your shoulders are too low in the frame."
    return None


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
        self.framing: Optional[str] = None

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
        self.break_total = 30.0
        self.break_exercises: List[tuple] = []
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
            dur = break_duration(interval_min)
            self.break_total = dur
            self.break_exercises = pick_stretches(dur)
            self.break_until = now + dur
            self._queue_chime("break", now, cooldown=0.0)
            return True
        return False

    def start_break_now(self, now: float, interval_min: float):
        dur = break_duration(interval_min)
        self.break_total = dur
        self.break_exercises = pick_stretches(dur)
        self.break_until = now + dur
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
    cv2.rectangle(overlay, (0, h - 46), (w, h), hex_to_bgr(BG_0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    cv2.line(frame, (0, h - 46), (w, h - 46), hex_to_bgr(color_hex), 2, cv2.LINE_AA)
    cv2.putText(frame, text, (18, h - 17), cv2.FONT_HERSHEY_DUPLEX, 0.62,
                hex_to_bgr(TEXT), 1, cv2.LINE_AA)
    return frame


def draw_calibration_ui(frame, phase: str, progress: float):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), hex_to_bgr(BG_0), -1)
    cv2.addWeighted(overlay, 0.42, frame, 0.58, 0, frame)

    cx, cy, r = w // 2, h // 2, int(min(w, h) * 0.20)
    cv2.circle(frame, (cx, cy), r, hex_to_bgr(LINE), 4, cv2.LINE_AA)
    cv2.ellipse(frame, (cx, cy), (r, r), -90, 0, 360 * clamp(progress, 0, 1),
                hex_to_bgr(ACCENT), 6, cv2.LINE_AA)

    if phase == "warmup":
        big, small = "GET SET", "Sit tall. Shoulders down. Eyes on the screen."
    else:
        big, small = f"{max(0, int(CAL_CAPTURE_S * (1 - progress)) + 1)}", "Hold your best posture - this becomes your baseline."
    (tw, th), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)
    cv2.putText(frame, big, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_DUPLEX,
                1.5, hex_to_bgr(TEXT), 2, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(small, cv2.FONT_HERSHEY_DUPLEX, 0.62, 1)
    cv2.putText(frame, small, (cx - sw // 2, cy + r + 46), cv2.FONT_HERSHEY_DUPLEX,
                0.62, hex_to_bgr(MUTED), 1, cv2.LINE_AA)
    return frame


def to_b64_jpeg(frame_bgr, quality: int = 82) -> str:
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


# ============================================================================
# SHARE CARD — built with OpenCV so there are no font files to ship.
# Three styles: a minimal card, a data-rich card, and a tall 9:16 story card.
# Pure OpenCV drawing (no HTML/Pillow render step) — this keeps the card
# generator on the same locked, offline stack as the rest of the app. Every
# element's position is derived from a running layout cursor rather than
# fixed coordinates, so nothing overlaps regardless of style or text length.
# ============================================================================
STYLE_LABELS = {"minimal": "Minimal", "data": "Data-rich", "story": "Story (9:16)"}


def _share_badge(engine: PostureEngine, total_t: float) -> str:
    if engine.recoveries >= 3:
        return "RECOVERY STREAK"
    if engine.best_streak_s >= 300:
        return "LONGEST ALIGNED STRETCH"
    if engine.time_in.get("GOOD", 0.0) / total_t >= 0.8:
        return "MOSTLY ALIGNED"
    if engine.breaks_taken >= 3:
        return "BREAK TAKER"
    return "SESSION LOGGED"


def _share_compare(prev_sessions: list, avg: float, age: int) -> str:
    if not prev_sessions:
        return "First session logged"
    prev_best_avg = max(p["avg_score"] for p in prev_sessions)
    prev_best_age = min(p["spine_age"] for p in prev_sessions)
    if avg > prev_best_avg + 0.5:
        return f"Beat your best avg score by {avg - prev_best_avg:.0f} points"
    if age < prev_best_age:
        return f"New best spine age - {prev_best_age - age} years younger than before"
    return f"Best avg score so far: {prev_best_avg:.0f}"


def build_share_card(engine: PostureEngine, minutes: float, prev_sessions: list,
                      day_streak: int, style: str = "minimal") -> bytes:
    age, label, note = engine.spine_age()
    avg = engine.avg_score()
    total_t = sum(engine.time_in.values()) or 1.0
    age_color = hex_to_bgr(C_GOOD if age <= 30 else C_WATCH if age <= 46 else C_BAD)
    badge = _share_badge(engine, total_t)
    compare = _share_compare(prev_sessions, avg, age)

    tall = style == "story"
    W, H = 1080, (1920 if tall else 1080)
    pad = 74
    # "story" gets extra breathing room so the tall format doesn't feel empty;
    # "data" packs a hero + gauge + stat grid into one square, so it needs the
    # opposite treatment or the grid runs off the bottom edge into the footer.
    sp = 1.5 if tall else (0.82 if style == "data" else 1.0)

    card = _gradient_canvas(W, H, SURFACE_2, BG_0)
    _add_corner_glow(card, W, H, ACCENT)
    _add_grid_texture(card, spacing=44, hex_color=MUTED, alpha=0.045)

    def text_w(s, scale, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
        return cv2.getTextSize(s, font, scale, thick)[0][0]

    def fit_scale(s, max_w, scale, thick=1, min_scale=0.34):
        while scale > min_scale and text_w(s, scale, thick) > max_w:
            scale -= 0.02
        return scale

    def text(s, x, y, scale, col, thick=1, center=False, font=cv2.FONT_HERSHEY_DUPLEX):
        if center:
            x = x - text_w(s, scale, thick, font) // 2
        cv2.putText(card, s, (int(x), int(y)), font, scale, col, thick, cv2.LINE_AA)

    y = 0
    # -- masthead --
    y += int(96 * sp)
    text("POSTURE:", pad, y, 1.05, hex_to_bgr(TEXT), 2)
    y += 26
    cv2.line(card, (pad, y), (W - pad, y), hex_to_bgr(LINE), 2)
    y += int(74 * sp)

    # -- spine age hero --
    text("SPINE AGE", W // 2, y, 0.82, hex_to_bgr(MUTED), 1, center=True)
    y += int(40 * sp)
    big = str(age)
    big_scale, big_thick = (6.6, 11) if style == "data" else (8.4, 13)
    (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, big_scale, big_thick)
    y += bh
    cv2.putText(card, big, (W // 2 - bw // 2, y), cv2.FONT_HERSHEY_DUPLEX, big_scale,
                age_color, big_thick, cv2.LINE_AA)
    y += int(62 * sp)
    text(label.upper(), W // 2, y, 1.05, hex_to_bgr(TEXT), 2, center=True)
    y += int(40 * sp)
    tagline = "A wear score for this session, not a literal age - lower is always better."
    text(tagline, W // 2, y, fit_scale(tagline, W - 2 * pad, 0.6), hex_to_bgr(MUTED), 1, center=True)
    y += int(56 * sp)

    # -- best-to-worst scale with a marker for this session --
    gx0, gx1 = pad + 10, W - pad - 10
    _gradient_bar(card, gx0, y, gx1, y + 14, [C_GOOD, C_WATCH, C_BAD])
    age_pct = clamp((age - 18) / (79 - 18), 0.0, 1.0)
    mx = int(gx0 + age_pct * (gx1 - gx0))
    tri = np.array([[mx - 11, y - 16], [mx + 11, y - 16], [mx, y - 1]], np.int32)
    cv2.fillPoly(card, [tri], hex_to_bgr(TEXT))
    y += 14 + 34
    text("BEST - 18", gx0, y, 0.5, hex_to_bgr(MUTED), 1)
    worst_label = "WORST - 79"
    ww = text_w(worst_label, 0.5, 1)
    text(worst_label, gx1 - ww, y, 0.5, hex_to_bgr(MUTED), 1)
    y += int(60 * sp)

    # -- achievement badge + comparison vs your own history + day streak --
    badge_scale = fit_scale(badge, W - 2 * pad - 44, 0.62, thick=2)
    bw2 = text_w(badge, badge_scale, 2)
    cv2.rectangle(card, (W // 2 - bw2 // 2 - 22, y - 34), (W // 2 + bw2 // 2 + 22, y + 10),
                  hex_to_bgr(C_GOOD), 2)
    text(badge, W // 2, y, badge_scale, hex_to_bgr(C_GOOD), 2, center=True)
    y += int(52 * sp)
    text(compare, W // 2, y, fit_scale(compare, W - 2 * pad, 0.62), hex_to_bgr(TEXT), 1, center=True)
    y += int(40 * sp)
    streak_line = f"{day_streak} day streak" if day_streak else "First day logged"
    text(streak_line, W // 2, y, 0.56, hex_to_bgr(MUTED), 1, center=True)
    y += int(50 * sp)

    # -- stat grid: skipped for "minimal", shown for "data" and "story" --
    if style != "minimal":
        stats = [
            ("AVG SCORE", f"{avg:.0f}"),
            ("SESSION", fmt_clock(minutes * 60)),
            ("BEST STREAK", fmt_clock(engine.best_streak_s)),
            ("SAVES", str(engine.recoveries)),
        ]
        cols, rows = 2, 2
        gap, cell_h = (int(22 * sp), int(150 * sp))
        cell_w = (W - 2 * pad - gap) // cols
        for i, (k, v) in enumerate(stats):
            r, c = divmod(i, cols)
            x0, y0 = pad + c * (cell_w + gap), y + r * (cell_h + gap)
            cv2.rectangle(card, (x0, y0), (x0 + cell_w, y0 + cell_h), hex_to_bgr(SURFACE), -1)
            cv2.rectangle(card, (x0, y0), (x0 + cell_w, y0 + cell_h), hex_to_bgr(LINE), 1)
            text(k, x0 + cell_w // 2, y0 + 46, 0.52, hex_to_bgr(MUTED), 1, center=True)
            text(v, x0 + cell_w // 2, y0 + int(cell_h * 0.72),
                 fit_scale(v, cell_w - 30, 1.3), hex_to_bgr(TEXT), 2, center=True)

    # -- footer: trust line centered, subtle wordmark watermark in the corner --
    foot_y = H - 60
    text("100% on-device - nothing leaves this laptop", W // 2, foot_y, 0.54,
         hex_to_bgr(MUTED), 1, center=True)
    wm = "POSTURE:"
    text(wm, W - pad - text_w(wm, 0.46, 1), H - 26, 0.46, hex_to_bgr("#3F5170"), 1)

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
        f'<div class="pr-stage pr-ticks" style="border-color:{hex_to_rgba(color_hex,0.55)};'
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


TIPS = {
    "posture": "How closely you're matching the posture you calibrated at the start of this "
               "session. 80-100 aligned, 60-79 drifting, below 60 slouched.",
    "fatigue": "Estimated from eye closure, blink rate and yawning. Fresh below 25, Mild "
               "25-49, Drowsy 50-74, Critical 75 and up.",
    "streak": "How long you've held aligned (80+) posture in a row. A quick slip is forgiven "
              "if you fix it within 12 seconds — that counts as a recovery save.",
    "trend": "Where your score has been heading over the last 45 seconds, not just this "
             "instant. A fast enough fall triggers an early heads-up before you actually slouch.",
    "cva": "Craniovertebral angle — the angle between your ear and shoulder versus horizontal. "
           "The clinical marker for forward-head posture; a smaller angle means more forward lean.",
    "next_break": "Countdown to your next scheduled stretch break. Break length scales with how "
                  "long you sit between breaks, so longer gaps earn longer resets.",
}


def render_gauge(pct: float, top_label: str, bottom_label: str, height: int = 84) -> str:
    """The signature 'drafting table' element: a ruled vertical scale with a
    moving indicator, like an engineering angle gauge. `pct` is 0..1, where 1
    fills the gauge to the top. Reused live (CVA card) and in the summary
    (Spine Age) so it's the one visual motif that repeats across the app."""
    pct = clamp(pct, 0.0, 1.0)
    fill_h = pct * height
    return (
        f'<div class="pr-gauge"><div class="pr-gauge-track" style="height:{height}px">'
        f'<div class="pr-gauge-fill" style="height:{fill_h:.0f}px"></div>'
        f'<div class="pr-gauge-marker" style="bottom:{fill_h:.0f}px"></div>'
        f'</div><div class="pr-gauge-labels" style="height:{height}px">'
        f'<span>{top_label}</span><span>{bottom_label}</span></div></div>'
    )


def _card(label, value, unit, sub, color, bar=None, wide=False, tip=None, extra=""):
    bar_html = ""
    if bar is not None:
        bar_html = (f'<div class="pr-bar"><i style="width:{clamp(bar,0,1)*100:.0f}%;'
                    f'background:{color}"></i></div>')
    u = f'<span class="u">{unit}</span>' if unit else ""
    tip_html = (f'<span class="pr-tip"><span class="pr-tip-ic">?</span>'
               f'<span class="pr-tip-box">{tip}</span></span>') if tip else ""
    return (f'<div class="pr-card{" wide" if wide else ""}">'
            f'<div class="pr-label-row"><div class="pr-label">{label}</div>{tip_html}</div>'
            f'<div class="pr-value" style="color:{color}">{value}{u}</div>'
            f'<div class="pr-sub">{sub}</div>{bar_html}{extra}</div>')


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
                  STATUS_TEXT.get(e.status, ""), color, bar=e.score / 100.0, tip=TIPS["posture"])
    html += _card("Fatigue", e.fatigue_label.upper(), "", fat_sub, fat_c, bar=e.fatigue / 100.0,
                  tip=TIPS["fatigue"])
    html += _card("Streak", fmt_clock(e.streak_s), "", streak_sub,
                  C_GOOD if e.streak_s > 0 else MUTED, tip=TIPS["streak"])
    html += _card("Trend", tr_v, "", tr_s, tr_c, tip=TIPS["trend"])
    cva_gauge = ""
    if e.metrics_ema:
        cva_pct = (clamp(e.metrics_ema["cva"], 30.0, 60.0) - 30.0) / 30.0
        cva_gauge = f'<div style="margin-top:10px">{render_gauge(cva_pct, "60°", "30°", height=64)}</div>'
    html += _card("CVA", f"{e.metrics_ema['cva']:.0f}" if e.metrics_ema else "—", "°",
                  f"Baseline {e.baseline['cva']:.0f}°" if e.baseline else "Not calibrated", ACCENT,
                  tip=TIPS["cva"], extra=cva_gauge)
    html += _card("Next break", nb, "", f"{e.breaks_taken} taken · day streak {daily_streak}",
                  ACCENT, tip=TIPS["next_break"])
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
    if e.framing:
        k, msg, c = "FRAMING", e.framing, C_WATCH
    elif e.status == "IDLE":
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
        f'<span style="color:{TEXT}">{msg}</span></div>',
        unsafe_allow_html=True,
    )


def _humanoid_svg(part_style: str) -> str:
    return (f'<svg class="pr-stretch-fig" viewBox="0 0 120 120">'
            f'<line x1="30" y1="82" x2="90" y2="82" stroke="{MUTED}" stroke-width="6" stroke-linecap="round"/>'
            f'<g class="part" style="{part_style}">'
            f'<circle cx="60" cy="50" r="20" fill="none" stroke="{ACCENT}" stroke-width="6"/>'
            f'<line x1="60" y1="70" x2="60" y2="82" stroke="{ACCENT}" stroke-width="6"/>'
            f'</g></svg>')


def _shoulders_svg(part_style: str) -> str:
    return (f'<svg class="pr-stretch-fig" viewBox="0 0 120 120">'
            f'<circle cx="60" cy="38" r="14" fill="none" stroke="{MUTED}" stroke-width="5"/>'
            f'<g class="part" style="{part_style}">'
            f'<line x1="25" y1="70" x2="95" y2="70" stroke="{ACCENT}" stroke-width="7" stroke-linecap="round"/>'
            f'</g></svg>')


def _eye_svg(part_style: str) -> str:
    return (f'<svg class="pr-stretch-fig" viewBox="0 0 120 120">'
            f'<path d="M20,60 Q60,30 100,60 Q60,90 20,60 Z" fill="none" stroke="{MUTED}" stroke-width="4"/>'
            f'<circle class="part" cx="60" cy="60" r="12" fill="{ACCENT}" style="{part_style}"/>'
            f'</svg>')


def _hand_svg(part_style: str) -> str:
    return (f'<svg class="pr-stretch-fig" viewBox="0 0 120 120">'
            f'<line x1="20" y1="60" x2="70" y2="60" stroke="{MUTED}" stroke-width="6" stroke-linecap="round"/>'
            f'<g class="part" style="{part_style}">'
            f'<circle cx="85" cy="60" r="16" fill="none" stroke="{ACCENT}" stroke-width="6"/>'
            f'<line x1="95" y1="50" x2="105" y2="42" stroke="{ACCENT}" stroke-width="5" stroke-linecap="round"/>'
            f'</g></svg>')


_STRETCH_SVG_BUILDERS = {"head": _humanoid_svg, "shoulders": _shoulders_svg,
                          "eye": _eye_svg, "hand": _hand_svg}

# Every icon reuses one of four simple, iconographic shapes; each exercise
# gets its own motion via CSS custom properties on a shared set of keyframes
# (see pr-tilt / pr-slide / pr-scale / pr-spin in the stylesheet).
STRETCH_ANIM = {
    "chin_tuck":    ("head", "animation:pr-slide 2s ease-in-out infinite; --x0:0px; --x1:-14px; --y0:0px; --y1:0px;"),
    "neck_tilt":    ("head", "animation:pr-tilt 2.4s ease-in-out infinite; --r0:-16deg; --r1:16deg; transform-origin:60px 82px;"),
    "neck_rotate":  ("head", "animation:pr-tilt 2.6s ease-in-out infinite; --r0:-28deg; --r1:28deg; transform-origin:60px 82px;"),
    "chin_chest":   ("head", "animation:pr-slide 2.2s ease-in-out infinite; --y0:0px; --y1:14px; --x0:0px; --x1:0px;"),
    "shoulder_roll":("shoulders", "animation:pr-spin 2.4s linear infinite; transform-origin:60px 70px;"),
    "blade_squeeze":("shoulders", "animation:pr-scale 2s ease-in-out infinite; --s0:1; --s1:0.7; transform-origin:60px 70px;"),
    "cross_arm":    ("hand", "animation:pr-tilt 2.2s ease-in-out infinite; --r0:-10deg; --r1:35deg; transform-origin:70px 60px;"),
    "back_round":   ("shoulders", "animation:pr-scale 2.4s ease-in-out infinite; --s0:1; --s1:1.18; transform-origin:60px 70px;"),
    "cat_cow":      ("shoulders", "animation:pr-slide 2.6s ease-in-out infinite; --y0:-6px; --y1:6px; --x0:0px; --x1:0px;"),
    "eye_far":      ("eye", "animation:pr-scale 2.2s ease-in-out infinite; --s0:1; --s1:0.4; transform-origin:60px 60px;"),
    "eye_blink":    ("eye", "animation:pr-scale 1.6s ease-in-out infinite; --s0:1; --s1:0.15; transform-origin:60px 60px;"),
    "eye_circle":   ("eye", "animation:pr-spin 2.4s linear infinite; transform-origin:60px 60px;"),
    "wrist_flex":   ("hand", "animation:pr-tilt 2s ease-in-out infinite; --r0:-20deg; --r1:20deg; transform-origin:85px 60px;"),
    "wrist_circle": ("hand", "animation:pr-spin 2s linear infinite; transform-origin:85px 60px;"),
    "finger_fist":  ("hand", "animation:pr-scale 1.8s ease-in-out infinite; --s0:1; --s1:0.6; transform-origin:85px 60px;"),
}


def render_stretch_icon(svg_key: str) -> str:
    family, style = STRETCH_ANIM.get(svg_key, ("head", ""))
    return _STRETCH_SVG_BUILDERS[family](style)


def render_break(ph, e: PostureEngine, now: float):
    total = max(e.break_total, 1.0)
    left = max(0.0, (e.break_until or now) - now)
    elapsed = clamp(total - left, 0.0, total)

    items = e.break_exercises or [STRETCHES[0]]
    n = len(items)
    seg = total / n
    idx = min(int(elapsed // seg), n - 1)
    into = elapsed - idx * seg
    name, group, how, svg_key = items[idx]

    check_html = ""
    if idx > 0 and into < 0.6:
        prev_name = items[idx - 1][0]
        check_html = (f'<div class="pr-check-mark"><span class="pr-check-glyph">&#10003;</span> '
                      f'{prev_name} done</div>')

    ph.markdown(
        f'<div class="pr-break"><div class="n">{int(left)+1}</div>'
        f'{render_stretch_icon(svg_key)}'
        f"<h3>{name}</h3><p>{how}</p>"
        f'{check_html}'
        f'<p style="margin-top:10px;font-size:.72rem;letter-spacing:.18em;color:#4E5F78">'
        f"MOVE {idx+1} OF {n} · {group.upper()} · SCORING PAUSED</p></div>",
        unsafe_allow_html=True,
    )


def friendly_date(iso_str: str) -> str:
    try:
        d = datetime.fromisoformat(iso_str).date()
    except Exception:
        return iso_str.split("T")[0]
    delta = (date.today() - d).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Yesterday"
    if 2 <= delta <= 6:
        return d.strftime("%A")
    return f"{d.strftime('%b')} {d.day}"


def score_color(score: float) -> str:
    return STATUS_COLORS[status_for(score)]


def render_history(store: dict) -> None:
    sessions = store.get("sessions", [])
    if not sessions:
        return
    best_avg = max(s["avg_score"] for s in sessions)
    best_age = min(s["spine_age"] for s in sessions)

    st.markdown(
        '<div class="pr-label" style="margin:22px 0 8px">SESSION HISTORY · STORED ON THIS MACHINE ONLY</div>',
        unsafe_allow_html=True,
    )
    strip = (
        '<div class="pr-hist-strip">'
        f'<div class="pr-hist-stat"><b>{len(sessions)}</b><span>Sessions</span></div>'
        f'<div class="pr-hist-stat"><b>{best_avg:.0f}</b><span>Best avg score</span></div>'
        f'<div class="pr-hist-stat"><b>{best_age}</b><span>Best spine age</span></div>'
        f'<div class="pr-hist-stat"><b>{store.get("daily_streak",0)}</b><span>Day streak</span></div>'
        '</div>'
    )
    rows = []
    for s in reversed(sessions[-8:]):
        c = score_color(s["avg_score"])
        rows.append(
            '<div class="pr-hist-row">'
            f'<div class="pr-hist-date">{friendly_date(s["at"])}</div>'
            f'<div class="pr-hist-mins">{s["minutes"]:.0f} min</div>'
            f'<div class="pr-hist-bar"><i style="width:{clamp(s["avg_score"],0,100):.0f}%;'
            f'background:{c}"></i></div>'
            f'<div class="pr-hist-score" style="color:{c}">{s["avg_score"]:.0f}</div>'
            f'<div class="pr-hist-age">age {s["spine_age"]}</div>'
            '</div>'
        )
    st.markdown(strip + "".join(rows), unsafe_allow_html=True)


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
    ss.setdefault("cap_index", None)
    ss.setdefault("store", load_store())
    ss.setdefault("share_pngs", {})
    ss.setdefault("share_style", "minimal")


def start_session():
    e = st.session_state.engine
    e.hard_reset()
    e.session_start = time.time()
    st.session_state.summary = None
    st.session_state.share_pngs = {}
    st.session_state.running = True


def stop_session():
    ss = st.session_state
    e = ss.engine
    ss.running = False
    if e.session_start and e.score_n > 3:
        dur = time.time() - e.session_start
        age, label, note = e.spine_age()
        prev_sessions = list(ss.store.get("sessions", []))  # history *before* this session is added
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
            "prev_sessions": prev_sessions,
            "day_streak": ss.store.get("daily_streak", 0),
        }
    release_camera()


def recalibrate():
    st.session_state.engine.reset_calibration()


def sidebar() -> dict:
    ss = st.session_state
    sb = st.sidebar
    sb.markdown(
        f'<div style="font-family:ui-monospace,monospace;font-size:1.5rem;font-weight:700;'
        f'letter-spacing:-.03em;color:{TEXT}">POSTU<span style="color:{ACCENT}">Re:</span></div>'
        f'<div style="font-family:ui-monospace,monospace;font-size:.6rem;letter-spacing:.22em;'
        f'color:{MUTED};margin:4px 0 14px">SPINE TELEMETRY · OFFLINE</div>',
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
                            help="Higher = stricter scoring, so even small slips cost you points. "
                                 "Lower = more forgiving, so only real slouching counts.")
    break_min = sb.slider("Stretch break every (min)", 5, 90, 20, 1)

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
        f'<div style="font-family:ui-monospace,monospace;font-size:1.8rem;font-weight:700;color:{C_GOOD}">'
        f'{store.get("daily_streak",0)}<span style="font-size:.8rem;color:{MUTED}"> days</span></div>'
        f'<div style="font-size:.72rem;color:{MUTED}">Best {store.get("best_streak",0)} · '
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
            e.framing = m["_framing"] if m else None
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
                        cal_fail_msg = "Couldn't see your shoulders clearly - restarting calibration."
                        e.reset_calibration()

                banner_ph.markdown(
                    f'<div class="pr-banner" style="background:{hex_to_rgba(ACCENT,0.08)};'
                    f'border-color:{hex_to_rgba(ACCENT,0.35)}">'
                    f'<span class="k" style="background:{hex_to_rgba(ACCENT,0.16)};color:{ACCENT}">CALIBRATING</span>'
                    f'<span style="color:{TEXT}">Sit the way you want to sit for the next hour. '
                    f'Everything after this is measured against this exact posture.</span></div>',
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
                    cv2.rectangle(ov, (0, 0), (w, h), hex_to_bgr(BG_0), -1)
                    cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)
                    draw_frame_hud(disp, "STRETCH BREAK - scoring paused", ACCENT)
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
                intensity = {"GOOD": 0.12, "WATCH": 0.80, "BAD": 1.45}.get(e.status, 0.0)
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
        f'<div class="pr-banner" style="background:{hex_to_rgba(ACCENT,0.08)};border-color:{hex_to_rgba(ACCENT,0.35)}">'
        f'<span class="k" style="background:{hex_to_rgba(ACCENT,0.16)};color:{ACCENT}">READY</span>'
        f'<span style="color:{TEXT}">Press <b>Start session</b> in the sidebar. '
        f'Five seconds of calibration, then the coaching is live.</span></div>',
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
                f'<div class="pr-card pr-ticks" style="height:100%"><div class="pr-label">{title}</div>'
                f'<div style="color:{TEXT};font-size:.92rem;line-height:1.5;margin-top:6px">{body}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:22px"></div>', unsafe_allow_html=True)
    render_history(store)


def render_summary(s: dict):
    ss = st.session_state
    age_color = C_GOOD if s["age"] <= 30 else C_WATCH if s["age"] <= 46 else C_BAD
    # gauge fills upward for a *better* (lower) age — best sits at the top
    age_pct = 1.0 - clamp((s["age"] - 18) / (79 - 18), 0.0, 1.0)
    gauge = render_gauge(age_pct, "BEST · 18", "WORST · 79", height=110)
    st.markdown(
        f'<div class="pr-hero pr-ticks"><div class="cap">SPINE AGE</div>'
        f'<div class="big" style="color:{age_color}">{s["age"]}</div>'
        f'<h2>{s["label"]}</h2>'
        f'<p style="color:var(--graphite);max-width:620px;margin:8px auto 0">{s["note"]}</p>'
        f'<p class="pr-explain">This isn\'t a literal age prediction — it\'s a wear score for '
        f'this one session, where lower is always better. Even a genuinely perfect session lands '
        f'in the low-to-mid 20s, so a young, healthy person seeing a number around there means '
        f'they did almost everything right, not that something is wrong.</p>'
        f'<div style="margin-top:20px;display:flex;justify-content:center">{gauge}</div>'
        f'</div>',
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

    st.markdown('<div class="pr-label" style="margin:26px 0 8px">SHARE CARD</div>', unsafe_allow_html=True)
    style = st.radio("Share card style", list(STYLE_LABELS.keys()),
                     format_func=lambda k: STYLE_LABELS[k], horizontal=True,
                     key="share_style", label_visibility="collapsed")

    if style not in ss.share_pngs:
        try:
            ss.share_pngs[style] = build_share_card(
                ss.engine, s["duration"] / 60.0,
                s.get("prev_sessions", []), s.get("day_streak", 0), style,
            )
        except Exception:
            ss.share_pngs[style] = b""
    png = ss.share_pngs.get(style, b"")

    c1, c2 = st.columns([1, 1])
    with c1:
        if png:
            st.image(png, width=260 if style == "story" else 340)
            st.download_button("Download share card", png,
                               file_name=f"posture-spine-age-{s['age']}-{style}.png", mime="image/png")
        else:
            st.info("Couldn't render the share card for this session.")
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
