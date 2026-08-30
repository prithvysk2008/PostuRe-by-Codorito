"""Small stateless helpers used across the backend — no Streamlit dependency."""
from typing import Tuple

from .constants import T_GOOD, T_WATCH


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
