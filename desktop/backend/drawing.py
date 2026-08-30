"""
DRAWING — a custom skeleton, ported verbatim from posture_app.py. MediaPipe's
default red/green dots look like a tutorial screenshot, and this has to look
good on a projector (and now, on a native window).
"""
import base64
from typing import Optional

import cv2
import numpy as np

from .constants import ACCENT, BG_0, CAL_CAPTURE_S, LINE, MUTED, TEXT
from .utils import clamp, hex_to_bgr


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
    from .geometry import EDGES  # local import avoids a circular import at module load

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
