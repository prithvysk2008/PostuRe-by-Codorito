"""GEOMETRY — pose landmarks to normalised posture metrics.

Ported verbatim from posture_app.py (zero Streamlit dependency there already).
Pure functions: MediaPipe landmarks in, plain dicts/tuples out.
"""
import math
from typing import List, Optional, Tuple

import numpy as np

from .utils import clamp

L_NOSE, L_EYE_L, L_EYE_R = 0, 2, 5
L_EAR_L, L_EAR_R = 7, 8
L_SH_L, L_SH_R = 11, 12
L_HIP_L, L_HIP_R = 23, 24

EDGES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


def extract_metrics(lms, w: int, h: int) -> Optional[dict]:
    """Turn 33 pose landmarks into scale-invariant posture metrics."""
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
        "chin": float((sh_mid[1] - nose[1]) / shoulder_w),
        "pitch": float((nose[1] - head[1]) / shoulder_w),
        # shoulder-width normalised (not frame-height) so a real slide down
        # the chair registers the same whether you're close to or far from
        # the camera
        "drop": float(sh_mid[1] / shoulder_w),
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
