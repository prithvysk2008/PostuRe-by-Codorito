"""
SPINE CARD — built with OpenCV so there are no font files to ship.

A cropped Polaroid-style photo of the person sits at the top; every stat
below it lives on its own tile with the same fixed teal-to-amber gradient
the live app uses (see TILE_GRADIENT_HEX / index.css's --grad-a/--grad-b) —
constant, not per-tile, so a screenshot of the app and this card read as
the same product.
"""
import base64
import math
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .constants import ACCENT, BG_0, C_BAD, C_GOOD, C_WATCH, LINE, MUTED, SURFACE, SURFACE_2, TEXT
from .engine import PostureEngine
from .utils import clamp, hex_to_bgr

STYLE_LABELS = {"minimal": "Minimal", "data": "Data-rich", "story": "Story (9:16)"}

# Same fixed teal-to-amber gradient as the frontend's --grad-a/--grad-b
# tokens (index.css) — constant, not per-tile, so the card and the live UI
# feel like the same app.
TILE_GRADIENT_HEX = ["#22C39F", "#D9A86C"]


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


def _multicolor_gradient(w: int, h: int, colors_hex: List[str], angle_deg: float = 120.0) -> np.ndarray:
    """Diagonal multi-stop gradient — the still-image version of the
    frontend's animated .card.gradient tile background."""
    colors = np.array([hex_to_bgr(c) for c in colors_hex], np.float32)
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = xx * dx + yy * dy
    proj -= proj.min()
    span = proj.max()
    proj = proj / span if span > 1e-6 else proj
    n = len(colors) - 1
    seg = np.clip(proj * n, 0, n - 1e-4)
    idx = seg.astype(np.int32)
    frac = (seg - idx)[..., None]
    c0, c1 = colors[idx], colors[np.clip(idx + 1, 0, n)]
    return (c0 * (1 - frac) + c1 * frac).astype(np.uint8)


def _rounded_mask(w: int, h: int, r: int) -> np.ndarray:
    r = min(r, w // 2, h // 2)
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(mask, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask


def _blit_rounded(card: np.ndarray, x0: int, y0: int, patch: np.ndarray, mask: np.ndarray) -> None:
    h, w = patch.shape[:2]
    region = card[y0:y0 + h, x0:x0 + w]
    m = (mask.astype(np.float32) / 255.0)[..., None]
    card[y0:y0 + h, x0:x0 + w] = (region.astype(np.float32) * (1 - m) + patch.astype(np.float32) * m).astype(np.uint8)


def _gradient_tile(card: np.ndarray, x0: int, y0: int, w: int, h: int, radius: int = 18) -> None:
    """Paint a rounded tile with the same fixed teal-to-amber gradient as
    every other tile — constant, matching the frontend's .card.gradient —
    darkened ~32% so text stays legible on top."""
    # prominent drop shadow, same treatment as the polaroid, so every tile
    # visibly sits above the card background
    shadow = np.zeros(card.shape[:2], np.uint8)
    cv2.rectangle(shadow, (x0 + 6, y0 + 10), (x0 + w + 6, y0 + h + 10), 255, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=14)
    alpha = (shadow.astype(np.float32) / 255.0 * 0.45)[..., None]
    card[:] = (card.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    grad = _multicolor_gradient(w, h, TILE_GRADIENT_HEX, angle_deg=135.0)
    dark = (grad.astype(np.float32) * 0.68).astype(np.uint8)
    _blit_rounded(card, x0, y0, dark, _rounded_mask(w, h, radius))
    cv2.rectangle(card, (x0, y0), (x0 + w - 1, y0 + h - 1), hex_to_bgr(LINE), 1, cv2.LINE_AA)


def _cv_donut(card: np.ndarray, cx: int, cy: int, r: int, thickness: int,
              segments: List[Tuple[float, str]]) -> None:
    cv2.circle(card, (cx, cy), r, hex_to_bgr("#081726"), thickness, cv2.LINE_AA)
    start = -90.0
    for pct, hex_color in segments:
        sweep = 360.0 * clamp(pct / 100.0, 0.0, 1.0)
        if sweep > 0.6:
            cv2.ellipse(card, (cx, cy), (r, r), 0, start, start + sweep,
                       hex_to_bgr(hex_color), thickness, cv2.LINE_AA)
        start += sweep


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


def person_bbox(lms, w: int, h: int, pad_frac: float = 0.30) -> Optional[Tuple[int, int, int, int]]:
    """A crop rect around just the visible person, from raw pose landmarks —
    there's no segmentation model in this stack, so this is a padded
    bounding box over confidently-visible landmarks rather than a pixel
    mask. Extra headroom above (hair/head) versus below (torso)."""
    xs = [lm.x * w for lm in lms if lm.visibility > 0.3]
    ys = [lm.y * h for lm in lms if lm.visibility > 0.3]
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    pad_x, pad_y = bw * pad_frac, bh * pad_frac
    x0 = max(0.0, x0 - pad_x)
    x1 = min(float(w), x1 + pad_x)
    y0 = max(0.0, y0 - pad_y * 1.6)
    y1 = min(float(h), y1 + pad_y * 0.6)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return None
    return (int(x0), int(y0), int(x1), int(y1))


def _cover_fit(img: np.ndarray, W: int, H: int) -> np.ndarray:
    """Resize + center-crop `img` to exactly WxH, like CSS object-fit: cover."""
    h, w = img.shape[:2]
    scale = max(W / w, H / h)
    nw, nh = max(int(math.ceil(w * scale)), W), max(int(math.ceil(h * scale)), H)
    interp = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(img, (nw, nh), interpolation=interp)
    x0, y0 = (nw - W) // 2, (nh - H) // 2
    return resized[y0:y0 + H, x0:x0 + W]


def _polaroid(card: np.ndarray, cx: int, y0: int, outer_w: int, hero_bgr: Optional[np.ndarray],
              bbox: Optional[Tuple[int, int, int, int]], caption: str, photo_aspect: float = 1.08) -> int:
    """Draws a Polaroid-framed crop of just the person, centered at `cx`
    starting at `y0`. Returns the y-coordinate just below the frame."""
    border_side = max(int(outer_w * 0.045), 10)
    border_bottom = max(int(outer_w * 0.16), 36)
    photo_w = outer_w - 2 * border_side
    photo_h = int(photo_w * photo_aspect)
    outer_h = photo_h + border_side + border_bottom
    x0 = cx - outer_w // 2

    # prominent, blurred drop shadow so the polaroid reads as sitting above
    # the card, not pasted flat onto it
    shadow = np.zeros(card.shape[:2], np.uint8)
    pad = 30
    cv2.rectangle(shadow, (x0 + pad // 2, y0 + pad), (x0 + outer_w + pad // 2, y0 + outer_h + pad), 255, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), sigmaX=26)
    alpha = (shadow.astype(np.float32) / 255.0 * 0.62)[..., None]
    card[:] = (card.astype(np.float32) * (1 - alpha)).astype(np.uint8)

    cv2.rectangle(card, (x0, y0), (x0 + outer_w, y0 + outer_h), (245, 248, 250), -1)

    cropped = None
    if hero_bgr is not None and hero_bgr.size:
        src = hero_bgr
        if bbox is not None:
            bx0, by0, bx1, by1 = bbox
            bx0, by0 = max(0, bx0), max(0, by0)
            bx1, by1 = min(hero_bgr.shape[1], bx1), min(hero_bgr.shape[0], by1)
            if bx1 - bx0 > 20 and by1 - by0 > 20:
                src = hero_bgr[by0:by1, bx0:bx1]
        cropped = _cover_fit(src, photo_w, photo_h)

    px0, py0 = x0 + border_side, y0 + border_side
    if cropped is not None:
        card[py0:py0 + photo_h, px0:px0 + photo_w] = cropped
    else:
        placeholder = _gradient_canvas(photo_w, photo_h, SURFACE_2, BG_0)
        card[py0:py0 + photo_h, px0:px0 + photo_w] = placeholder

    cv2.rectangle(card, (x0, y0), (x0 + outer_w, y0 + outer_h), (255, 255, 255), 2, cv2.LINE_AA)

    cap_y = y0 + outer_h - border_bottom // 2 + 6
    (cw, _), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_DUPLEX, 0.46, 1)
    cv2.putText(card, caption, (cx - cw // 2, cap_y), cv2.FONT_HERSHEY_DUPLEX, 0.46,
                hex_to_bgr(ACCENT), 1, cv2.LINE_AA)

    return y0 + outer_h


def build_share_card(engine: PostureEngine, minutes: float, prev_sessions: list,
                      day_streak: int, style: str = "minimal",
                      hero_jpg: Optional[str] = None,
                      hero_bbox: Optional[Tuple[int, int, int, int]] = None) -> bytes:
    """The Spine Card: a Polaroid-framed crop of just the person up top, then
    every stat on its own vivid gradient tile below it — the same visual
    language as the live app's stat cards."""
    age, label, note = engine.spine_age()
    avg = engine.avg_score()
    total_t = sum(engine.time_in.values()) or 1.0
    good_pct = engine.time_in.get("GOOD", 0.0) / total_t * 100.0
    watch_pct = engine.time_in.get("WATCH", 0.0) / total_t * 100.0
    bad_pct = engine.time_in.get("BAD", 0.0) / total_t * 100.0
    age_color = hex_to_bgr(C_GOOD if age <= 30 else C_WATCH if age <= 46 else C_BAD)
    badge = _share_badge(engine, total_t)
    compare = _share_compare(prev_sessions, avg, age)
    prev = prev_sessions[-1] if prev_sessions else None

    tall = style == "story"
    W, H = 1080, (1920 if tall else 1080)
    pad = 36
    card = _gradient_canvas(W, H, SURFACE_2, BG_0)
    _add_corner_glow(card, W, H, ACCENT)
    _add_grid_texture(card, spacing=44, hex_color=MUTED, alpha=0.04)

    def text_w(s, scale, thick=1, font=cv2.FONT_HERSHEY_DUPLEX):
        return cv2.getTextSize(s, font, scale, thick)[0][0]

    def fit_scale(s, max_w, scale, thick=1, min_scale=0.3):
        while scale > min_scale and text_w(s, scale, thick) > max_w:
            scale -= 0.02
        return scale

    def text(s, x, y, scale, col, thick=1, center=False, font=cv2.FONT_HERSHEY_DUPLEX):
        if center:
            x = x - text_w(s, scale, thick, font) // 2
        cv2.putText(card, s, (int(x), int(y)), font, scale, col, thick, cv2.LINE_AA)

    # -- header: brand + trust tag, directly on the card background --
    text("POSTURE:", pad, 40, 0.56, hex_to_bgr(TEXT), 2)
    text("SPINE CARD", pad, 64, 0.36, hex_to_bgr(ACCENT), 1)
    trust = "100% ON-DEVICE"
    text(trust, W - pad - text_w(trust, 0.4, 1), 40, 0.4, hex_to_bgr(MUTED), 1)

    # -- Polaroid: a tight crop of just the person, not the whole room --
    hero_bgr = None
    if hero_jpg:
        try:
            arr = np.frombuffer(base64.b64decode(hero_jpg), dtype=np.uint8)
            hero_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception:
            hero_bgr = None

    # story has ~1.8x the vertical room of the square styles — scale every
    # block up rather than just tacking one extra thin strip on the end.
    k = 1.0 if not tall else 1.35

    cx = W // 2
    polaroid_w = int(W * 0.40)
    y = _polaroid(card, cx, 86, polaroid_w, hero_bgr, hero_bbox, note[:28],
                  photo_aspect=1.08 if not tall else 1.5)
    y += int(24 * k)

    # -- hero tile: the spine-age number, biggest/boldest element on the card --
    hero_h = int(152 * k)
    hx0 = pad
    _gradient_tile(card, hx0, y, W - 2 * pad, hero_h)
    hcx = W // 2
    num_cx = hcx - int(W * 0.16)
    hy = y + int(34 * k)
    text("SPINE AGE", num_cx, hy, 0.46, (255, 255, 255), 1, center=True)
    big = str(age)
    big_scale = fit_scale(big, int(W * 0.3), 2.6 * k, thick=6)
    (bw, bh), _ = cv2.getTextSize(big, cv2.FONT_HERSHEY_DUPLEX, big_scale, 6)
    cv2.putText(card, big, (num_cx - bw // 2, hy + int(14 * k) + bh), cv2.FONT_HERSHEY_DUPLEX, big_scale,
                age_color, 6, cv2.LINE_AA)
    lbl_scale = fit_scale(label.upper(), int(W * 0.4), 0.62 * k, thick=2)
    text(label.upper(), num_cx, hy + int(14 * k) + bh + int(30 * k), lbl_scale, age_color, 2, center=True)
    # supporting line: session length + note, right side of the hero tile
    note_x = hcx + int(W * 0.05)
    note_w = (W - pad - 20) - note_x
    text(f"{minutes:.0f} MIN SESSION", note_x, hy, 0.4 * k, (235, 240, 242), 1)
    text(note, note_x, hy + int(28 * k), fit_scale(note, note_w, 0.4 * k), (215, 222, 225), 1)
    y += hero_h + int(20 * k)

    # -- three supporting tiles in a row: avg score / time split / streak --
    tile_gap = 16
    tile_w = (W - 2 * pad - 2 * tile_gap) // 3
    tile_h = int(195 * k)
    tx = pad
    for i in range(3):
        _gradient_tile(card, tx + i * (tile_w + tile_gap), y, tile_w, tile_h)

    # tile 1: avg score
    t1x = tx + tile_w // 2
    text("AVG SCORE", t1x, y + int(30 * k), 0.42, (255, 255, 255), 1, center=True)
    val = f"{avg:.0f}"
    vscale = fit_scale(val, tile_w - 30, 1.6 * k, thick=4)
    text(val, t1x, y + int(92 * k), vscale, (255, 255, 255), 4, center=True)
    if prev:
        diff = avg - prev["avg_score"]
        dtxt = f"{'+' if diff >= 0 else ''}{diff:.0f} vs last"
        dcol = C_GOOD if diff >= 0 else C_WATCH
    else:
        dtxt, dcol = "First session", MUTED
    text(dtxt, t1x, y + tile_h - 20, fit_scale(dtxt, tile_w - 20, 0.4), hex_to_bgr(dcol), 1, center=True)

    # tile 2: time split donut — a real "picture", not just numbers
    t2x0 = tx + tile_w + tile_gap
    t2cx = t2x0 + tile_w // 2
    text("TIME SPLIT", t2cx, y + int(30 * k), 0.42, (255, 255, 255), 1, center=True)
    ring_r = min(tile_w, tile_h) // 2 - 44
    ring_cy = y + int(46 * k) + ring_r
    _cv_donut(card, t2cx, ring_cy, ring_r, 14, [(good_pct, C_GOOD), (watch_pct, C_WATCH), (bad_pct, C_BAD)])
    leg_y = ring_cy + ring_r + 24
    labels = [("OK", good_pct, C_GOOD), ("WATCH", watch_pct, C_WATCH), ("BAD", bad_pct, C_BAD)]
    seg_w = tile_w // 3
    for i, (lbl, pct, hex_c) in enumerate(labels):
        chip = f"{pct:.0f}%"
        ccx = t2x0 + seg_w * i + seg_w // 2
        cv2.circle(card, (ccx - 18, leg_y - 5), 5, hex_to_bgr(hex_c), -1, cv2.LINE_AA)
        text(chip, ccx + 2, leg_y, 0.36, (255, 255, 255), 1, center=True)

    # tile 3: streak + achievement badge
    t3x0 = tx + 2 * (tile_w + tile_gap)
    t3cx = t3x0 + tile_w // 2
    text("DAY STREAK", t3cx, y + int(30 * k), 0.42, (255, 255, 255), 1, center=True)
    streak_txt = str(day_streak)
    sscale = fit_scale(streak_txt, tile_w - 30, 1.6 * k, thick=4)
    text(streak_txt, t3cx, y + int(92 * k), sscale, (255, 255, 255), 4, center=True)
    badge_scale = fit_scale(badge, tile_w - 24, 0.36, thick=1)
    text(badge, t3cx, y + tile_h - 20, badge_scale, (255, 255, 255), 1, center=True)

    y += tile_h + int(24 * k)

    # data/story styles: room (and reason) for one more line of context
    if style != "minimal":
        text(compare, cx, y, fit_scale(compare, W - 2 * pad, 0.5 * k), hex_to_bgr(TEXT), 1, center=True)
        y += int(34 * k)

    if tall:
        tagline = "A wear score for this session, not a literal age - lower is always better."
        text(tagline, cx, y, fit_scale(tagline, W - 2 * pad, 0.4), hex_to_bgr(MUTED), 1, center=True)
        y += int(40 * k)

        cva_now = engine.metrics_ema["cva"] if engine.metrics_ema else None
        cva_base = engine.baseline["cva"] if engine.baseline else None
        if cva_now is not None and cva_base is not None:
            cva_h = tile_h
            _gradient_tile(card, pad, y, W - 2 * pad, cva_h)
            text("CVA vs BASELINE", cx, y + int(36 * k), 0.5 * k, (255, 255, 255), 1, center=True)
            cva_txt = f"{cva_base:.0f} deg -> {cva_now:.0f} deg"
            text(cva_txt, cx, y + int(100 * k),
                 fit_scale(cva_txt, W - 2 * pad - 40, 1.0 * k, thick=3), (255, 255, 255), 3, center=True)
            y += cva_h + int(24 * k)

    # Anchored to whatever the flow above ended at, never to a fixed offset
    # from H — a fixed anchor collided with the compare line on the square
    # (data) canvas, since that style has no CVA tile to push it down first.
    foot_y = y + int(24 * k)
    text("PostuRe: - nothing leaves this laptop", cx, foot_y, 0.4, hex_to_bgr(MUTED), 1, center=True)

    ok, buf = cv2.imencode(".png", card)
    return buf.tobytes() if ok else b""
