"""
THE BRIDGE — a local FastAPI + WebSocket server standing in for Streamlit's
`run_session()` / `main()`. Same engine, same camera/MediaPipe pipeline, same
drawing — the only thing that changed is *how the result reaches the UI*:
instead of `st.markdown(html)` on a rerun-free loop, this pushes plain JSON
state + a base64 JPEG frame over a WebSocket at ~20Hz, and the React
frontend renders its own cards from that JSON.

Runs on 127.0.0.1 only — never exposed to the network. Electron spawns this
process on launch and talks to it over localhost; the end user never sees a
terminal or types a command.
"""
import asyncio
import base64
import json
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .audio import chime_wav_bytes
from .camera import CameraManager, load_face, load_pose
from .constants import ACCENT, BG_0, CAP_W, C_GOOD, C_WATCH, STATUS_COLORS, STATUS_TEXT, TARGET_FPS
from .drawing import draw_calibration_ui, draw_frame_hud, draw_skeleton, to_b64_jpeg
from .engine import PostureEngine
from .geometry import extract_face, extract_metrics
from .share_card import STYLE_LABELS, build_share_card, person_bbox
from .store import init_db, load_store, register_day, save_session, save_user_stats
from .suggestions import pick_suggestion
from .utils import fmt_clock, hex_to_bgr


DEFAULT_SETTINGS = {
    "sensitivity": 1.0,
    "break_min": 20,
    "ambient": True,
    "audio": True,
    "audio_volume": 70,
    "fatigue": True,
    "skeleton": True,
    "show_angle": True,
    "snapshot": True,
    "cam_index": 0,
    "complexity": 1,
    "face_every": 2,
}


class SessionServer:
    """Owns the single live PostureEngine + camera for this desktop app.

    One user, one session at a time — mirrors the Streamlit app's
    st.session_state, just as a plain object instead of a framework hook.
    """

    def __init__(self):
        self.engine = PostureEngine()
        self.camera = CameraManager()
        self.db_error = init_db()
        self.store = load_store()
        self.settings = dict(DEFAULT_SETTINGS)
        self.running = False
        self.summary: Optional[dict] = None
        self.end_chime_played = False

        self._lock = threading.Lock()
        self._latest_frame_b64 = ""
        self._latest_status = "IDLE"
        self._latest_fps = 0.0
        self._latest_state: dict = {}
        self._pending_chime: Optional[dict] = None  # {"kind": ..., "wav_b64": ...}
        self._thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._cal_fail_msg = ""

    # -- controls, called from the WebSocket handler -------------------------
    def start_session(self):
        if self.running:
            return
        self.engine.hard_reset()
        self.engine.session_start = time.time()
        self.summary = None
        self.end_chime_played = False
        self.running = True
        self._stop_flag.clear()
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop_session(self):
        if not self.running:
            return
        self._stop_flag.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.running = False
        self._finalize_summary()
        if not self.end_chime_played and self.settings.get("audio"):
            self._queue_chime("end")
        self.end_chime_played = True
        self.camera.release_camera()

    def recalibrate(self):
        self.engine.reset_calibration()

    def update_settings(self, patch: dict):
        self.settings.update({k: v for k, v in patch.items() if k in DEFAULT_SETTINGS})

    def build_card(self, style: str) -> bytes:
        prev_sessions = (self.summary or {}).get("prev_sessions", [])
        day_streak = self.store.get("daily_streak", 0)
        hero_jpg = (self.summary or {}).get("hero_jpg") or self.engine.hero_jpg
        hero_bbox = (self.summary or {}).get("hero_bbox") or self.engine.hero_bbox
        minutes = (self.summary or {}).get("duration", 0.0) / 60.0
        return build_share_card(self.engine, minutes, prev_sessions, day_streak, style, hero_jpg, hero_bbox)

    # -- background capture/inference/scoring loop ----------------------------
    def _finalize_summary(self):
        e = self.engine
        if not (e.session_start and e.score_n > 3):
            return
        dur = time.time() - e.session_start
        age, label, note = e.spine_age()
        prev_sessions = list(self.store.get("sessions", []))
        total_t = sum(e.time_in.values()) or 1.0
        slouch_pct = e.time_in["BAD"] / total_t * 100.0
        drowsy_pct = (e.fatigue_sum / e.fatigue_n) if e.fatigue_n else 0.0
        if dur >= 60:
            self.store = register_day(self.store)
            save_user_stats(self.store)
            save_session({
                "at": datetime.now().isoformat(timespec="seconds"),
                "minutes": round(dur / 60, 1),
                "avg_score": round(e.avg_score(), 1),
                "spine_age": age,
            })
            self.store = load_store()
        self.summary = {
            "duration": dur, "avg": e.avg_score(), "age": age, "label": label, "note": note,
            "time_in": dict(e.time_in), "best_streak": e.best_streak_s, "recoveries": e.recoveries,
            "bonus": e.bonus, "blinks": e.blinks, "yawns": e.yawns, "microsleeps": e.microsleeps,
            "blink_rate": e.blink_rate, "breaks": e.breaks_taken,
            "baseline_jpg": e.baseline_jpg, "current_jpg": e.current_jpg, "hero_jpg": e.hero_jpg,
            "hero_bbox": e.hero_bbox,
            "prev_sessions": prev_sessions, "day_streak": self.store.get("daily_streak", 0),
            "slouch_pct": slouch_pct, "drowsy_pct": drowsy_pct,
            "suggestion": pick_suggestion(slouch_pct, drowsy_pct),
        }

    def _queue_chime(self, kind: str):
        try:
            wav = chime_wav_bytes(kind, self.settings.get("audio_volume", 70) / 100.0)
            with self._lock:
                self._pending_chime = {"kind": kind, "wav_b64": base64.b64encode(wav).decode("ascii")}
        except Exception:
            pass

    def _capture_loop(self):
        e = self.engine
        cfg = self.settings
        cap = self.camera.get_camera(cfg["cam_index"])
        if cap is None or not cap.isOpened():
            with self._lock:
                self._latest_status = "ERROR"
            self.running = False
            return

        pose = load_pose(cfg["complexity"])
        face = load_face() if cfg["fatigue"] else None
        frame_i = 0
        errors = 0
        fps = 0.0
        self._cal_fail_msg = ""

        while not self._stop_flag.is_set():
            t0 = time.time()
            try:
                ok, frame = cap.read()
                if not ok or frame is None:
                    errors += 1
                    if errors > 40:
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
                state: dict = {"calibrated": e.calibrated}

                if not e.calibrated:
                    phase, prog = e.cal_phase(now)
                    if phase == "capture":
                        e.add_calibration(m, ear, mar)
                    if lms is not None and cfg["skeleton"]:
                        draw_skeleton(disp, lms, m, ACCENT, show_angle=False)
                    draw_calibration_ui(disp, phase, prog)
                    if self._cal_fail_msg:
                        draw_frame_hud(disp, self._cal_fail_msg, C_WATCH)

                    if phase == "capture" and prog >= 1.0:
                        if e.finish_calibration():
                            self._cal_fail_msg = ""
                            snap = frame.copy()
                            if lms is not None:
                                draw_skeleton(snap, lms, m, C_GOOD, cfg["show_angle"])
                            e.baseline_jpg = to_b64_jpeg(
                                cv2.resize(snap, (460, int(h * 460 / w))), 78)
                            e.schedule_breaks(now, cfg["break_min"])
                            if cfg["audio"]:
                                e._queue_chime("start", now, cooldown=0.0)
                        else:
                            self._cal_fail_msg = "Couldn't see your shoulders clearly - restarting calibration."
                            e.reset_calibration()

                    state.update({"phase": phase, "progress": prog, "cal_fail_msg": self._cal_fail_msg})
                    status = "IDLE"
                else:
                    on_break = e.check_break(now, cfg["break_min"])
                    e.update_posture(m, now, cfg["sensitivity"], on_break)
                    perclos = e.update_fatigue(ear, mar, now) if cfg["fatigue"] else 0.0
                    if not on_break:
                        e.maybe_alert(now, cfg["audio"])

                    status = e.status
                    color_hex = STATUS_COLORS.get(status, ACCENT)
                    if lms is not None and cfg["skeleton"]:
                        draw_skeleton(disp, lms, m, color_hex, cfg["show_angle"])

                    if on_break:
                        ov = disp.copy()
                        cv2.rectangle(ov, (0, 0), (w, h), hex_to_bgr(BG_0), -1)
                        cv2.addWeighted(ov, 0.55, disp, 0.45, 0, disp)
                        draw_frame_hud(disp, "STRETCH BREAK - scoring paused", ACCENT)
                    else:
                        hud = f"SCORE {e.score:.0f}   STREAK {fmt_clock(e.streak_s)}   {e.fatigue_label.upper()}"
                        draw_frame_hud(disp, hud, color_hex)

                    trend_label = ("RISING" if e.trend > 3 else "FALLING" if e.trend < -6 else "STABLE")
                    state.update({
                        "status": status,
                        "status_text": STATUS_TEXT.get(status, status),
                        "score": e.score,
                        "streak_s": e.streak_s,
                        "best_streak_s": e.best_streak_s,
                        "recoveries": e.recoveries,
                        "bonus": e.bonus,
                        "fatigue": e.fatigue,
                        "fatigue_label": e.fatigue_label,
                        "blink_rate": e.blink_rate,
                        "yawns": e.yawns,
                        "microsleeps": e.microsleeps,
                        "dry_eye": e.dry_eye,
                        "trend": e.trend,
                        "trend_label": trend_label,
                        "predicting": e.predicting,
                        "eta": e.eta,
                        "cva": e.metrics_ema["cva"] if e.metrics_ema else None,
                        "cva_baseline": e.baseline["cva"] if e.baseline else None,
                        "next_break_s": (max(0.0, e.next_break - now) if e.next_break else None),
                        "breaks_taken": e.breaks_taken,
                        "on_break": on_break,
                        "framing": e.framing,
                        "tip": e.tip,
                        "day_streak": self.store.get("daily_streak", 0),
                    })
                    if on_break:
                        left = max(0.0, (e.break_until or now) - now)
                        total = max(e.break_total, 1.0)
                        elapsed = max(0.0, min(total, total - left))
                        items = e.break_exercises or []
                        n = max(len(items), 1)
                        seg = total / n
                        idx = min(int(elapsed // seg), n - 1) if items else 0
                        into = elapsed - idx * seg
                        cur = items[idx] if items else None
                        state["break"] = {
                            "seconds_left": int(left) + 1,
                            "exercises": [{"name": x[0], "group": x[1], "how": x[2], "icon": x[3]} for x in items],
                            "index": idx,
                            "just_completed": bool(idx > 0 and into < 0.6),
                        }

                if e.calibrated and cfg["snapshot"] and now - e._snap_t > 1.5:
                    e._snap_t = now
                    e.current_jpg = to_b64_jpeg(cv2.resize(disp, (460, int(h * 460 / w))), 78)
                if e.calibrated and now - e._hero_t > 1.5:
                    e._hero_t = now
                    e.hero_jpg = to_b64_jpeg(disp, 88)
                    e.hero_bbox = person_bbox(lms, w, h) if lms is not None else None

                if e.pending_chime:
                    self._queue_chime(e.pending_chime)
                    e.pending_chime = None

                b64 = to_b64_jpeg(disp, 82)
                dt = time.time() - t0
                fps = (1.0 / dt) if fps == 0 else 0.15 * (1.0 / max(dt, 1e-3)) + 0.85 * fps
                with self._lock:
                    self._latest_frame_b64 = b64
                    self._latest_status = status
                    self._latest_fps = fps
                    self._latest_state = state

                frame_i += 1
                time.sleep(max(0.0, (1.0 / TARGET_FPS) - (time.time() - t0)))
            except Exception:
                errors += 1
                if errors > 25:
                    break
                time.sleep(0.05)

        self.running = False

    def snapshot(self) -> dict:
        with self._lock:
            frame_b64 = self._latest_frame_b64
            status = self._latest_status
            fps = self._latest_fps
            state = dict(self._latest_state)
            chime = self._pending_chime
            self._pending_chime = None
        return {
            "type": "tick",
            "running": self.running,
            "frame": frame_b64,
            "status": status,
            "fps": fps,
            "state": state,
            "chime": chime,
        }


# ----------------------------------------------------------------------------
# FastAPI app
# ----------------------------------------------------------------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local-only server (127.0.0.1); no network exposure
    allow_methods=["*"],
    allow_headers=["*"],
)
session = SessionServer()


@app.get("/api/ping")
def ping():
    return {"ok": True}


@app.get("/api/history")
def history():
    return session.store


@app.get("/api/spine_card_styles")
def spine_card_styles():
    return STYLE_LABELS


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    async def sender():
        while True:
            await asyncio.sleep(1.0 / TARGET_FPS)
            try:
                await websocket.send_text(json.dumps(session.snapshot()))
            except Exception:
                break

    send_task = asyncio.create_task(sender())
    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except Exception:
                continue
            mtype = msg.get("type")

            if mtype == "start_session":
                session.start_session()
            elif mtype == "stop_session":
                session.stop_session()
                await websocket.send_text(json.dumps({"type": "summary", "summary": session.summary}))
            elif mtype == "recalibrate":
                session.recalibrate()
            elif mtype == "update_settings":
                session.update_settings(msg.get("settings", {}))
            elif mtype == "get_history":
                await websocket.send_text(json.dumps({"type": "history", "store": session.store}))
            elif mtype == "get_spine_card":
                style = msg.get("style", "minimal")
                try:
                    png = session.build_card(style)
                    await websocket.send_text(json.dumps({
                        "type": "spine_card", "style": style,
                        "png_b64": base64.b64encode(png).decode("ascii"),
                    }))
                except Exception as exc:
                    await websocket.send_text(json.dumps({"type": "error", "message": str(exc)}))
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        if session.running:
            session.stop_session()


def main():
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="warning")


if __name__ == "__main__":
    main()
