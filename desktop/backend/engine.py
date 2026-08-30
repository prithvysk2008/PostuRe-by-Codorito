"""
THE ENGINE — ported verbatim from posture_app.py's PostureEngine class.

Confirmed zero Streamlit dependency in the source (grepped the full class
body before extracting). One object holding every piece of session state:
calibration, scoring, streaks, prediction, fatigue and break scheduling.
"""
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import CAL_CAPTURE_S, CAL_MIN_SAMPLES, CAL_WARMUP_S, T_WATCH
from .metrics import METRIC_KEYS, METRICS
from .stretches import BREAK_DURATION_S, pick_stretches
from .utils import clamp, status_for


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
        self.hero_jpg: Optional[str] = None  # full-res frame, used as the Spine Card background
        self.hero_bbox: Optional[tuple] = None  # (x0, y0, x1, y1) person crop, same frame as hero_jpg
        self._hero_t = 0.0

        # alerts
        self.last_chime = {"watch": 0.0, "bad": 0.0, "predict": 0.0, "fatigue": 0.0,
                           "break": 0.0, "recover": 0.0, "start": 0.0, "end": 0.0}
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
            self.break_total = BREAK_DURATION_S
            self.break_exercises = pick_stretches()
            self.break_until = now + BREAK_DURATION_S
            self._queue_chime("break", now, cooldown=0.0)
            return True
        return False

    def start_break_now(self, now: float):
        self.break_total = BREAK_DURATION_S
        self.break_exercises = pick_stretches()
        self.break_until = now + BREAK_DURATION_S
        self._queue_chime("break", now, cooldown=0.0)

    # -- alerts --------------------------------------------------------------
    def _queue_chime(self, kind: str, now: float, cooldown: float):
        if now - self.last_chime.get(kind, 0.0) >= cooldown:
            self.last_chime[kind] = now
            self.pending_chime = kind

    def maybe_alert(self, now: float, audio_on: bool):
        if not audio_on:
            self.pending_chime = None
            return
        # posture nudges — short cooldowns so a sustained slump/drift keeps
        # getting reminders instead of alerting once and going quiet
        if self.status == "BAD":
            self._queue_chime("bad", now, cooldown=14.0)
        elif self.status == "WATCH":
            self._queue_chime("watch", now, cooldown=22.0)
        elif self.predicting:
            self._queue_chime("predict", now, cooldown=25.0)
        # fatigue nudge is independent of posture status, so it can fire in
        # the same stretch as (and doesn't wait on) a posture alert
        if self.fatigue_label in ("Drowsy", "Critical"):
            self._queue_chime("fatigue", now, cooldown=25.0)

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
