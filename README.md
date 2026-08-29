# PostuRe: — setup, tuning and demo guide

Webcam posture + fatigue coach. MediaPipe Pose, MediaPipe Face Mesh, OpenCV, NumPy, Streamlit.
Runs entirely on the laptop. No cloud, no API keys, no database.

```
posture_app.py        the whole app
check_setup.py        run this BEFORE the demo — verifies libs, camera, FPS
requirements.txt      pinned versions that are known to work together
.streamlit/config.toml  dark theme so there is no white flash on load
run.sh / run.bat      one-click launchers
posture_data.json     created on first run — local streak history
```

---

## 1. Install (do this once, not on demo day)

**Python 3.10 or 3.11.** MediaPipe supports 3.9–3.12. Check with `python --version`.

macOS / Linux:
```bash
cd posture-app
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
cd posture-app
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Then verify:
```bash
python check_setup.py
```
It checks every import, finds your camera, and benchmarks the real pipeline FPS. Fix anything
it flags now, not five minutes before you present.

Run the app:
```bash
streamlit run posture_app.py
```
It opens at `http://localhost:8501`. Grant camera permission when the OS asks.

---

## 2. First session, step by step

1. Sidebar → **Start session**.
2. Two seconds of "GET SET", then a five-second ring countdown. **Sit the way you want to sit
   for the next hour** — this frame becomes your baseline and everything is measured against it.
3. Live coaching starts. Score, fatigue, streak, trend, CVA and next-break cards on the right;
   the telemetry strip under the video shows the last 120 seconds with the 80 / 60 bands.
4. Slouch. The border and screen edge shift green → amber → red, a soft chime plays, and the
   banner tells you which specific thing to fix.
5. Sit up within 12 seconds → **recovery save**, bright chime, +25 bonus, streak survives.
6. Sidebar → **Stop session** → Spine Age summary with a downloadable share card.

**Recalibrate baseline** re-runs the 5-second capture without losing your session stats. Use it
if you move the laptop or change chairs.

---

## 3. How the scoring actually works (for the judges' questions)

Six metrics come out of the 33 pose landmarks, each normalised so it is scale-invariant:

| Metric | What it is | Weight |
|---|---|---|
| `neck` | vertical gap between ear-line and shoulder-line ÷ shoulder width | 0.28 |
| `cva` | craniovertebral angle: ear→shoulder vector vs horizontal | 0.18 |
| `pitch` | nose below the ear-line ÷ shoulder width (looking down) | 0.18 |
| `drop` | shoulder height as a fraction of frame height (sliding down the chair) | 0.14 |
| `face` | inter-eye distance ÷ shoulder width (head creeping toward the screen) | 0.14 |
| `tilt` | shoulder line rotation in degrees (leaning on one arm) | 0.08 |

Dividing by shoulder width is the important part: if you simply lean closer to the camera,
every pixel measurement grows together and the ratios do not move, so the score does not lie.

Each metric is scored **against your own calibrated baseline**, not a textbook number:

```
deviation = (current − baseline) in the unhealthy direction
tolerance = max(2.5 × MAD of your calibration samples, metric floor)
penalty   = clamp((deviation − tolerance) / span, 0, 1)
score     = 100 × (1 − Σ weight × penalty)
```

MAD (median absolute deviation) is measured during your own calibration, so a person who
fidgets gets a wider dead-zone automatically. Median, not mean, so one bad calibration frame
cannot poison the baseline.

Stability comes from three layers: EMA on the raw metrics (α 0.35), EMA on the score (α 0.25),
and 7-frame hysteresis on the status, so the border never strobes.

**Prediction:** a least-squares fit over the last 45 seconds of scores gives the slope in
points per minute. If it is steeper than −6 pts/min while you are still above 60, the app
computes the seconds until you cross into slouch and warns you then — before the slump.

**Fatigue:** EAR from the 6-point eye landmarks, thresholded at 72% of *your* calibrated open-eye
EAR rather than a fixed 0.21, so it works with different eye shapes and glasses. A closure of
0.05–0.7s is a blink; over 1.1s is a micro-sleep. PERCLOS is the fraction of the last 60 seconds
with eyes closed. MAR over 0.8s is a yawn. Blink rate under 10/min after 90 seconds flags dry-eye risk.

**Spine Age:** `22 + (100 − avg score) × 0.55 + red_fraction × 18 + fatigue/100 × 8 − saves × 0.4`,
clamped to 18–79.

---

## 4. Tuning for your demo room

| Symptom | Fix |
|---|---|
| Score barely moves when you slouch | Sensitivity slider → 1.4–1.6 |
| Score drops when you breathe | Sensitivity → 0.7–0.8, and recalibrate while sitting *very* still |
| Under 12 FPS | Sidebar → Performance → Pose model **Fast (0)**, Face mesh every **3** frames |
| Still slow | Uncheck "Evolution snapshot" — it re-encodes an extra JPEG every 1.5s |
| Want the alert states fast on stage | Stretch break every **5** min so the break demo fires during the pitch |

Stage lighting: sit with light **in front of you**. A bright window behind you turns you into a
silhouette and pose detection degrades. Frame yourself so the camera sees your head and **both
shoulders** — the app needs shoulders, not your face.

---

## 5. Troubleshooting

**Camera does not open / black frame**
Close Zoom, Meet, Teams, OBS, and any other browser tab holding the camera. Only one process
gets it. Then try camera index 1 or 2 in the sidebar. On macOS: System Settings → Privacy &
Security → Camera → enable Terminal. On Windows: Settings → Privacy → Camera → allow desktop apps.

**`ImportError: numpy.core.multiarray failed to import`**
NumPy 2.x against MediaPipe 0.10.x. `pip install numpy==1.26.4`

**`protobuf` / `descriptor` errors**
`pip install protobuf==4.25.3`

**`No matching distribution found for mediapipe`**
Python is 3.13+ or 32-bit. Install Python 3.11 64-bit and rebuild the venv.

**No sound**
Browsers block autoplay until you interact with the page. Clicking Start counts, but if you
never clicked in the tab, click anywhere once. Check the tab is not muted.

**The app feels laggy after clicking a checkbox**
Every sidebar change restarts the loop. The camera and models are cached, so it recovers in
well under a second. Do not fiddle with the sidebar mid-pitch — set it up before you go on.

**Status stuck on NO SUBJECT**
Your shoulders are out of frame or the shoulder landmarks are below 0.4 confidence. Sit back,
raise the laptop, add front light.

---

## 6. Demo script (about 3 minutes)

1. **Open on the idle screen.** "Everything you're about to see runs on this laptop. No server,
   no upload. I could turn the wifi off." Turn the wifi off. It keeps working.
2. **Start session, calibrate.** "It doesn't judge you against a textbook spine. Five seconds,
   and *your* posture becomes the reference."
3. **Sit normally for ten seconds.** Point at the telemetry strip filling in. Green.
4. **Slowly slouch.** Do not collapse — drift. Let the judges watch the trend card go FALLING
   and the banner say *slump predicted in ~30s*. "This is the part that matters. It's warning me
   before I'm slouching, not after."
5. **Complete the slouch.** Red glow on the screen edges, soft chime. "No pop-up. Nothing to
   dismiss. You feel it in your peripheral vision."
6. **Sit up fast.** Recovery save fires, bonus lands, streak survives. "It rewards the correction,
   not just the posture."
7. **Point at the snapshot.** Baseline versus now, side by side.
8. **Stop.** Spine Age lands. "Fifty-two. That's the number people actually share."
9. **Close on the fatigue card**: blinks per minute, yawns, micro-sleeps. "Same camera, same
   frame, no extra hardware."

Have a **second person ready** to sit down and try it. A judge seeing it calibrate to a body it
has never seen, on stage, is worth more than any slide.

### Questions you will get

**"Is this medically valid?"** The craniovertebral angle is the standard clinical measure of
forward head posture. What we compute is a 2D projection of it from a single webcam, so we treat
it as a *relative* signal against your own calibrated baseline rather than a diagnostic number.
That is also why the score is a deviation, not an absolute.

**"What about a side view?"** A side view gives a truer CVA. The app works front-on because that
is where a laptop camera is, and it compensates with five other normalised metrics.

**"Why not deep learning?"** MediaPipe *is* the learned model — 33 keypoints and 468 face
landmarks. On top of that we deliberately use interpretable geometry, because a hackathon judge
and a physiotherapist can both audit a ratio, and because it needs zero training data from you.

**"Privacy?"** Frames never leave the process. Nothing is written to disk except a JSON file of
scores and timestamps, and you can delete it. That is a design constraint, not a feature.

**"Does it work for different bodies?"** Every metric is divided by that person's shoulder width
and compared to their own baseline, and the dead-zone is derived from their own calibration
spread. There is no population average anywhere in the scoring.

---

## 7. Split the work three ways

- **You (ECE):** the sensing layer — calibration, metric extraction, scoring, thresholds.
  You own the geometry table in section 3; that is the technical core of the pitch.
- **CSE/AI teammate:** fatigue engine and prediction — EAR/MAR tuning, PERCLOS window, the
  regression and its thresholds. Have them collect ten minutes of their own footage and tune
  `ear_thresh` and the trend threshold on it.
- **Mechanical teammate:** the health argument and the demo. Cervical load numbers for the
  30kg framing, the stretch content, and running the on-stage laptop so you can talk.

## 8. If you have spare time

Ranked by demo impact per hour of work:

1. A second body on stage (zero code, huge credibility).
2. Persist per-day average scores and draw a 7-day bar chart on the idle screen.
3. A "focus mode" that hides everything except the video and the ambient glow.
4. Export the session as a CSV for the physiotherapy angle.
5. Side-view calibration mode for a true CVA when a phone can be propped sideways.
