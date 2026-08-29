"""
PostuRe: — pre-demo check.

Run this BEFORE you present. It verifies Python, the libraries, your camera,
and measures the real frame rate of the full pipeline on this machine.

    python check_setup.py
"""
import sys
import time

OK, BAD, WARN = "  [OK]  ", "  [FAIL]", "  [WARN]"
problems = []


def fail(msg, fix):
    print(BAD + " " + msg)
    print("         fix: " + fix)
    problems.append(msg)


print("\nPostuRe: setup check")
print("=" * 62)

# 1. Python -------------------------------------------------------------------
v = sys.version_info
print(f"  Python {v.major}.{v.minor}.{v.micro}")
if (v.major, v.minor) < (3, 9) or (v.major, v.minor) > (3, 12):
    fail("MediaPipe supports Python 3.9-3.12 only.",
         "Install Python 3.11 and rebuild the virtual environment.")
else:
    print(OK + " Python version supported")

# 2. Imports ------------------------------------------------------------------
mods = {}
for name, pip in [("numpy", "numpy==1.26.4"), ("cv2", "opencv-python==4.10.0.84"),
                  ("mediapipe", "mediapipe==0.10.14"), ("streamlit", "streamlit>=1.31")]:
    try:
        mods[name] = __import__(name)
        ver = getattr(mods[name], "__version__", "?")
        print(OK + f" {name:<11} {ver}")
    except Exception as exc:
        fail(f"{name} failed to import ({exc.__class__.__name__}).", f"pip install {pip}")

if "numpy" in mods and mods["numpy"].__version__.startswith("2."):
    print(WARN + " numpy 2.x can break mediapipe 0.10.x -> pip install numpy==1.26.4")

if problems:
    print("\n" + "=" * 62)
    print(f"  {len(problems)} problem(s) found. Fix them before the demo.\n")
    sys.exit(1)

import cv2
import mediapipe as mp
import numpy as np

# 3. Camera -------------------------------------------------------------------
print("-" * 62)
found = []
for idx in range(4):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok and frame is not None:
            found.append((idx, frame.shape[1], frame.shape[0]))
            print(OK + f" camera index {idx}: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
if not found:
    fail("No camera responded on indexes 0-3.",
         "Close Zoom/Meet/Teams/OBS, check OS camera permissions, replug the webcam.")
    print("\n  Cannot continue without a camera.\n")
    sys.exit(1)

# 4. Pipeline throughput ------------------------------------------------------
print("-" * 62)
idx = found[0][0]
print(f"  Benchmarking the full pipeline on camera {idx} (about 6 seconds)...")
cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

pose = mp.solutions.pose.Pose(model_complexity=1, min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)
face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=False,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)

n, pose_hits, face_hits, t0 = 0, 0, 0, time.time()
while time.time() - t0 < 6.0:
    ok, frame = cap.read()
    if not ok:
        continue
    rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    r = pose.process(rgb)
    if r.pose_landmarks:
        pose_hits += 1
    if n % 2 == 0:
        f = face.process(rgb)
        if f.multi_face_landmarks:
            face_hits += 1
    n += 1
cap.release()

elapsed = time.time() - t0
fps = n / elapsed
print(f"  {n} frames in {elapsed:.1f}s  ->  {fps:.1f} FPS")
print(f"  body detected in {pose_hits}/{n} frames · face detected in {face_hits}/{max(n//2,1)} checks")

print("-" * 62)
if fps < 8:
    print(WARN + " Below 8 FPS. In the sidebar set Pose model to 'Fast (0)' and"
                 " Face mesh to every 3 frames.")
elif fps < 14:
    print(WARN + f" {fps:.0f} FPS is usable but not smooth. Close other apps before the demo.")
else:
    print(OK + f" {fps:.0f} FPS — plenty for a live demo.")

if pose_hits < n * 0.5:
    print(WARN + " Your body was detected in under half the frames. Sit so the camera"
                 " sees your head AND both shoulders, and add more light from the front.")
if face_hits < (n // 2) * 0.5:
    print(WARN + " Face detection was weak. The fatigue engine needs a well-lit,"
                 " unobstructed face. Glasses glare and backlight are the usual culprits.")

print("\n  All clear. Launch with:  streamlit run posture_app.py\n")
