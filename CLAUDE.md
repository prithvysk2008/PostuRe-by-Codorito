# CLAUDE.md

Project rules for working in this repo. These are constraints, not suggestions — follow them even if a task seems to call for a shortcut around them.

## Stack is locked

- Python, MediaPipe Pose + Face Mesh, OpenCV, Streamlit, NumPy.
- Do not introduce a different pose/face model, a different UI framework, or a different CV library, even if it looks like a better fit for a specific feature. If a task seems to need something outside this stack, flag it instead of silently adding a dependency.
- Keep `requirements.txt` pinned. Don't loosen or bump versions without a reason tied to an actual bug.

## Must run 100% offline / on-device

- No cloud calls, no external API keys, no network requests of any kind at runtime.
- No database. Local persistence only (e.g. the existing `posture_data.json` pattern) — nothing that requires a server or an account.
- Before adding any library or code path, check it doesn't phone home (telemetry, model downloads at runtime, license pings, etc.). This is a hard requirement, not a preference — the offline claim is part of the product.

## Built for a live on-stage hackathon demo

- Reliability and graceful failure handling matter more than new features. A crash mid-demo is the worst possible outcome — prefer degrading visibly (an error banner, a fallback state) over throwing.
- Wrap risky per-frame or per-session logic (camera reads, model inference, file I/O) so a single bad frame or transient failure doesn't kill the session loop.
- Don't add features, abstractions, or "nice to have" polish beyond what's asked — every added surface is something that can break on stage.
- When in doubt between a clever fix and a boring, predictable one, take the boring one.

## Two apps in this repo

- `posture_app.py` (repo root) is the original Streamlit demo. Everything above ("Stack is locked") governs this app.
- `desktop/` is a separate, sanctioned Electron rewrite: an Electron shell (`desktop/electron/`) driving a React frontend (`desktop/frontend/`), talking to a local Python FastAPI + WebSocket sidecar (`desktop/backend/server.py`) that Electron spawns on launch and binds to `127.0.0.1:8765` only — never exposed to the network. The backend logic (engine, metrics, geometry, drawing, audio, stretches, share cards) is ported from `posture_app.py`.
- The "no different UI framework" rule under Stack is locked does not apply inside `desktop/` — Electron + React there is intentional, not a shortcut. The offline/on-device rule (no cloud calls, no external network requests) still applies fully to `desktop/`; its local FastAPI server is a same-machine sidecar, not a network service.
- When a task says "the app" without naming one, ask or infer from context (sidebar/`st.*` calls → Streamlit; Electron/React/frontend calls → `desktop/`) rather than guessing.

## Team

- Vatsal — ECE. Owns the sensing layer: calibration, metric extraction, scoring, thresholds.
- CSE / AI-ML teammate — owns the fatigue engine and prediction: EAR/MAR tuning, PERCLOS window, regression thresholds.
- Mechanical teammate — owns the health argument and demo logistics: cervical load framing, stretch content, running the on-stage laptop.
