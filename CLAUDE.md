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

## Team

- Vatsal — ECE. Owns the sensing layer: calibration, metric extraction, scoring, thresholds.
- CSE / AI-ML teammate — owns the fatigue engine and prediction: EAR/MAR tuning, PERCLOS window, regression thresholds.
- Mechanical teammate — owns the health argument and demo logistics: cervical load framing, stretch content, running the on-stage laptop.
