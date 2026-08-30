#!/usr/bin/env bash
# Builds the standalone Python backend executable via PyInstaller.
# MediaPipe ships large native binaries + model files as package data that
# PyInstaller's static import analysis routinely misses — --collect-all
# pulls those in explicitly rather than hoping the analyzer finds them.
set -euo pipefail
cd "$(dirname "$0")/.."   # -> desktop/

VENV_PYTHON="../.venv/bin/python3"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Expected a virtualenv at $(cd .. && pwd)/.venv — run:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r desktop/backend/requirements.txt pyinstaller"
  exit 1
fi

rm -rf backend/build backend/dist backend/posture-backend.spec

# mediapipe's drawing_utils.py imports matplotlib.pyplot at module load time
# for one visualization helper (plot_landmarks()) this app never calls — the
# *real* matplotlib costs a slow one-time font-cache build for zero benefit,
# so --paths shadows it with backend/_stubs/matplotlib (see that package's
# docstring). Must come before --collect-all mediapipe so the import
# statement resolves to the stub during analysis.
"$VENV_PYTHON" -m PyInstaller \
  --name posture-backend \
  --onedir \
  --noconfirm \
  --distpath backend/dist \
  --workpath backend/build \
  --specpath backend \
  --paths backend/_stubs \
  --collect-all mediapipe \
  --collect-all cv2 \
  --hidden-import uvicorn.protocols.http.h11_impl \
  --hidden-import uvicorn.protocols.http.httptools_impl \
  --hidden-import uvicorn.protocols.websockets.websockets_impl \
  --hidden-import uvicorn.protocols.websockets.wsproto_impl \
  --hidden-import uvicorn.lifespan.on \
  --hidden-import uvicorn.lifespan.off \
  --hidden-import uvicorn.loops.auto \
  --hidden-import uvicorn.loops.asyncio \
  build_entry.py

echo "Built: desktop/backend/dist/posture-backend/posture-backend"
