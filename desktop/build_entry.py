"""
PyInstaller entry point for the bundled desktop backend.

Lives as a sibling of the `backend` package (not inside it) so `backend`
resolves as an ordinary top-level package for PyInstaller's import analysis,
matching how it's imported everywhere else in this project.

Known cold-start cost: mediapipe.python.solutions.__init__ unconditionally
imports drawing_utils/drawing_styles (even though this app never calls them —
it draws its own overlays in backend/drawing.py), which imports matplotlib.
On a genuinely fresh machine, matplotlib's *first ever* import does a one-time
font-cache scan that can take 30-40s; it's cached under ~/.matplotlib after
that, so every launch after the first is fast. MPLBACKEND=Agg (set by
electron/main.js when it spawns this process) at least skips matplotlib's
interactive-backend auto-detection on top of that.
"""
from backend.server import main

if __name__ == "__main__":
    main()
