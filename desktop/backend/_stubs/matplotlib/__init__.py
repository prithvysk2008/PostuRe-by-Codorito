"""Build-time stub — see ../../build_pyinstaller.sh.

This app never uses matplotlib. mediapipe's own drawing_utils.py imports
matplotlib.pyplot at module load time, purely to support one visualization
helper (plot_landmarks()) that this app never calls — it draws its own
overlays in backend/drawing.py instead. Importing the *real* matplotlib just
to satisfy that unused import costs a slow one-time font-cache build for no
benefit, so the PyInstaller build points --paths at this stub package
instead, shadowing the real one for that one import statement only. The dev
virtualenv and the original Streamlit app are unaffected — this directory is
never added to their sys.path.
"""
__version__ = "0.0-stub"
