"""
MODELS + CAMERA — ported from posture_app.py's get_camera/load_pose/load_face.

The only real change from the Streamlit version: `st.cache_resource` and
`st.session_state` (which only make sense inside a Streamlit script rerun)
are replaced with plain process-wide caching, since this desktop app is a
single long-lived Python process with one session at a time — the same
"keep the model/camera alive across reruns" intent, just expressed without
a web-framework runtime underneath it.
"""
import os
from functools import lru_cache
from typing import Optional

import cv2
import mediapipe as mp

from .constants import CAP_H, CAP_W


@lru_cache(maxsize=4)
def load_pose(model_complexity: int):
    return mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


@lru_cache(maxsize=1)
def load_face():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


class CameraManager:
    """Keeps one VideoCapture alive across the life of a session — the
    process-local equivalent of the old st.session_state.cap/cap_index."""

    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.cap_index: Optional[int] = None

    def get_camera(self, index: int):
        if self.cap is not None and self.cap.isOpened():
            if self.cap_index == index:
                return self.cap
            # a different index was requested — release the old handle first
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():  # retry with the default backend
            cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAP_W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAP_H)
        cap.set(cv2.CAP_PROP_FPS, 30)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        self.cap = cap
        self.cap_index = index
        return cap

    def release_camera(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.cap_index = None
