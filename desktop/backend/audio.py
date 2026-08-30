"""
OFFLINE AUDIO — chimes synthesised with NumPy at runtime. No sound files to
ship, nothing to download.

Ported from posture_app.py's make_chime()/CHIME_SPECS. The only change from
the Streamlit version: `@st.cache_data` -> `functools.lru_cache`, since there
is no Streamlit runtime here. `play_chime()` (the Streamlit `st.audio()`
wrapper) is dropped — the desktop server instead sends the raw WAV bytes to
the frontend over the WebSocket, which plays them with a plain <audio> tag.
"""
import io
import wave as wave_lib
from functools import lru_cache
from typing import Tuple

import numpy as np

# "watch"/"bad"/"predict"/"fatigue" are short attention-grabbing phrases meant
# to cut through background noise without a screen glance. "break"/"recover"/
# "start"/"end" stay closer to a single soft, positive tone since they're not
# urgent — a session bookend, not an alert.
CHIME_SPECS = {
    "watch": {  # falling two-note phrase — "you're drifting"
        "notes": ((659.25, 0.00, 0.16), (523.25, 0.15, 0.24)),
        "volume": 0.30, "decay": 9.0,
    },
    "bad": {  # firm falling run — "you've slumped"
        "notes": ((587.33, 0.00, 0.14), (493.88, 0.14, 0.14), (392.00, 0.28, 0.32)),
        "volume": 0.34, "decay": 10.0,
    },
    "predict": {  # two soft knocks then a rise — "heads up, before it happens"
        "notes": ((659.25, 0.00, 0.13), (659.25, 0.16, 0.13), (783.99, 0.32, 0.26)),
        "volume": 0.28, "decay": 9.5,
    },
    "fatigue": {  # low, slow double knock — "you're fading"
        "notes": ((392.00, 0.00, 0.20), (349.23, 0.22, 0.32)),
        "volume": 0.32, "decay": 8.0,
    },
    "break": {  # bright ascending triad — "time to move"
        "notes": ((523.25, 0.00, 0.40), (659.25, 0.12, 0.40), (783.99, 0.24, 0.55)),
        "volume": 0.20, "decay": 3.2,
    },
    "recover": {  # bright two-note lift — "nice save"
        "notes": ((783.99, 0.00, 0.40), (1046.50, 0.12, 0.55)),
        "volume": 0.20, "decay": 3.2,
    },
    "start": {  # single soft rising tone — "tracking has begun"
        "notes": ((523.25, 0.00, 0.32),),
        "volume": 0.16, "decay": 3.4,
    },
    "end": {  # gentle two-note close — "session complete"
        "notes": ((659.25, 0.00, 0.36), (523.25, 0.18, 0.46)),
        "volume": 0.16, "decay": 3.0,
    },
}


@lru_cache(maxsize=256)
def make_chime(notes: Tuple[Tuple[float, float, float], ...], volume: float = 0.22,
                decay: float = 3.2) -> bytes:
    """Return raw WAV bytes built from a short sequence of percussive notes.

    Each note is (freq, start_offset_s, note_len_s) and gets its own fast
    attack + exponential decay, so a multi-note phrase reads as distinct taps
    or a falling run rather than one blurred chord.
    """
    sr = 22050
    total = max(start + length for _, start, length in notes) + 0.12
    t = np.linspace(0.0, total, int(sr * total), endpoint=False)
    tone = np.zeros_like(t)
    for freq, start, length in notes:
        local = t - start
        env = np.exp(-decay * np.maximum(local, 0.0)) * (local >= 0.0)
        attack = np.clip(local / 0.012, 0.0, 1.0)  # gentle attack so it never clicks
        tone += np.sin(2 * np.pi * freq * local) * env * attack
    peak = float(np.max(np.abs(tone))) or 1.0
    audio = np.int16(np.clip(tone / peak * volume, -1, 1) * 32767)

    buf = io.BytesIO()
    with wave_lib.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    return buf.getvalue()


def chime_wav_bytes(kind: str, volume_scale: float = 1.0) -> bytes:
    """Look up a chime kind and synthesise its WAV bytes at the given volume."""
    from .utils import clamp
    spec = CHIME_SPECS[kind]
    vol = spec["volume"] * clamp(volume_scale, 0.0, 1.5)
    return make_chime(spec["notes"], vol, spec["decay"])
