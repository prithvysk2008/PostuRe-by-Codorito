"""
END-OF-SESSION SUGGESTION — ported verbatim from posture_app.py.

One line, picked by where the session landed on two independent axes: how
much of it was slouched, and how tired the fatigue engine measured. 5 slouch
bands x 6 drowsiness bands = 30 standard suggestions.
"""
from typing import List

from .utils import clamp

SLOUCH_BAND_BOUNDS = [10.0, 25.0, 45.0, 70.0]        # 5 bands
DROWSY_BAND_BOUNDS = [15.0, 30.0, 45.0, 60.0, 80.0]  # 6 bands

SUGGESTIONS = [
    [  # slouch: minimal (<10%)
        "Textbook session — keep this exact setup and posture as your new normal.",
        "Great posture throughout; a two-minute walk would keep your energy matching it.",
        "Your posture was excellent — your eyes were working harder than your spine today.",
        "Alignment was great, but your eyes were fighting fatigue — consider a short break next time.",
        "Impressive posture despite clear drowsiness — a proper break matters more than the chair does.",
        "Your spine held up beautifully; your eyes did not — please rest before your next session.",
    ],
    [  # slouch: low (10-25%)
        "Solid session with only minor slips — you're close to a fully aligned baseline.",
        "Good posture with a touch of drift — a quick shoulder roll would tighten it up.",
        "Posture stayed mostly aligned, but fatigue crept in — a short walk would help both.",
        "A few slips paired with real tiredness — your body may be asking for an earlier break.",
        "Minor slouching but heavy fatigue — treat this as a sign to stop and rest.",
        "Posture held up, but you were seriously fatigued — that's a stop-and-rest signal, not a stretch one.",
    ],
    [  # slouch: moderate (25-45%)
        "Noticeable slouching crept in even though you were sharp — try raising your screen a notch.",
        "Posture drifted moderately — a lumbar cushion or seat adjustment could make a real difference.",
        "Both posture and focus drifted together — a mid-session stretch break would help either.",
        "Moderate slouching and rising fatigue — your setup and your sleep both deserve a look.",
        "You're slouching more as you tire — that's your body telling you it's time to stop.",
        "Significant slouch and heavy fatigue — wrap up soon and prioritize real rest.",
    ],
    [  # slouch: high (45-70%)
        "You were alert but slouched often — that's likely a setup problem, not an energy one; raise your monitor.",
        "Frequent slouching with mild fatigue — check your chair height and screen distance today.",
        "Posture slipped a lot and focus followed — a longer break than usual would help.",
        "High slouch and noticeable fatigue reinforcing each other — take a proper break before continuing.",
        "Heavy slouching and heavy fatigue — your body needs rest more than it needs another correction.",
        "Both posture and energy were struggling hard — stop for now and come back refreshed.",
    ],
    [  # slouch: severe (70%+)
        "Posture broke down often despite being alert — this session was really about your setup, not you.",
        "Severe slouching with only mild fatigue — your chair or desk height needs a real fix.",
        "Posture collapsed frequently as tiredness set in — consider ending sessions earlier.",
        "Heavy slouching paired with real fatigue — this is exactly the combination that leads to strain.",
        "Severe slouch and heavy fatigue together — please take a real break, this is a hard signal.",
        "This was a rough session for both your spine and your energy — rest is the priority now.",
    ],
]


def _band(value: float, bounds: List[float]) -> int:
    for i, b in enumerate(bounds):
        if value < b:
            return i
    return len(bounds)


def pick_suggestion(slouch_pct: float, drowsy_pct: float) -> str:
    si = _band(clamp(slouch_pct, 0.0, 100.0), SLOUCH_BAND_BOUNDS)
    di = _band(clamp(drowsy_pct, 0.0, 100.0), DROWSY_BAND_BOUNDS)
    return SUGGESTIONS[si][di]
