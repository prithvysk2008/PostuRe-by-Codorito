"""
METRIC SPECIFICATION — ported verbatim from posture_app.py.

Every metric is scale-invariant (normalised by shoulder width) and scored as
a *deviation from your own calibrated baseline*.
"""
from typing import Dict


class MetricSpec:
    __slots__ = ("weight", "direction", "tol", "span", "tip")

    def __init__(self, weight, direction, tol, span, tip):
        self.weight = weight        # contribution to the 0-100 score
        self.direction = direction  # "dec" = falling is bad, "inc" = rising is bad, "abs" = either
        self.tol = tol              # minimum dead-zone (natural fidgeting)
        self.span = span            # deviation beyond tolerance that costs full weight
        self.tip = tip              # coaching line shown when this metric dominates


METRICS: Dict[str, MetricSpec] = {
    # Vertical gap between ear-line and shoulder-line, in shoulder widths.
    "neck": MetricSpec(0.20, "dec", 0.020, 0.115, "Lift the crown of your head — your neck is collapsing."),
    # 2D craniovertebral angle (ear→shoulder vs horizontal). The clinical
    # measure this app is framed around, so it stays weighted highly, but not
    # so highly it drowns out every other signal.
    "cva": MetricSpec(0.22, "dec", 1.4, 8.0, "Craniovertebral angle dropping — tuck your chin back."),
    # Vertical gap between the shoulder-line and the tip of the nose — a
    # second, independent read on forward-head collapse anchored at the
    # shoulders rather than the ears, so it catches a chin-to-chest droop
    # that "neck" alone can miss.
    "chin": MetricSpec(0.12, "dec", 0.020, 0.115, "Your chin is dropping toward your chest — lift your head."),
    # Nose below the ear-line = looking down at the keyboard.
    "pitch": MetricSpec(0.16, "inc", 0.020, 0.110, "Head tilted down — raise your screen to eye level."),
    # Shoulders sinking relative to shoulder width = sliding down the chair.
    # Normalised by shoulder width (not frame height) so it stays equally
    # sensitive regardless of how far you're sitting from the camera.
    "drop": MetricSpec(0.14, "inc", 0.05, 0.22, "You're sinking into the chair — sit back into the backrest."),
    # Face appears larger relative to shoulders = head craning toward screen.
    "face": MetricSpec(0.10, "inc", 0.012, 0.070, "You're creeping toward the screen — push your chair in instead."),
    # Shoulder line rotated = leaning on one arm.
    "tilt": MetricSpec(0.06, "abs", 2.5, 12.0, "One shoulder is dropping — even out your weight."),
}
METRIC_KEYS = list(METRICS.keys())
