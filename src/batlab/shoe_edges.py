"""Finds the outer edges of the batter's shoes in the picture, from the white of the shoes against the turf.

Pose models give points along the middle of each foot (ankle, heel, toe), not the outside of the
shoe. Front-on, MediaPipe also tends to put both feet on the nearer shoe, so the back shoe, whose
heel is often the right-most part of the base, is missed: on 10 Nov and IMG_0552 the right edge
came out 9-11 cm inside Rohan's marks. Starting from the model's foot points, this walks sideways
along a thin band at shoe height while the pixels stay shoe-white, and returns where the white ends.
"""

# OpenCV for the colour conversion.
import cv2
# numpy for array math.
import numpy as np

# The shared keypoint record.
from .pose.base import Keypoints

# Foot points the search starts from.
FOOT = ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_toe", "right_toe")
# Shoe-white: low colour (HSV saturation below this) and bright (value above this).
# Chosen with the band sizes below on the 10 Nov and IMG_0552 tune frames (shoe edge error
# 2.1 cm left and 1.9 cm right, against 2.7 and 9.4 from the foot points alone).
MAX_SATURATION, MIN_VALUE = 40, 200
# Band searched: from this far above the lowest foot point to this far below it, cm. Low enough to
# stay under the pads, which are white too.
BAND_UP_CM, BAND_DOWN_CM = 8.0, 2.0
# Gaps in the white narrower than this (cm) are bridged, e.g. a dark stripe on the shoe.
GAP_CM = 0.5


def shoe_edges(frame: np.ndarray, kp: Keypoints, cm_per_px: float) -> tuple[float, float, float] | None:
    """Return (left x, right x, y) of the shoes' outer edges in full-frame pixels, or None.

    None when fewer than two foot points were found. Each edge is at least as far out as the
    model's own foot points, so the result never narrows the base.
    """
    # Foot points the model found.
    pts = [p for p in (kp.point(n) for n in FOOT) if p is not None]
    # Need at least two to know where the feet are.
    if len(pts) < 2:
        # Not measurable.
        return None
    # Lowest foot point (largest y): the soles are just below it.
    y_low = max(p[1] for p in pts)
    # Left-most and right-most foot point x.
    x_lo, x_hi = min(p[0] for p in pts), max(p[0] for p in pts)
    # Frame size.
    h, w = frame.shape[:2]
    # Band rows, clamped to the frame.
    y1, y2 = max(0, int(y_low - BAND_UP_CM / cm_per_px)), min(h, int(y_low + BAND_DOWN_CM / cm_per_px) + 1)
    # Search up to 40 cm beyond the foot points on each side (a shoe is about 30 cm long).
    reach = int(40.0 / cm_per_px)
    # Columns searched, clamped to the frame.
    x1, x2 = max(0, int(x_lo) - reach), min(w, int(x_hi) + reach + 1)
    # Nothing to search (feet at the frame edge).
    if y2 <= y1 or x2 <= x1:
        # Keep the foot points' own extent.
        return float(x_lo), float(x_hi), float(y_low)
    # Colour as hue, saturation, value.
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    # Columns with any shoe-white pixel in the band.
    white = ((hsv[..., 1] < MAX_SATURATION) & (hsv[..., 2] > MIN_VALUE)).any(axis=0)
    # Largest gap bridged, in columns.
    gap = max(1, int(GAP_CM / cm_per_px))
    # Walk right from the right-most foot point while white continues (allowing small gaps).
    x = int(x_hi) - x1
    # Step while there is white within the next gap + 1 columns (so a gap of `gap` columns is bridged).
    while x + 1 < len(white) and white[x + 1: x + 2 + gap].any():
        # One column further right.
        x += 1
    # Right edge in full-frame pixels.
    right = float(x + x1)
    # Walk left from the left-most foot point the same way.
    x = int(x_lo) - x1
    # Step while there is white within the previous gap + 1 columns.
    while x - 1 >= 0 and white[max(0, x - 1 - gap): x].any():
        # One column further left.
        x -= 1
    # Left edge in full-frame pixels.
    left = float(x + x1)
    # Both edges and the sole height.
    return left, right, float(y_low)


def with_shoe_edges(frame: np.ndarray, kp: Keypoints | None, cm_per_px: float) -> Keypoints | None:
    """Return kp with "shoe_left_edge" and "shoe_right_edge" added as extra points (if found)."""
    # Nothing to add to.
    if kp is None:
        # Pass through.
        return None
    # Measure the edges.
    found = shoe_edges(frame, kp, cm_per_px)
    # Not measurable: return unchanged.
    if found is None:
        # Pass through.
        return kp
    # Unpack.
    left, right, y = found
    # Add both edges with full confidence (they come from the image, not a model score).
    extra = {**kp.extra, "shoe_left_edge": (np.array([left, y]), 1.0), "shoe_right_edge": (np.array([right, y]), 1.0)}
    # New record with the extra points.
    return Keypoints(kp.xy, kp.conf, kp.box, extra)
