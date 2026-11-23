"""Finds the outer edges of the batter's shoes from MediaPipe's person mask.

Pose models give points along the middle of each foot (ankle, heel, toe), not the outside of the
shoe. Front-on, MediaPipe also tends to put both feet on the nearer shoe, so the back shoe, whose
heel is often the right-most part of the base, is missed: on 10 Nov and IMG_0552 the right edge
came out 9-11 cm inside Rohan's marks.

MediaPipe also outlines the whole person (a mask of which pixels belong to them). It was trained on
people of all kinds in all kinds of places, so it does not depend on shoe colour or the floor. The
shoes' outer edges are the left-most and right-most mask pixels in a thin band at sole height.

Tune frames (10 Nov, IMG_0552): left edge 0.8 cm, right 1.1 cm from the marks. On copies of the same
frames with the shoes recoloured black, red or navy and the turf recoloured grey or brown, errors
were 2.4-4.8 cm (the recolouring is crude: shaded parts of the shoe keep their old colour).
"""

# numpy for array math.
import numpy as np

# The shared keypoint record.
from .pose.base import Keypoints

# Foot points that set the sole height and the minimum width of the base.
FOOT = ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_toe", "right_toe")
# Band searched: from this far above the lowest foot point to this far below it, cm. Low enough to
# stay under the pads. Chosen with MASK_LEVEL on the tune frames over 3 band heights and 3 levels.
BAND_UP_CM, BAND_DOWN_CM = 5.0, 2.0
# Mask value (0-1) above which a pixel counts as the person.
MASK_LEVEL = 0.5


def shoe_edges(kp: Keypoints, mask, cm_per_px: float) -> tuple[float, float, float] | None:
    """Return (left x, right x, y) of the shoes' outer edges in full-frame pixels, or None.

    `mask` is (values, x offset, y offset) from the MediaPipe adapter, or None. None is returned
    when fewer than two foot points were found. Each edge is at least as far out as the model's own
    foot points, so a missing or poor mask never narrows the base.
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
    left, right = min(p[0] for p in pts), max(p[0] for p in pts)
    # Without a mask, the foot points are all there is.
    if mask is None:
        # Foot points' own extent.
        return float(left), float(right), float(y_low)
    # Mask values and where its top-left corner sits in the frame.
    values, ox, oy = mask
    # Band rows inside the mask.
    y1 = max(0, int(y_low - BAND_UP_CM / cm_per_px) - oy)
    # Bottom row (exclusive).
    y2 = min(values.shape[0], int(y_low + BAND_DOWN_CM / cm_per_px) - oy + 1)
    # Band outside the mask: keep the foot points.
    if y2 <= y1:
        # Foot points' own extent.
        return float(left), float(right), float(y_low)
    # Columns where the person shows anywhere in the band.
    cols = np.flatnonzero((values[y1:y2] > MASK_LEVEL).any(axis=0))
    # Person not in the band: keep the foot points.
    if len(cols) == 0:
        # Foot points' own extent.
        return float(left), float(right), float(y_low)
    # Widen to the mask's extent (never narrower than the foot points).
    return float(min(left, cols[0] + ox)), float(max(right, cols[-1] + ox)), float(y_low)


def with_shoe_edges(kp: Keypoints | None, mask, cm_per_px: float) -> Keypoints | None:
    """Return kp with "shoe_left_edge" and "shoe_right_edge" added as extra points (if found)."""
    # Nothing to add to.
    if kp is None:
        # Pass through.
        return None
    # Measure the edges.
    found = shoe_edges(kp, mask, cm_per_px)
    # Not measurable: return unchanged.
    if found is None:
        # Pass through.
        return kp
    # Unpack.
    left, right, y = found
    # Add both edges with full confidence.
    extra = {**kp.extra, "shoe_left_edge": (np.array([left, y]), 1.0), "shoe_right_edge": (np.array([right, y]), 1.0)}
    # New record with the extra points.
    return Keypoints(kp.xy, kp.conf, kp.box, extra)
