"""Measures head tilt from the helmet: the angle of the grille bars and peak, which tilt with the head.

At 4K from the bowler's end the eyes are hidden in the shadow of the helmet peak, so an eye line
cannot be measured precisely (one pixel up or down is about 3 degrees). The helmet's straight,
high-contrast bars span about 4 times the eye gap, so their angle is far more precise.

Method: crop the face guard around the head point, then try angles from -15 to +15 degrees. At the
helmet's true angle, rotating the crop by that amount makes its bars horizontal, so the vertical
edge strength piles up in a few rows. The angle with the most concentrated row sums wins (the same
idea used to straighten scanned pages).
"""

# OpenCV for edge filtering and rotation.
import cv2
# numpy for array math.
import numpy as np

# Width of the helmet grille's top bar in cm, measured from the 10 Nov marks (about 101 px at
# 0.2116 cm/px). Used to size the crop at any zoom.
GRILLE_CM = 21.0
# Crop around the head point, in grille widths: none above, 0.9 below, 0.6 either side.
# The crop starts at the eyes so the helmet peak is left out: the peak's lower edge is a curved
# brim whose picture angle differs from the grille's (on IMG_0552 a peak-only crop read 4 deg off
# the marked grille, and including it made the detector read 1.9 deg high there). Chosen on the
# 10 Nov and IMG_0552 tune frames (mean error 0.4 and 0.9 deg, was 0.5 and 1.9).
CROP_UP, CROP_DOWN, CROP_HALF = 0.0, 0.9, 0.6


def helmet_tilt(frame: np.ndarray, head_xy, cm_per_px: float) -> float | None:
    """Angle of the helmet's bars in degrees, measured left to right in the picture.

    Positive when the right end is lower (image y grows downward), the same convention as
    atan2(dy, dx) on two marked points. Returns None if the crop falls outside the frame.
    """
    # Expected grille width in pixels at this clip's scale.
    w_px = GRILLE_CM / cm_per_px
    # Head point.
    cx, cy = float(head_xy[0]), float(head_xy[1])
    # Crop corners around the head.
    x1, x2 = int(cx - CROP_HALF * w_px), int(cx + CROP_HALF * w_px)
    # Top and bottom of the crop.
    y1, y2 = int(cy - CROP_UP * w_px), int(cy + CROP_DOWN * w_px)
    # A crop cut off by the frame edge would bias the result; give up instead.
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0]:
        # Not measurable.
        return None
    # Greyscale crop as decimals.
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Vertical edge strength: high on the top and bottom edges of horizontal bars.
    edges = np.abs(cv2.Sobel(crop, cv2.CV_32F, 0, 1, ksize=3))
    # Crop height and width.
    h, w = edges.shape
    # Row and column index of every pixel.
    yy, xx = np.mgrid[0:h, 0:w]
    # Keep an oval in the middle, so rotating does not move the corners in and out of view.
    edges *= (((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2 <= 1)
    # Angles to try, in 0.25-degree steps.
    angles = np.arange(-15.0, 15.001, 0.25)
    # Score for each angle.
    scores = []
    # Try every angle.
    for a in angles:
        # Rotation by a degrees about the crop's centre (OpenCV: positive = anticlockwise).
        m = cv2.getRotationMatrix2D((w / 2, h / 2), float(a), 1.0)
        # Rotate the edge image.
        rot = cv2.warpAffine(edges, m, (w, h), flags=cv2.INTER_LINEAR)
        # Sum each row; squared sums reward edge strength piled into few rows (bars made level).
        scores.append(float(np.sum(rot.sum(axis=1) ** 2)))
    # Best angle's position in the list.
    i = int(np.argmax(scores))
    # Refine between the 0.25-degree steps by fitting a parabola through the best score and its neighbours.
    if 0 < i < len(angles) - 1:
        # Scores either side of the best.
        s0, s1, s2 = scores[i - 1], scores[i], scores[i + 1]
        # Curvature of the parabola (negative at a peak).
        denom = s0 - 2 * s1 + s2
        # Offset of the parabola's top from the best step, in steps (0 if the scores are flat).
        shift = 0.5 * (s0 - s2) / denom if denom != 0 else 0.0
        # Refined angle.
        return float(angles[i] + shift * 0.25)
    # Best angle at the edge of the range: return it unrefined.
    return float(angles[i])
