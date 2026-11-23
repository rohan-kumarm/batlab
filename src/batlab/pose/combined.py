"""Combines two pose models, using each for the points it measures best against Rohan's marks.

YOLO11 is used for the head, hands, knees and every other body point. MediaPipe is used for the
feet (ankles, heels and toes), because its foot points give the outer shoe edges far better
(tune set: left edge 2.8 cm vs YOLO's 23.6 cm, which has no heel or toe points). When MediaPipe
does not find the batter, the result falls back to YOLO alone (and to MediaPipe alone if YOLO
finds nobody).
"""

# numpy for array copies.
import numpy as np

# The shared keypoint record and point index.
from .base import IDX, Keypoints

# Standard points taken from MediaPipe instead of YOLO.
FEET = ("left_ankle", "right_ankle")
# Padding around YOLO's box, as a share of its height, for the second MediaPipe try.
PAD = 0.25


class CombinedPose:
    """Pose adapter that merges a YOLO adapter (body) and a MediaPipe adapter (feet)."""

    def __init__(self, yolo, mediapipe, cm_per_px: float | None = None):
        # Clip scale; when set, the outer shoe edges are also found from MediaPipe's person mask (see shoe_edges).
        # Set it per clip, since it comes from the stump calibration.
        self.cm_per_px = cm_per_px
        # Adapter used for the body points.
        self.yolo = yolo
        # Adapter used for the foot points.
        self.mediapipe = mediapipe

    def estimate(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Keypoints | None:
        """Return YOLO's keypoints with MediaPipe's ankles, heels and toes, or whichever model found the batter."""
        # Body from YOLO.
        body = self.yolo.estimate(frame, roi)
        # Feet from MediaPipe, in the region around the stumps.
        feet = self.mediapipe.estimate(frame, roi)
        # Missed there: try again in a region around the batter YOLO found, padded on every side.
        if feet is None and body is not None:
            # YOLO's box corners.
            x1, y1, x2, y2 = body.box
            # Padding: a quarter of the box height (the box is taller than wide).
            pad = PAD * (y2 - y1)
            # Retry on that region.
            feet = self.mediapipe.estimate(frame, (x1 - pad, y1 - pad, x2 + pad, y2 + pad))
        # Fall back to whichever model found someone.
        if body is None or feet is None:
            # Either one result or None, with shoe edges when the scale is known.
            return self._shoes(frame, body if body is not None else feet)
        # Copy YOLO's points so its result is not changed.
        xy, conf = body.xy.copy(), body.conf.copy()
        # Replace the ankles with MediaPipe's.
        for name in FEET:
            # Position.
            xy[IDX[name]] = feet.xy[IDX[name]]
            # Confidence.
            conf[IDX[name]] = feet.conf[IDX[name]]
        # Heels and toes exist only in MediaPipe; keep any extra points YOLO had too.
        return self._shoes(frame, Keypoints(xy, conf, body.box, {**body.extra, **feet.extra}))

    def _shoes(self, frame: np.ndarray, kp: Keypoints | None) -> Keypoints | None:
        """Add the outer shoe edges found in the image, if the clip's scale is known."""
        # Without a scale the band sizes are unknown: return as is.
        if self.cm_per_px is None:
            # Unchanged.
            return kp
        # Imported here to keep the pose package free of the shoe search unless it is used.
        from ..shoe_edges import with_shoe_edges
        # Keypoints with the two shoe edge points added, from MediaPipe's latest person mask.
        return with_shoe_edges(kp, getattr(self.mediapipe, "last_mask", None), self.cm_per_px)
