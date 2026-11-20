"""The pose interface every pose model adapter must follow."""

# dataclass for the Keypoints record.
from dataclasses import dataclass
# Protocol describes the methods a class must have, without forcing inheritance.
from typing import Protocol

# numpy arrays hold the keypoint coordinates.
import numpy as np

# The 17 body points in COCO order, which both YOLO and the planned MediaPipe adapter map to.
# "left" and "right" are the player's own left and right, not the image's.
NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
# Lookup from name to position in the arrays, e.g. IDX["left_knee"] == 13.
IDX = {name: i for i, name in enumerate(NAMES)}


# One person's pose in one frame.
@dataclass(frozen=True)
class Keypoints:
    # (17, 2) array of x, y pixel positions in the full frame.
    xy: np.ndarray
    # (17,) array of confidence scores from 0 to 1.
    conf: np.ndarray
    # Person box as (x1, y1, x2, y2) in the full frame.
    box: tuple[float, float, float, float]

    def point(self, name: str, min_conf: float = 0.5) -> np.ndarray | None:
        """Return (x, y) of a named point, or None if the model was not confident enough."""
        # Position of this point in the arrays.
        i = IDX[name]
        # Below the confidence limit the point is treated as missing.
        if self.conf[i] < min_conf:
            # None tells callers to skip any check that needs this point.
            return None
        # Return the coordinates.
        return self.xy[i]

    def midpoint(self, a: str, b: str, min_conf: float = 0.5) -> np.ndarray | None:
        """Return the point halfway between two named points, or None if either is missing."""
        # Look up both points.
        pa, pb = self.point(a, min_conf), self.point(b, min_conf)
        # If either is missing, so is the midpoint.
        if pa is None or pb is None:
            # Signal missing.
            return None
        # Average of the two positions.
        return (pa + pb) / 2


class PoseEstimator(Protocol):
    """Any pose model adapter (YOLO11, MediaPipe, ...) must provide this method."""

    def estimate(self, frame: np.ndarray) -> Keypoints | None:
        """Return the batter's keypoints in a BGR frame, or None if no person is found."""
        # Protocol methods have no body; "..." marks that.
        ...
