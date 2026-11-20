# Re-export the interface so callers can write "from batlab.pose import Keypoints".
from .base import IDX, NAMES, Keypoints, PoseEstimator

# Names made available by "from batlab.pose import *".
__all__ = ["IDX", "NAMES", "Keypoints", "PoseEstimator"]
