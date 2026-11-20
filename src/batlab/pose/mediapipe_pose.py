"""MediaPipe Pose adapter. License: Apache 2.0 (free for commercial use)."""

# os sets environment variables that quieten MediaPipe's startup log.
import os
# ssl builds a secure connection for the model download.
import ssl
# sys gives access to Python's error stream, flushed before muting.
import sys
# contextmanager turns a generator function into a `with` block helper.
from contextlib import contextmanager
# urllib downloads the model file on first use.
import urllib.request

# Hide MediaPipe's INFO and WARNING startup lines (they must be set before mediapipe is imported).
os.environ.setdefault("GLOG_minloglevel", "2")
# Same for the TensorFlow Lite part of MediaPipe.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# certifi provides an up-to-date list of trusted website certificates.
import certifi
# Path builds the cache location.
from pathlib import Path

# OpenCV converts frames from BGR to the RGB order MediaPipe expects.
import cv2
# numpy for array math.
import numpy as np

# The 17 standard point names and the Keypoints record this adapter must return.
from .base import NAMES, Keypoints

# MediaPipe gives 33 landmarks. Position of each of our 17 standard points in that list.
MP_INDEX = {
    "nose": 0, "left_eye": 2, "right_eye": 5, "left_ear": 7, "right_ear": 8,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_hip": 23, "right_hip": 24,
    "left_knee": 25, "right_knee": 26, "left_ankle": 27, "right_ankle": 28,
}
# Foot points YOLO11 does not have; the balance check uses them for the edges of the base.
MP_EXTRA = {"left_heel": 29, "right_heel": 30, "left_toe": 31, "right_toe": 32}
# Google's download address for the most accurate ("heavy") pose model, about 30 MB.
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")
# Where the model is kept after the first download.
CACHE = Path.home() / ".cache" / "batlab" / "pose_landmarker_heavy.task"


@contextmanager
def _quiet_stderr():
    """Mute everything written to the error stream (file descriptor 2) inside a `with` block.

    MediaPipe's C++ code writes straight to descriptor 2, so replacing Python's sys.stderr is not
    enough; the descriptor itself is pointed at /dev/null and restored afterwards.
    """
    # Make sure Python's own buffered error text is written out before muting.
    sys.stderr.flush()
    # Keep a copy of the real error stream so it can be restored.
    saved = os.dup(2)
    # Open the "discard everything" device.
    devnull = os.open(os.devnull, os.O_WRONLY)
    # Point descriptor 2 at it.
    os.dup2(devnull, 2)
    # try/finally restores the stream even if the code inside raises an error.
    try:
        # Run the code inside the `with` block.
        yield
    finally:
        # Point descriptor 2 back at the real error stream.
        os.dup2(saved, 2)
        # Close the two descriptors no longer needed.
        os.close(saved)
        # Close /dev/null.
        os.close(devnull)


def landmarks_to_keypoints(landmarks, width: int, height: int, offset=(0.0, 0.0)) -> Keypoints:
    """Convert one person's 33 MediaPipe landmarks to Keypoints in full-frame pixels.

    MediaPipe gives x and y as fractions (0 to 1) of the image it was given, so they are
    multiplied by that image's width and height, then shifted by where that image sat in the frame.
    """
    # Shift for a crop's position in the full frame.
    ox, oy = offset

    def to_px(lm) -> np.ndarray:
        """Fraction coordinates to full-frame pixel coordinates."""
        # Scale by image size, then add the crop offset.
        return np.array([lm.x * width + ox, lm.y * height + oy])

    def conf(lm) -> float:
        """Visibility score clamped to 0..1 (missing scores count as 0)."""
        # Some versions leave visibility unset (None); treat that as not visible.
        return float(min(1.0, max(0.0, lm.visibility or 0.0)))

    # Positions of the 17 standard points, in NAMES order.
    xy = np.array([to_px(landmarks[MP_INDEX[n]]) for n in NAMES])
    # Confidence of the 17 standard points.
    c = np.array([conf(landmarks[MP_INDEX[n]]) for n in NAMES])
    # Heel and toe points with their confidences.
    extra = {n: (to_px(landmarks[i]), conf(landmarks[i])) for n, i in MP_EXTRA.items()}
    # Every landmark's pixel position, for the person box.
    allpts = np.array([to_px(lm) for lm in landmarks])
    # Box around all landmarks.
    box = (float(allpts[:, 0].min()), float(allpts[:, 1].min()), float(allpts[:, 0].max()), float(allpts[:, 1].max()))
    # Package the result.
    return Keypoints(xy, c, box, extra)


class MediaPipePose:
    """Runs MediaPipe's pose model on a region around the batter."""

    def __init__(self, model_path: str | None = None):
        # Import here so the rest of the package works without mediapipe installed.
        from mediapipe import Image, ImageFormat
        # The pose task classes, imported by module path (same path in MediaPipe 0.10 and 1.x).
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision.pose_landmarker import PoseLandmarker, PoseLandmarkerOptions
        # Use the given model file, or the cached download.
        path = Path(model_path) if model_path else CACHE
        # Download the model the first time.
        if not path.exists():
            # Make the cache folder.
            path.parent.mkdir(parents=True, exist_ok=True)
            # Tell the user why there is a pause.
            print(f"Downloading MediaPipe pose model to {path} ...")
            # Use certifi's list of trusted certificates, so the download works even when Python's
            # own list is missing (as with the python.org installer on a Mac).
            context = ssl.create_default_context(cafile=certifi.where())
            # Open the download.
            with urllib.request.urlopen(MODEL_URL, context=context) as resp:
                # Write the bytes to the cache file.
                path.write_bytes(resp.read())
        # Keep the image classes for converting frames.
        self._Image, self._fmt = Image, ImageFormat.SRGB
        # Build the landmarker: one person, single-image mode (each frame handled on its own).
        # Loading prints about 40 INFO/WARNING lines from MediaPipe's C++ code, which ignores the
        # log settings above, so its error output is muted while it loads.
        with _quiet_stderr():
            # Create the model.
            self.landmarker = PoseLandmarker.create_from_options(
                PoseLandmarkerOptions(base_options=BaseOptions(model_asset_path=str(path)), num_poses=1)
            )

    def _run(self, image: np.ndarray):
        """Run MediaPipe on one BGR image; return the 33 landmarks or None."""
        # MediaPipe wants RGB; ascontiguousarray gives it a clean memory layout.
        rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # Run the model.
        result = self.landmarker.detect(self._Image(image_format=self._fmt, data=rgb))
        # The first (only) person's landmarks, or None if nobody was found.
        return result.pose_landmarks[0] if result.pose_landmarks else None

    def estimate(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Keypoints | None:
        """Return the batter's keypoints in full-frame pixels, or None.

        MediaPipe's person detector shrinks its input to about 224 px, so in a 4K frame the batter
        becomes too small to find (tested: 0 of 3 releases found on the full frame). `roi` is a
        region (x1, y1, x2, y2) around the batter, worked out from the stumps; MediaPipe runs on
        that crop instead (tested: 3 of 3 found).
        """
        # Frame height and width.
        h, w = frame.shape[:2]
        # Without a region, use the whole frame (works when the batter fills much of it).
        x1, y1, x2, y2 = roi if roi is not None else (0, 0, w, h)
        # Clamp the region to the frame edges.
        x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), min(w, int(x2)), min(h, int(y2))
        # The region itself.
        crop = frame[y1:y2, x1:x2]
        # Run on the region.
        lms = self._run(crop)
        # Found: convert to full-frame pixels by adding the region's top-left corner.
        if lms is not None:
            # Fractions are of the crop's own width and height.
            return landmarks_to_keypoints(lms, x2 - x1, y2 - y1, (x1, y1))
        # Second try: pad the crop to a square with black on the right and bottom. MediaPipe works
        # on square inputs, and on the 10 Nov clip this found a release the plain crop missed.
        side = max(crop.shape[:2])
        # Add the black border.
        square = cv2.copyMakeBorder(crop, 0, side - crop.shape[0], 0, side - crop.shape[1],
                                    cv2.BORDER_CONSTANT, value=(0, 0, 0))
        # Run on the square.
        lms = self._run(square)
        # Nobody found either way.
        if lms is None:
            # Signal missing.
            return None
        # Fractions are of the square's size; padding was added right and bottom, so the offset is unchanged.
        return landmarks_to_keypoints(lms, side, side, (x1, y1))
