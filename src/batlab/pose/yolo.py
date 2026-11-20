"""YOLO11-pose adapter. License: AGPL-3.0 (see docs/licensing.md)."""

# numpy for array math.
import numpy as np

# The Keypoints record this adapter must return.
from .base import Keypoints


class YoloPose:
    """Finds the batter in two passes: locate the person in the whole frame, then re-run on a crop.

    In a 4K front-on frame the batter is a small part of the image. The model shrinks its input
    to `imgsz` pixels, so the second pass on a tight crop gives it far more pixels of the batter.
    """

    def __init__(self, weights: str = "yolo11m-pose.pt", imgsz: int = 1280, crop_imgsz: int = 960):
        # Import here so the rest of the package works without ultralytics installed.
        from ultralytics import YOLO
        # Load the model; a bare name like "yolo11m-pose.pt" downloads from GitHub on first use.
        self.model = YOLO(weights)
        # Input size for the whole-frame pass.
        self.imgsz = imgsz
        # Input size for the crop pass.
        self.crop_imgsz = crop_imgsz

    def _run(self, image: np.ndarray, imgsz: int):
        """Run the model once and return (boxes, keypoint xy, keypoint conf) as numpy arrays."""
        # verbose=False stops ultralytics printing a line per image.
        r = self.model(image, imgsz=imgsz, verbose=False)[0]
        # No people found: return empty results.
        if r.keypoints is None or len(r.boxes) == 0:
            # Three empty values.
            return None, None, None
        # Move the tensors to numpy and return them.
        return r.boxes.xyxy.cpu().numpy(), r.keypoints.xy.cpu().numpy(), r.keypoints.conf.cpu().numpy()

    def estimate(self, frame: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> Keypoints | None:
        """Return the batter's keypoints in full-frame pixels, or None.

        `roi` is accepted to match the interface but not used: YOLO finds the batter in the whole
        frame on its own, then re-runs on a crop.
        """
        # First pass on the whole frame, only to find where the person is.
        boxes, _, _ = self._run(frame, self.imgsz)
        # Nobody found.
        if boxes is None:
            # Signal missing.
            return None
        # Areas of all person boxes.
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # The batter is the largest person in frame (the camera is aimed at them).
        x1, y1, x2, y2 = boxes[int(np.argmax(areas))]
        # Add 25% padding around the box so arms and the bat stay inside the crop.
        pad_x, pad_y = 0.25 * (x2 - x1), 0.25 * (y2 - y1)
        # Frame height and width, for clamping the crop to the image.
        h, w = frame.shape[:2]
        # Crop corners, clamped so they stay inside the frame.
        cx1, cy1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
        # Bottom-right corner, clamped.
        cx2, cy2 = int(min(w, x2 + pad_x)), int(min(h, y2 + pad_y))
        # Second pass on the crop.
        cboxes, cxy, cconf = self._run(frame[cy1:cy2, cx1:cx2], self.crop_imgsz)
        # The crop pass can fail on a blurred frame.
        if cboxes is None:
            # Signal missing.
            return None
        # Largest person in the crop is the batter.
        j = int(np.argmax((cboxes[:, 2] - cboxes[:, 0]) * (cboxes[:, 3] - cboxes[:, 1])))
        # Shift crop coordinates back to full-frame coordinates.
        offset = np.array([cx1, cy1], dtype=float)
        # Person box in full-frame coordinates.
        box = tuple(float(v) for v in (cboxes[j] + np.r_[offset, offset]))
        # Package the result.
        return Keypoints(cxy[j] + offset, cconf[j], box)
