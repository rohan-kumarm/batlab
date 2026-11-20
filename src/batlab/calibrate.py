"""Finds the stumps in the front-on view and turns pixels into centimetres."""

# dataclass for the result record.
from dataclasses import dataclass
# OpenCV for colour conversion and blob finding.
import cv2
# numpy for array math and medians.
import numpy as np

# VideoInfo and read_frames come from this package's video module.
from .video import VideoInfo, read_frames


# Result of calibrating one camera.
@dataclass(frozen=True)
class Calibration:
    # Centimetres per pixel at the stumps' distance from the camera.
    cm_per_px: float
    # x position (full-resolution pixels) of each stump's centre, sorted left to right in the image.
    stump_x: tuple[float, float, float]
    # y position of the stump tops.
    stump_top_y: float
    # y position of the stump bases (where they meet the ground).
    stump_base_y: float
    # Number of video frames whose stump measurements were combined.
    frames_used: int


def stump_boxes(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return (x, y, w, h) boxes of tall, thin red/orange shapes: the stump candidates."""
    # HSV separates colour (hue) from brightness, which makes colour masks reliable.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Red/orange sits at both ends of OpenCV's 0-180 hue range, so two ranges are combined.
    mask = cv2.inRange(hsv, (0, 140, 120), (12, 255, 255)) | cv2.inRange(hsv, (170, 140, 120), (180, 255, 255))
    # Label each connected group of mask pixels and get its bounding box.
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    # Smallest stump height allowed: 3% of the frame height (drops balls and specks).
    min_h = 0.03 * frame.shape[0]
    # Keep boxes at least 3 times taller than wide and tall enough; label 0 is background.
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n)
             if stats[i, 3] > 3 * stats[i, 2] and stats[i, 3] > min_h]
    # A pad in front of a stump can split it into two pieces; merge boxes whose x ranges overlap.
    merged: list[list[int]] = []
    # Go through boxes from left to right.
    for x, y, w, h in sorted(boxes):
        # If this box overlaps the previous merged box sideways, it is the same stump.
        if merged and x < merged[-1][0] + merged[-1][2]:
            # Previous box, unpacked.
            mx, my, mw, mh = merged[-1]
            # New top is the higher of the two tops.
            top = min(my, y)
            # New bottom is the lower of the two bottoms.
            bottom = max(my + mh, y + h)
            # New right edge is the further right of the two.
            right = max(mx + mw, x + w)
            # Replace the previous box with the combined one.
            merged[-1] = [mx, top, right - mx, bottom - top]
        else:
            # No overlap: this is a new stump candidate.
            merged.append([x, y, w, h])
    # Return as tuples.
    return [tuple(b) for b in merged]


def cluster_stumps(boxes: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group stump boxes collected from many frames into one group per stump, by x position."""
    # Typical stump width in pixels (median is not thrown off by a few odd boxes).
    width = float(np.median([b[2] for b in boxes]))
    # Keep boxes of normal stump width; slivers and stump-plus-pad blobs give wrong positions.
    good = [b for b in boxes if 0.5 * width <= b[2] <= 1.5 * width]
    # Sort by centre x so neighbouring boxes are next to each other.
    good.sort(key=lambda b: b[0] + b[2] / 2)
    # Start with one empty group.
    groups: list[list] = []
    # Centre x of the previous box.
    prev = None
    # Walk through boxes left to right.
    for b in good:
        # Centre x of this box.
        cx = b[0] + b[2] / 2
        # A gap wider than 0.8 stump widths means a new stump starts here.
        if prev is None or cx - prev > 0.8 * width:
            # Open a new group.
            groups.append([])
        # Add the box to the current group.
        groups[-1].append(b)
        # Remember this centre for the next comparison.
        prev = cx
    # Keep the 3 groups seen in the most frames (the real stumps), then sort them left to right.
    top3 = sorted(groups, key=len, reverse=True)[:3]
    # Return in image order, left to right.
    return sorted(top3, key=lambda g: np.median([b[0] + b[2] / 2 for b in g]))


def calibrate_front(info: VideoInfo, stump_height_cm: float, step_s: float = 0.5,
                    scan_width: int = 1080) -> Calibration:
    """Measure the stumps across the clip and combine the views into one calibration.

    The batter's pads hide part of the stumps in most frames, but the camera and stumps do
    not move, so each stump's true top and base are the extremes seen across all frames.
    """
    # Factor to convert scan-size pixels back to full-size pixels.
    k = info.width / scan_width
    # Every stump box from every sampled frame.
    boxes: list[tuple[int, int, int, int]] = []
    # Number of frames that contained at least one stump box.
    frames_used = 0
    # Sample one frame every step_s seconds across the clip.
    for t in np.arange(0.0, info.duration - 0.1, step_s):
        # Read a single downscaled frame (fast).
        frame = read_frames(info, float(t), 1, scale_width=scan_width)
        # Skip if the read returned nothing (end of clip).
        if len(frame) == 0:
            # Next time point.
            continue
        # Stump candidates in this frame.
        found = stump_boxes(frame[0])
        # Count the frame if anything was found.
        frames_used += bool(found)
        # Add them to the pool.
        boxes.extend(found)
    # Group the pooled boxes into stumps.
    groups = cluster_stumps(boxes) if boxes else []
    # Without three stumps the off-stump line and scale cannot be set; stop with a clear message.
    if len(groups) < 3:
        # Explain the likely cause.
        raise RuntimeError("Could not find 3 stumps; check all three are in frame and the camera is still.")
    # Each stump's x: median centre over the frames, converted to full-size pixels.
    stump_x = tuple(float(np.median([b[0] + b[2] / 2 for b in g])) * k for g in groups)
    # Each stump's top: 5th percentile of tops, which ignores rare mask errors above the stump.
    tops = [np.percentile([b[1] for b in g], 5) for g in groups]
    # Each stump's base: 95th percentile of bottoms, i.e. the frames where the base was not hidden.
    bases = [np.percentile([b[1] + b[3] for b in g], 95) for g in groups]
    # Use the median stump so one badly seen stump does not set the scale.
    top, base = float(np.median(tops)) * k, float(np.median(bases)) * k
    # Scale: known stump height divided by its height in pixels.
    return Calibration(stump_height_cm / (base - top), stump_x, top, base, frames_used)
