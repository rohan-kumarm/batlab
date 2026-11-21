"""Writes the run's outputs: annotated release frames, a CSV, a JSON file and a summary table."""

# csv writes the per-delivery table.
import csv
# json writes the full machine-readable result.
import json
# Path builds output file paths.
from pathlib import Path

# OpenCV draws on and saves images.
import cv2
# numpy for the summary statistics.
import numpy as np

# Calibration, for drawing the stump lines.
from .calibrate import Calibration
# Sides, for knowing which stump is off stump.
from .geometry import Sides
# Keypoints, for drawing the skeleton.
from .pose import Keypoints

# Pairs of keypoints joined by lines when drawing the skeleton.
BONES = [("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"),
         ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
         ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
         ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
         ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
         ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
         ("left_eye", "right_eye"),
         # Feet: drawn only when the model gives heel and toe points (MediaPipe).
         ("left_ankle", "left_heel"), ("left_heel", "left_toe"), ("left_ankle", "left_toe"),
         ("right_ankle", "right_heel"), ("right_heel", "right_toe"), ("right_ankle", "right_toe")]
# Order and display names of the checks in every output.
CHECK_ORDER = ["eyes_level", "head_tilt", "head_on_middle_off", "balance_side_to_side",
               "front_knee_inside_off_stump", "hands_under_head", "head_stillness_front"]


def draw_release(frame: np.ndarray, kp: Keypoints | None, cal: Calibration, sides: Sides,
                 lines: list[str], path: Path) -> None:
    """Save a cropped copy of the release frame with the skeleton, stump lines and check values."""
    # Draw on a copy so the original frame stays unchanged.
    img = frame.copy()
    # Off stump x.
    off_x = cal.stump_x[sides.off_stump]
    # Draw the off-stump line in red, full image height.
    cv2.line(img, (int(off_x), 0), (int(off_x), img.shape[0]), (0, 0, 255), 2)
    # Draw the middle-stump line in yellow; the head zone lies between the red and yellow lines.
    cv2.line(img, (int(cal.stump_x[1]), 0), (int(cal.stump_x[1]), img.shape[0]), (0, 255, 255), 2)
    # Draw the skeleton only if a pose was found.
    if kp is not None:
        # Each bone is a pair of point names.
        for a, b in BONES:
            # Both end points, or None if missing or unsure (also None for feet with YOLO11).
            pa, pb = kp.point(a), kp.point(b)
            # Skip bones with a missing end point.
            if pa is None or pb is None:
                # Next bone.
                continue
            # Draw the bone in green.
            cv2.line(img, tuple(pa.astype(int)), tuple(pb.astype(int)), (0, 220, 0), 4)
        # Crop around the person box and the stumps, with a margin.
        x1, y1, x2, y2 = kp.box
        # Left edge: whichever is further left, the person or the stumps, minus a margin.
        left = int(max(0, min(x1, cal.stump_x[0]) - 250))
        # Right edge, plus a margin.
        right = int(min(img.shape[1], max(x2, cal.stump_x[2]) + 250))
        # Top edge, above the person.
        top = int(max(0, y1 - 150))
        # Bottom edge, below the stump bases.
        bottom = int(min(img.shape[0], max(y2, cal.stump_base_y) + 150))
        # Apply the crop.
        img = img[top:bottom, left:right].copy()
    # Write each text line near the top-left corner.
    for i, text in enumerate(lines):
        # Black outline first so text reads on any background.
        cv2.putText(img, text, (20, 50 + 42 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        # White text on top.
        cv2.putText(img, text, (20, 50 + 42 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    # Save as JPEG.
    cv2.imwrite(str(path), img)


def summarise(deliveries: list[dict]) -> dict[str, dict]:
    """Average, spread and count of each check across eligible deliveries."""
    # Result per check.
    out = {}
    # Go through checks in display order.
    for name in CHECK_ORDER:
        # Values from eligible deliveries where this check was measured.
        vals = [d["checks"][name] for d in deliveries if d["eligible"] and d["checks"].get(name) is not None]
        # Mean and standard deviation, or None if nothing was measured.
        out[name] = {"mean": float(np.mean(vals)) if vals else None,
                     "spread": float(np.std(vals)) if len(vals) > 1 else None,
                     "n": len(vals)}
    # Return the per-check summary.
    return out


def write_outputs(out_dir: Path, result: dict, reference: dict | None, targets: dict, grade_fn) -> str:
    """Write report.json, deliveries.csv and summary.md; return the summary text."""
    # Make sure the output folder exists.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Full result as indented JSON.
    (out_dir / "report.json").write_text(json.dumps(result, indent=2))
    # Per-delivery CSV.
    with open(out_dir / "deliveries.csv", "w", newline="") as f:
        # CSV writer.
        w = csv.writer(f)
        # Header row.
        w.writerow(["delivery", "release_s", "release_frame", "eligible", *CHECK_ORDER])
        # One row per delivery.
        for d in result["deliveries"]:
            # Round numbers to 1 decimal; blanks for missing.
            vals = ["" if d["checks"].get(c) is None else round(d["checks"][c], 1) for c in CHECK_ORDER]
            # Write the row.
            w.writerow([d["index"], round(d["release_s"], 3), d["release_frame"], d["eligible"], *vals])
    # Build the Markdown summary table.
    rows = ["| Check | Ours (mean) | Spread | n | Grade | WillowCoach |", "|---|---|---|---|---|---|"]
    # One row per check.
    for name in CHECK_ORDER:
        # This check's summary.
        s = result["summary"][name]
        # Formatted mean, or a dash.
        mean = "-" if s["mean"] is None else f"{s['mean']:+.1f} {targets[name]['unit']}"
        # Formatted spread, or a dash.
        spread = "-" if s["spread"] is None else f"{s['spread']:.1f}"
        # WillowCoach's value as they wrote it, if a reference was given.
        ref = reference["front"].get(name, {}).get("note", "-") if reference else "-"
        # Add the row.
        rows.append(f"| {name} | {mean} | {spread} | {s['n']} | {grade_fn(name, s['mean'], targets)} | {ref} |")
    # Join into one text block.
    text = "\n".join(rows) + "\n"
    # Save it.
    (out_dir / "summary.md").write_text(text)
    # Return it for printing.
    return text
