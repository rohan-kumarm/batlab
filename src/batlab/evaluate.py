"""Compares each pose model against hand-marked points (labels.json from the marking page)."""

# numpy for distances and averages.
import numpy as np

# Calibration (scale and stumps) and the batter region.
from .calibrate import Calibration, batter_roi
# The checks, reused so true values and model values are computed the same way.
from .checks_front import _head, front_checks
# Helmet-based head tilt detector.
from .head_tilt import helmet_tilt
# Batting-hand directions.
from .geometry import Sides
# The keypoints record.
from .pose import NAMES, Keypoints
# Frame reading.
from .video import VideoInfo, read_frames

# Checks that can be worked out from the marked points (balance needs the whole body, so it is
# left out; head stillness is handled separately because it compares frames).
CHECKS = ["eyes_level", "head_tilt_helmet", "head_on_middle_off", "front_knee_inside_off_stump", "hands_under_head"]


def marks_to_keypoints(points: dict, sides: Sides) -> Keypoints:
    """Turn one frame's marks into a Keypoints record so the normal check code can use them."""
    # Start with every standard point missing (zero confidence).
    xy, conf = np.zeros((len(NAMES), 2)), np.zeros(len(NAMES))

    def put(name: str, p) -> None:
        """Store a marked point under a standard name, if it was marked."""
        # Skipped ("can't see it") or unmarked points stay missing.
        if p:
            # Position.
            xy[NAMES.index(name)] = p
            # Full confidence: a human placed it.
            conf[NAMES.index(name)] = 1.0

    # Eyes: the checks only use their order across the picture, so picture-left/right is enough.
    put("left_eye", points.get("eye_left_in_picture"))
    # Second eye.
    put("right_eye", points.get("eye_right_in_picture"))
    # Hands: both wrists at the marked centre of the hands.
    put("left_wrist", points.get("hands"))
    # Same point for the other wrist, so their midpoint is the marked point.
    put("right_wrist", points.get("hands"))
    # Front knee under the batting hand's front-leg name.
    put(f"{sides.front}_knee", points.get("front_knee"))
    # Shoe edges as extra foot points.
    extra = {k: (np.array(points[k], dtype=float), 1.0) for k in ("feet_left_edge", "feet_right_edge") if points.get(k)}
    # Box is not used by the checks.
    return Keypoints(xy, conf, (0, 0, 0, 0), extra)


def point_errors(kp: Keypoints, marks: dict, sides: Sides, cm_per_px: float) -> dict[str, float | None]:
    """Distance in cm between the model's points and the marked points, per point."""
    # Results per point.
    out: dict[str, float | None] = {}

    def dist(a, b) -> float | None:
        """Distance in cm between two points, or None if either is missing."""
        # Missing on either side means no comparison.
        if a is None or b is None:
            # Signal missing.
            return None
        # Straight-line distance in pixels, converted to cm.
        return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) * cm_per_px)

    # Eye midpoint: compared as a midpoint so left/right label swaps do not count as errors.
    me = kp.midpoint("left_eye", "right_eye")
    # Marked eyes.
    a, b = marks.get("eye_left_in_picture"), marks.get("eye_right_in_picture")
    # Midpoint of the marked eyes, if both were marked.
    out["head (eye midpoint)"] = dist(me, (np.array(a) + np.array(b)) / 2 if a and b else None)
    # Eye-line tilt difference in degrees, comparing eyes in picture order (label swaps do not matter).
    def tilt(p, q) -> float:
        """Angle of the line from the left-in-picture eye to the right one, degrees."""
        # Sort the two points left to right.
        l, r = sorted([np.asarray(p, float), np.asarray(q, float)], key=lambda v: v[0])
        # Angle from the horizontal; image y grows downward, which flips sign the same way for both.
        return float(np.degrees(np.arctan2(r[1] - l[1], r[0] - l[0])))
    # Model eyes.
    le, re_ = kp.point("left_eye"), kp.point("right_eye")
    # Difference when both the model's and the marked eyes exist.
    out["eye tilt (deg)"] = abs(tilt(le, re_) - tilt(a, b)) if le is not None and re_ is not None and a and b else None
    # Marked ends of the helmet grille's top bar: a longer, clearer line that tilts with the head.
    g1, g2 = marks.get("grille_left_end"), marks.get("grille_right_end")
    # The model's eye-line tilt against the marked helmet tilt: how well the engine's head tilt matches.
    out["head tilt vs helmet marks (deg)"] = (abs(tilt(le, re_) - tilt(g1, g2))
                                              if le is not None and re_ is not None and g1 and g2 else None)
    # Model hands: midpoint of wrists, or the one visible wrist (as in the checks).
    hands = kp.midpoint("left_wrist", "right_wrist")
    # Fall back to a single wrist.
    if hands is None:
        # First visible wrist, if any.
        hands = kp.point("left_wrist") if kp.point("left_wrist") is not None else kp.point("right_wrist")
    # Hands error.
    out["hands"] = dist(hands, marks.get("hands"))
    # Front knee error (uses the model's own left/right label for the front leg).
    out["front knee"] = dist(kp.point(f"{sides.front}_knee"), marks.get("front_knee"))
    # Model foot points, for the outer edges of the base.
    feet = [kp.point(n) for n in ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_toe", "right_toe")]
    # Keep the ones found.
    feet = [p for p in feet if p is not None]
    # Sideways error of the left-most foot point (only x matters for the base).
    out["feet left edge (sideways)"] = (abs(min(p[0] for p in feet) - marks["feet_left_edge"][0]) * cm_per_px
                                        if feet and marks.get("feet_left_edge") else None)
    # Sideways error of the right-most foot point.
    out["feet right edge (sideways)"] = (abs(max(p[0] for p in feet) - marks["feet_right_edge"][0]) * cm_per_px
                                         if feet and marks.get("feet_right_edge") else None)
    # Return all point errors.
    return out


def head_movement(kps_by_back: dict[int, Keypoints | None], cm_per_px: float) -> float | None:
    """Largest distance (cm) the eye midpoint moves from the earliest marked frame to later ones."""
    # Eye midpoints in time order (most frames before release first).
    heads = [kps_by_back[b].midpoint("left_eye", "right_eye") if kps_by_back.get(b) else None
             for b in sorted(kps_by_back, reverse=True)]
    # Need the earliest frame and at least one later frame.
    if heads[0] is None or all(h is None for h in heads[1:]):
        # Not measurable.
        return None
    # Distance of each later head position from the first, in cm; keep the largest.
    return max(float(np.linalg.norm(h - heads[0]) * cm_per_px) for h in heads[1:] if h is not None)


def evaluate(info: VideoInfo, cal: Calibration, sides: Sides, labels: dict, models: dict) -> dict:
    """Run every model on every marked frame and compare with the marks.

    `models` maps a name ("yolo", "mediapipe") to a loaded pose adapter.
    Returns per-point mean errors and per-check true-vs-model values.
    """
    # Region around the batter, as in the main pipeline.
    roi = batter_roi(cal)
    # Per model: point name -> list of errors (cm).
    errors = {m: {} for m in models}
    # Per source ("marks" and each model): check name -> list of values from release frames.
    values = {s: {c: [] for c in CHECKS + ["head_movement"]} for s in ["marks", *models]}
    # Group marked frames by delivery, for head movement.
    by_delivery: dict[int, dict] = {}
    # Go through every marked frame.
    for fr in labels["frames"]:
        # Read the frame.
        frame = read_frames(info, fr["frame"] / info.fps, 1)[0]
        # True keypoints from the marks.
        truth = marks_to_keypoints(fr["points"], sides)
        # This delivery's entry, created on first use.
        entry = by_delivery.setdefault(fr["delivery"], {s: {} for s in ["marks", *models]})
        # Store the marks for head movement, keyed by frames before release.
        entry["marks"][fr["frames_before_release"]] = truth
        # The release frame gives the check values.
        if fr["frames_before_release"] == 0:
            # Checks from the marks.
            for c, v in front_checks(truth, cal, sides).items():
                # Keep only the checks the marks can support, when measured.
                if c in CHECKS and v is not None:
                    # Store the true value.
                    values["marks"][c].append(v)
            # Head tilt from the grille marks: treat the grille ends as the "eyes" and reuse eyes_level.
            g = fr["points"]
            # Only when both grille ends were marked.
            if g.get("grille_left_end") and g.get("grille_right_end"):
                # Same marks, with the grille ends in place of the eyes.
                helmet = marks_to_keypoints({**g, "eye_left_in_picture": g["grille_left_end"],
                                             "eye_right_in_picture": g["grille_right_end"]}, sides)
                # Its eyes_level is the helmet tilt.
                values["marks"]["head_tilt_helmet"].append(front_checks(helmet, cal, sides)["eyes_level"])
        # Each model on the same frame.
        for m, pose in models.items():
            # Run the model.
            kp = pose.estimate(frame, roi)
            # Store for head movement.
            entry[m][fr["frames_before_release"]] = kp
            # Nothing found: no errors to record for this frame.
            if kp is None:
                # Next model.
                continue
            # Helmet detector, placed at this model's head point.
            head = _head(kp)
            # Detector angle (picture convention), or None.
            det = helmet_tilt(frame, head, cal.cm_per_px) if head is not None else None
            # Marked grille ends.
            g1, g2 = fr["points"].get("grille_left_end"), fr["points"].get("grille_right_end")
            # Compare with the marked grille angle when both exist.
            if det is not None and g1 and g2:
                # Marked angle, same convention.
                marked = float(np.degrees(np.arctan2(g2[1] - g1[1], g2[0] - g1[0])))
                # Store the absolute difference.
                errors[m].setdefault("head tilt (helmet detector, deg)", []).append(abs(det - marked))
            # On release frames, the detector gives the engine's head-tilt check value.
            if det is not None and fr["frames_before_release"] == 0:
                # Same sign convention as the marks' value (positive = tipping toward off side).
                values[m]["head_tilt_helmet"].append(det * sides.off_dir)
            # Point errors against the marks.
            for p, e in point_errors(kp, fr["points"], sides, cal.cm_per_px).items():
                # Collect measured errors only.
                if e is not None:
                    # Append to this model's list for this point.
                    errors[m].setdefault(p, []).append(e)
            # Model check values on release frames.
            if fr["frames_before_release"] == 0:
                # Same check code as for the marks.
                for c, v in front_checks(kp, cal, sides).items():
                    # Keep the comparable checks that were measured.
                    if c in CHECKS and v is not None:
                        # Store the model's value.
                        values[m][c].append(v)

    # Head movement per delivery, for the marks and each model.
    for entry in by_delivery.values():
        # Each source.
        for s, kps in entry.items():
            # Movement across this delivery's marked frames.
            mv = head_movement(kps, cal.cm_per_px)
            # Store it if measured.
            if mv is not None:
                # Append to the list.
                values[s]["head_movement"].append(mv)
    # Count of marked frames and of frames where each model found the batter.
    found = {m: sum(1 for e in by_delivery.values() for k in e[m].values() if k is not None) for m in models}
    # Return the raw lists so results from several clips can be combined before averaging.
    return {"errors": errors, "values": values, "frames": len(labels["frames"]), "found": found}


def combine(raws: list[dict]) -> dict:
    """Merge raw results from several clips and average them."""
    # Model names from the first clip (every clip is run with the same models).
    models = list(raws[0]["errors"])
    # Merged error lists per model and point.
    errors = {m: {} for m in models}
    # Merged value lists per source and check.
    values = {s: {} for s in raws[0]["values"]}
    # Add each clip's lists.
    for r in raws:
        # Errors.
        for m in models:
            # Each point's list.
            for p, v in r["errors"][m].items():
                # Extend the merged list.
                errors[m].setdefault(p, []).extend(v)
        # Check values.
        for src, cs in r["values"].items():
            # Each check's list.
            for c, v in cs.items():
                # Extend the merged list.
                values[src].setdefault(c, []).extend(v)
    # Average error per point, with how many frames it is based on.
    mean_err = {m: {p: (float(np.mean(v)), len(v)) for p, v in pts.items()} for m, pts in errors.items()}
    # Average value per check and source.
    mean_val = {s: {c: (float(np.mean(v)) if v else None, len(v)) for c, v in cs.items()} for s, cs in values.items()}
    # Detection rate per model.
    found = {m: (sum(r["found"][m] for r in raws), sum(r["frames"] for r in raws)) for m in models}
    # Return the averaged tables.
    return {"point_error_cm": mean_err, "check_values": mean_val, "found": found, "clips": len(raws)}


def format_report(result: dict, reference: dict | None, targets: dict | None = None, title: str = "") -> str:
    """Markdown tables of point errors (with pass/fail against targets) and check values."""
    # Model names in order.
    models = list(result["point_error_cm"])
    # Every point name any model was scored on.
    points = sorted({p for m in models for p in result["point_error_cm"][m]})
    # Section heading with the amount of data behind it.
    lines = [f"# {title}", "", f"Clips: {result['clips']}. Batter found: " +
             ", ".join(f"{m} {f}/{n} frames" for m, (f, n) in result["found"].items()), ""]
    # First table: how far each model's points are from your marks, and whether that meets the target.
    lines += ["## Point error vs your marks (lower is better)", "",
              "| Point | Target | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
    # One row per point.
    for p in points:
        # This point's target, if one is set.
        t = targets.get(p) if targets else None

        def cell(m: str) -> str:
            """'mean (n) PASS/FAIL' for one model, or a dash."""
            # No result for this model and point.
            if p not in result["point_error_cm"][m]:
                # Dash.
                return "-"
            # Mean error and frame count.
            v, n = result["point_error_cm"][m][p]
            # Pass or fail against the target, if there is one.
            mark = "" if t is None else (" PASS" if v <= t else " FAIL")
            # Formatted cell.
            return f"{v:.1f} ({n}){mark}"
        # Add the row.
        lines.append(f"| {p} | {'-' if t is None else t} | " + " | ".join(cell(m) for m in models) + " |")
    # Second table: check values from your marks next to each model and WillowCoach.
    lines += ["", "## Check values on release frames", "",
              "| Check | Your marks | " + " | ".join(models) + " | WillowCoach |",
              "|---|---|" + "---|" * len(models) + "---|"]
    # WillowCoach names for the comparable checks (head movement matches their head stillness).
    wc_name = {"head_movement": "head_stillness_front", "head_tilt_helmet": "head_tilt"}
    # One row per check.
    for c in CHECKS + ["head_movement"]:
        # Format one source's value.
        fmt = lambda s: "-" if result["check_values"][s][c][0] is None else f"{result['check_values'][s][c][0]:+.1f}"
        # WillowCoach's note for this check, if any.
        wc = reference["front"].get(wc_name.get(c, c), {}).get("note", "-") if reference else "-"
        # Add the row.
        lines.append(f"| {c} | {fmt('marks')} | " + " | ".join(fmt(m) for m in models) + f" | {wc} |")
    # Explain the numbers in brackets and head movement.
    lines += ["", "Numbers in brackets: how many frames the error is averaged over.",
              "head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery "
              "(0.2 s before release to release), averaged over deliveries."]
    # Join into one text.
    return "\n".join(lines) + "\n"
