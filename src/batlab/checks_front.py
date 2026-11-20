"""The front-on setup checks, measured at the release frame.

Sideways distances are in cm with a sign: positive means toward the batter's off side.
"""

# numpy for vector math.
import numpy as np

# Calibration holds the scale and stump positions.
from .calibrate import Calibration
# Sides holds the batting-hand directions.
from .geometry import Sides
# Keypoints is one pose result.
from .pose import Keypoints


def _lateral_cm(x_px: float, ref_px: float, cal: Calibration, sides: Sides) -> float:
    """Sideways distance from ref to x in cm, positive toward the off side."""
    # Pixel difference, times cm per pixel, times the off-side direction sign.
    return (x_px - ref_px) * cal.cm_per_px * sides.off_dir


def _head(kp: Keypoints) -> np.ndarray | None:
    """Head position: midpoint of the eyes, or the nose if the eyes are not both visible."""
    # Midpoint of the eyes, if both were found.
    eyes = kp.midpoint("left_eye", "right_eye")
    # Use the nose as the fallback.
    return eyes if eyes is not None else kp.point("nose")


# Share of body mass in each segment, with where along the segment its centre sits (0 = first
# point, 1 = second point). Values are rounded from standard adult tables (Dempster / de Leva).
# Each entry: (mass share, first point, second point, position along segment).
SEGMENTS = [
    # Trunk: shoulder midpoint to hip midpoint, centre about halfway.
    (0.497, "shoulder_mid", "hip_mid", 0.50),
    # Head and neck: centred on the eyes.
    (0.081, "head", "head", 0.0),
    # Upper arms: shoulder to elbow, centre 44% of the way down.
    (0.028, "left_shoulder", "left_elbow", 0.44),
    (0.028, "right_shoulder", "right_elbow", 0.44),
    # Forearm plus hand: elbow to wrist, centre 68% of the way (the hand adds weight at the end).
    (0.022, "left_elbow", "left_wrist", 0.68),
    (0.022, "right_elbow", "right_wrist", 0.68),
    # Thighs: hip to knee, centre 43% of the way down.
    (0.100, "left_hip", "left_knee", 0.43),
    (0.100, "right_hip", "right_knee", 0.43),
    # Shank plus foot: knee to ankle, centre 60% of the way down.
    (0.061, "left_knee", "left_ankle", 0.60),
    (0.061, "right_knee", "right_ankle", 0.60),
]


def centre_of_mass(kp: Keypoints) -> np.ndarray | None:
    """Estimate the whole body's centre of mass in image pixels from the segment table."""
    # Points that are not single keypoints, built from pairs.
    extra = {"shoulder_mid": kp.midpoint("left_shoulder", "right_shoulder"),
             "hip_mid": kp.midpoint("left_hip", "right_hip"), "head": _head(kp)}
    # Running weighted sum of positions and of weights.
    total, weight = np.zeros(2), 0.0
    # Add each segment's centre, weighted by its mass share.
    for share, a, b, frac in SEGMENTS:
        # Look up both ends, from the extra points or the keypoints.
        pa = extra[a] if a in extra else kp.point(a)
        # Second end.
        pb = extra[b] if b in extra else kp.point(b)
        # Front-on, the back leg stands directly behind the front leg, so the pose model often
        # cannot see it. Its sideways position is then about the same as the visible leg's, so a
        # hidden leg segment borrows the other leg's matching segment.
        if (pa is None or pb is None) and a.endswith(("_hip", "_knee")):
            # The other side's name, e.g. "left_knee" -> "right_knee".
            swap = (lambda n: n.replace("left_", "right_") if n.startswith("left_") else n.replace("right_", "left_"))
            # Look up the mirrored segment's two ends.
            pa, pb = kp.point(swap(a)), kp.point(swap(b))
        # Skip a segment with a missing end (front-on, the back arm is usually hidden).
        if pa is None or pb is None:
            # Leave this segment out of the average.
            continue
        # Segment centre: the point `frac` of the way from pa to pb.
        total += share * (pa + frac * (pb - pa))
        # Add the mass share.
        weight += share
    # Allow at most 12% of body mass to be missing (one whole arm is 5%; both arms are 10%).
    # More than that means the trunk, head or a leg is missing, and the estimate is unreliable.
    if weight < 0.88:
        # Signal missing.
        return None
    # Divide by the share actually found, so the result is the average of the visible segments.
    return total / weight


def front_checks(kp: Keypoints, cal: Calibration, sides: Sides) -> dict[str, float | None]:
    """Compute the five single-frame front-on checks; None means a needed point was missing."""
    # Results, filled in below.
    out: dict[str, float | None] = {}
    # Head position.
    head = _head(kp)
    # Both eyes, for the tilt.
    le, re = kp.point("left_eye"), kp.point("right_eye")
    # Middle stump x.
    middle_x = cal.stump_x[1]
    # Off stump x.
    off_x = cal.stump_x[sides.off_stump]

    # 1. Eyes level: angle of the eye line, positive when the off-side eye is lower (head tipping to off).
    if le is not None and re is not None:
        # Order the eyes so `off_eye` is the one further toward the off side in the image.
        off_eye, leg_eye = (le, re) if (le[0] - re[0]) * sides.off_dir > 0 else (re, le)
        # Image y grows downward, so a lower off-side eye gives a positive drop.
        drop = off_eye[1] - leg_eye[1]
        # Sideways gap between the eyes in pixels (always positive).
        gap = abs(off_eye[0] - leg_eye[0])
        # Angle in degrees from the drop and the gap.
        out["eyes_level"] = float(np.degrees(np.arctan2(drop, gap)))
    else:
        # An eye is missing; the check cannot be measured.
        out["eyes_level"] = None

    # 2. Head on middle/off: 0 while the head is between the off and middle stump lines; otherwise
    #    the distance to the nearer of those lines, positive if outside off, negative if past middle.
    if head is not None:
        # Head's sideways position measured from the off stump (positive = further toward off).
        from_off = _lateral_cm(head[0], off_x, cal, sides)
        # Middle stump's sideways position measured from the off stump (negative: it is on the leg side).
        middle_from_off = _lateral_cm(middle_x, off_x, cal, sides)
        # Outside off stump: report how far.
        if from_off > 0:
            # Positive distance toward off.
            out["head_on_middle_off"] = from_off
        # Past the middle stump toward leg: report how far, as a negative number.
        elif from_off < middle_from_off:
            # Negative distance toward leg.
            out["head_on_middle_off"] = from_off - middle_from_off
        else:
            # Inside the middle/off zone.
            out["head_on_middle_off"] = 0.0
    else:
        # Head not found.
        out["head_on_middle_off"] = None

    # 3. Side-to-side balance: how far the centre of mass sits inside the outer foot lines.
    com = centre_of_mass(kp)
    # Foot points that mark the base: ankles always; heels and toes when the model gives them.
    foot = [kp.point(n) for n in ("left_ankle", "right_ankle", "left_heel", "right_heel", "left_toe", "right_toe")]
    # Keep the ones that were found.
    foot = [p for p in foot if p is not None]
    # Measurable when the centre of mass and at least both ankles' worth of points exist.
    if com is not None and len(foot) >= 2:
        # Outer edges of the base in image x: the furthest-left and furthest-right foot points.
        lo, hi = min(p[0] for p in foot), max(p[0] for p in foot)
        # Margin to the nearer edge in pixels; negative means the centre of mass is outside the base.
        margin_px = min(com[0] - lo, hi - com[0])
        # Convert to cm.
        out["balance_side_to_side"] = float(margin_px * cal.cm_per_px)
    else:
        # A needed point is missing.
        out["balance_side_to_side"] = None

    # 4. Front knee inside off stump: positive when the knee is on the leg side of the off-stump line.
    knee = kp.point(f"{sides.front}_knee")
    # The minus sign turns "toward off" into "inside" (toward leg).
    out["front_knee_inside_off_stump"] = -_lateral_cm(knee[0], off_x, cal, sides) if knee is not None else None

    # Hands: midpoint of the two wrists; if one is hidden (common front-on), the visible one,
    # since both hands hold the bat handle a few cm apart.
    hands = kp.midpoint("left_wrist", "right_wrist")
    # Fall back to whichever single wrist was found.
    if hands is None:
        # First visible wrist, or None if neither is.
        hands = kp.point("left_wrist") if kp.point("left_wrist") is not None else kp.point("right_wrist")
    # 5. Hands under head: sideways distance of the hands from a vertical line through the head.
    #    ("Hands centred" is measured side-on, so it is not in this file.)
    out["hands_under_head"] = _lateral_cm(hands[0], head[0], cal, sides) if hands is not None and head is not None else None
    # Return all five values.
    return out


def head_stillness(kps: list[Keypoints | None], cal: Calibration) -> float | None:
    """7. Head stillness: largest distance (cm) the head moves from where it was at the window start."""
    # Head positions in the frames where the head was found.
    heads = [h for h in (_head(k) for k in kps if k is not None) if h is not None]
    # Need at least half the window's frames to trust the result.
    if len(heads) < max(3, len(kps) // 2):
        # Not enough frames.
        return None
    # Stack into an (n, 2) array.
    pts = np.array(heads)
    # Distance of every frame's head from the first frame's head, in pixels.
    dist_px = np.linalg.norm(pts - pts[0], axis=1)
    # Largest distance, converted to cm.
    return float(dist_px.max() * cal.cm_per_px)


def grade(name: str, value: float | None, targets: dict) -> str:
    """Return 'textbook', 'near textbook', 'unorthodox', or 'not measured'."""
    # Missing values cannot be graded.
    if value is None:
        # Fixed label for missing values.
        return "not measured"
    # Target settings for this check.
    t = targets[name]
    # Checks where a larger positive value is better (front knee inside off stump).
    if t.get("inside_is_good"):
        # At or above the good line is textbook.
        if value >= t["good"]:
            # Best grade.
            return "textbook"
        # Between ok and good is near textbook; below ok is unorthodox.
        return "near textbook" if value >= t["ok"] else "unorthodox"
    # All other checks: the size of the value matters, not its sign.
    size = abs(value)
    # Within the good limit.
    if size <= t["good"]:
        # Best grade.
        return "textbook"
    # Within the ok limit, otherwise unorthodox.
    return "near textbook" if size <= t["ok"] else "unorthodox"
