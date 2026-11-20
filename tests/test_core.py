"""Unit tests for the parts that do not need video: run with `pytest`."""

# numpy builds the fake keypoints and audio.
import numpy as np
# pytest.approx compares decimals with a tolerance.
import pytest

# Modules under test.
from batlab.calibrate import Calibration, cluster_stumps
from batlab.checks_front import front_checks, grade
from batlab.geometry import front_on_sides
from batlab.pose import IDX, Keypoints
from batlab.release import find_releases


def test_left_hander_is_mirror_of_right_hander():
    # Right-hander directions.
    r = front_on_sides("right")
    # Left-hander directions.
    l = front_on_sides("left")
    # Off side flips, front leg flips, off stump flips.
    assert (r.off_dir, r.front, r.off_stump) == (-1, "left", 0)
    # Mirror image for the left-hander.
    assert (l.off_dir, l.front, l.off_stump) == (1, "right", 2)


def test_release_found_and_bat_sound_grouped():
    # One second of quiet background noise per 48000 samples; fixed seed for repeatable tests.
    rng = np.random.default_rng(0)
    # Sample rate.
    sr = 48000
    # Six seconds of faint noise.
    audio = rng.normal(0, 1e-4, 6 * sr).astype(np.float32)
    # Machine click at 1.0 s and bat click 0.7 s later, each a 2 ms burst of loud noise.
    for t in (1.0, 1.7):
        # Start sample of the burst.
        s = int(t * sr)
        # Add the loud burst.
        audio[s:s + 96] += rng.normal(0, 0.3, 96)
    # Run detection.
    rel = find_releases(audio, sr)
    # Exactly one delivery: the second click belongs to the first.
    assert len(rel) == 1
    # Release within 5 ms of 1.0 s.
    assert abs(rel[0].time_s - 1.0) < 0.005
    # Bat sound reported about 1.7 s.
    assert rel[0].impact_audio_s is not None and abs(rel[0].impact_audio_s - 1.7) < 0.01


def test_stump_clusters_ignore_slivers():
    # Three 200-px stumps at x = 100, 125, 150 seen in four frames, a 3-px sliver, and a
    # far-off orange object at x = 600 that must not be taken as a stump.
    boxes = ([(100, 0, 10, 200), (125, 0, 10, 200), (150, 0, 10, 200)] * 4
             + [(118, 0, 3, 40)] + [(600, 0, 10, 200)] * 4)
    # Group them.
    groups = cluster_stumps(boxes)
    # Three groups, each from the four frames.
    assert [len(g) for g in groups] == [4, 4, 4]


def _pose(points: dict[str, tuple[float, float]]) -> Keypoints:
    """Build a Keypoints object with the given points confident and the rest missing."""
    # Start with all points at 0 and zero confidence.
    xy, conf = np.zeros((17, 2)), np.zeros(17)
    # Fill in the given points with full confidence.
    for name, p in points.items():
        # Position.
        xy[IDX[name]] = p
        # Confidence.
        conf[IDX[name]] = 1.0
    # Wrap into a Keypoints object.
    return Keypoints(xy, conf, (0, 0, 1, 1))


def test_head_zone_and_knee_for_right_hander():
    # Stumps at x = 100 (off, for a right-hander), 120 (middle), 140 (leg); 1 cm per pixel.
    cal = Calibration(1.0, (100.0, 120.0, 140.0), 0.0, 71.0, 1)
    # Eyes level at x 105-115 (inside the off-to-middle zone); front (left) knee at x 130.
    kp = _pose({"left_eye": (105, 10), "right_eye": (115, 10), "left_knee": (130, 50)})
    # Run the checks for a right-hander.
    out = front_checks(kp, cal, front_on_sides("right"))
    # Head inside the zone gives 0.
    assert out["head_on_middle_off"] == 0.0
    # Knee is 30 px = 30 cm on the leg side of off stump.
    assert out["front_knee_inside_off_stump"] == 30.0
    # Level eyes give 0 degrees.
    assert out["eyes_level"] == 0.0


def test_grading_rules():
    # Targets in the same shape as config/targets.toml.
    targets = {"a": {"good": 5, "ok": 10}, "b": {"good": 0, "ok": -5, "inside_is_good": True}}
    # Size-based grading ignores the sign.
    assert grade("a", -4, targets) == "textbook"
    # Between good and ok.
    assert grade("a", 7, targets) == "near textbook"
    # Inside-is-good grading: slightly outside is near textbook.
    assert grade("b", -2, targets) == "near textbook"
    # Missing values.
    assert grade("a", None, targets) == "not measured"


def test_mediapipe_landmarks_map_to_pixels_and_feet():
    # SimpleNamespace makes small objects with x, y, visibility, like MediaPipe's landmarks.
    from types import SimpleNamespace
    # Converter under test (needs no model file).
    from batlab.pose.mediapipe_pose import landmarks_to_keypoints
    # 33 landmarks at the image centre, all visible.
    lms = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
    # Move the left toe (landmark 31) to the top-left corner of the crop.
    lms[31] = SimpleNamespace(x=0.0, y=0.0, visibility=0.8)
    # Crop is 200 x 100 px and sits at (1000, 2000) in the full frame.
    kp = landmarks_to_keypoints(lms, 200, 100, (1000, 2000))
    # A centre point lands at the crop centre plus the offset.
    assert np.allclose(kp.point("nose"), [1100, 2050])
    # The toe is available as an extra point, at the crop's corner.
    assert np.allclose(kp.point("left_toe"), [1000, 2000])
    # YOLO-style Keypoints have no toes, so the same call returns None there.
    assert _pose({}).point("left_toe") is None


def test_balance_uses_toes_to_widen_base():
    # 1 cm per pixel; stump positions do not matter for this check.
    cal = Calibration(1.0, (0.0, 20.0, 40.0), 0.0, 71.0, 1)
    # Body points stacked on x = 100. The shins run from the knees (100) to the ankles (102, 104),
    # so the centre of mass is 100.22: 0.061 x (0.6 x 2) + 0.061 x (0.6 x 4) = 0.22 cm right of 100.
    names = ["left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
             "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee"]
    # Same x for all of them; y values do not affect a sideways check.
    pts = {n: (100.0, 50.0) for n in names}
    # Ankles at x = 102 and 104: the centre of mass (100) sits outside them.
    pts.update({"left_ankle": (102.0, 90.0), "right_ankle": (104.0, 90.0)})
    # Without toes, the margin is negative: 100.22 - 102 = -1.78 cm (outside the base).
    assert front_checks(_pose(pts), cal, front_on_sides("right"))["balance_side_to_side"] == pytest.approx(-1.78, abs=0.01)
    # Same pose plus a toe at x = 90, which widens the base to 90..104.
    kp = _pose(pts)
    # Keypoints is read-only, so build a new one with the extra toe point.
    kp = Keypoints(kp.xy, kp.conf, kp.box, {"left_toe": (np.array([90.0, 95.0]), 0.9)})
    # Now the centre of mass is inside: 104 - 100.22 = 3.78 cm from the nearer edge.
    assert front_checks(kp, cal, front_on_sides("right"))["balance_side_to_side"] == pytest.approx(3.78, abs=0.01)


def test_mediapipe_runs_on_batter_region_and_maps_back():
    # SimpleNamespace stands in for MediaPipe's landmark objects.
    from types import SimpleNamespace
    # The adapter and the region function under test.
    from batlab.calibrate import batter_roi
    from batlab.pose.mediapipe_pose import MediaPipePose
    # Stumps 100 px tall (top 900, base 1000), middle stump at x = 500.
    cal = Calibration(0.711, (470.0, 500.0, 530.0), 900.0, 1000.0, 1)
    # Region: 1.8 stump heights either side, 3.6 above the base, 0.6 below.
    assert batter_roi(cal) == (320, 640, 680, 1060)
    # Build the adapter without loading a real model (__new__ skips __init__).
    mp_pose = MediaPipePose.__new__(MediaPipePose)
    # Record the size of the image the fake model receives.
    seen = {}

    def fake_run(image):
        # Remember the crop size.
        seen["shape"] = image.shape
        # Return 33 landmarks, all at the crop's centre.
        return [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]

    # Replace the model call with the fake one.
    mp_pose._run = fake_run
    # A blank 2000 x 1200 frame.
    frame = np.zeros((2000, 1200, 3), dtype=np.uint8)
    # Run with the region.
    kp = mp_pose.estimate(frame, batter_roi(cal))
    # The model saw only the 360 x 420 region.
    assert seen["shape"][:2] == (420, 360)
    # The crop's centre maps back to the region's centre in the full frame.
    assert np.allclose(kp.point("nose"), [500, 850])


def test_hidden_back_arm_still_gives_balance_and_hands():
    # 1 cm per pixel; stumps at 0, 20, 40 (off stump at 0 for a right-hander).
    cal = Calibration(1.0, (0.0, 20.0, 40.0), 0.0, 71.0, 1)
    # Everything at x = 100 except the right (back) arm, which is left out as if hidden.
    names = ["left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_elbow", "left_wrist",
             "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"]
    # Build the pose with no right elbow or right wrist.
    kp = _pose({n: (100.0, 50.0) for n in names} | {"left_ankle": (90.0, 90.0), "right_ankle": (110.0, 90.0)})
    # Run the checks.
    out = front_checks(kp, cal, front_on_sides("right"))
    # Balance is still measured: the missing arm is only 5% of body mass.
    assert out["balance_side_to_side"] is not None
    # Hands under head uses the one visible wrist (x 100) against the head (x 100): 0 cm.
    assert out["hands_under_head"] == 0.0


def test_mediapipe_retries_with_square_crop():
    # SimpleNamespace stands in for MediaPipe's landmark objects.
    from types import SimpleNamespace
    # The adapter under test.
    from batlab.pose.mediapipe_pose import MediaPipePose
    # Build the adapter without loading a model.
    mp_pose = MediaPipePose.__new__(MediaPipePose)
    # Shapes of the images the fake model receives, in order.
    seen = []

    def fake_run(image):
        # Record the shape.
        seen.append(image.shape[:2])
        # Fail on the first (non-square) try; succeed on the square try with a point at its centre.
        return None if len(seen) == 1 else [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]

    # Replace the model call.
    mp_pose._run = fake_run
    # Region 100 wide, 200 tall, at (300, 400) in a blank frame.
    kp = mp_pose.estimate(np.zeros((1000, 1000, 3), dtype=np.uint8), (300, 400, 400, 600))
    # First try on the 200 x 100 crop, second on a 200 x 200 square.
    assert seen == [(200, 100), (200, 200)]
    # Centre of the 200 x 200 square, shifted by the region's corner.
    assert np.allclose(kp.point("nose"), [400, 500])


def test_hidden_back_leg_borrows_front_leg_for_balance():
    # 1 cm per pixel; stumps at 0, 20, 40.
    cal = Calibration(1.0, (0.0, 20.0, 40.0), 0.0, 71.0, 1)
    # Everything visible except the right (back) knee and both right-arm points, as front-on.
    names = ["left_eye", "right_eye", "left_shoulder", "right_shoulder", "left_elbow", "left_wrist",
             "left_hip", "right_hip", "left_knee", "left_ankle", "right_ankle"]
    # All at x = 100; feet at 90 and 110 so the base is 90..110.
    kp = _pose({n: (100.0, 50.0) for n in names} | {"left_ankle": (90.0, 90.0), "right_ankle": (110.0, 90.0)})
    # Without borrowing, 5% (arm) + 16.1% (thigh + shank) would be missing; with it, balance is measured.
    assert front_checks(kp, cal, front_on_sides("right"))["balance_side_to_side"] is not None
