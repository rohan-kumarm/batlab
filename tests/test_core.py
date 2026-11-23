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


def test_marks_give_true_check_values():
    # Converter under test.
    from batlab.evaluate import marks_to_keypoints
    # 1 cm per pixel; off stump at x 100, middle at 120 (right-hander).
    cal = Calibration(1.0, (100.0, 120.0, 140.0), 0.0, 71.0, 1)
    # Marks as the page saves them: level eyes over the off-to-middle zone, hands 3 px right of
    # the head, front knee 30 px inside off stump.
    marks = {"eye_left_in_picture": [105, 10], "eye_right_in_picture": [115, 10], "hands": [113, 40],
             "front_knee": [130, 60], "feet_left_edge": [95, 90], "feet_right_edge": [150, 90]}
    # Checks computed from the marks.
    out = front_checks(marks_to_keypoints(marks, front_on_sides("right")), cal, front_on_sides("right"))
    # Level eyes.
    assert out["eyes_level"] == 0.0
    # Head inside the zone.
    assert out["head_on_middle_off"] == 0.0
    # Knee 30 cm on the leg side of off stump.
    assert out["front_knee_inside_off_stump"] == 30.0
    # Hands 3 px to the right of the head = 3 cm toward leg for a right-hander, so -3.
    assert out["hands_under_head"] == -3.0


def test_point_errors_ignore_eye_label_swaps():
    # Error function under test.
    from batlab.evaluate import point_errors
    # Model output with the eyes labelled the other way round from the picture order.
    kp = _pose({"left_eye": (115.0, 10.0), "right_eye": (105.0, 10.0), "left_wrist": (113.0, 44.0)})
    # Marks.
    marks = {"eye_left_in_picture": [105, 10], "eye_right_in_picture": [115, 10], "hands": [113, 40]}
    # 1 cm per pixel.
    err = point_errors(kp, marks, front_on_sides("right"), 1.0)
    # The eye midpoint is the same either way, so the swap costs nothing.
    assert err["head (eye midpoint)"] == 0.0
    # One visible wrist 4 px below the marked hands: 4 cm.
    assert err["hands"] == 4.0


def test_combine_averages_across_clips():
    # Combiner under test.
    from batlab.evaluate import combine
    # Two clips' raw results: hands errors 2 and 4 cm in clip A, 6 cm in clip B.
    a = {"errors": {"yolo": {"hands": [2.0, 4.0]}}, "values": {"marks": {"eyes_level": [1.0]}, "yolo": {"eyes_level": [2.0]}},
         "frames": 9, "found": {"yolo": 9}}
    # Second clip, where the batter was found in 7 of 9 frames.
    b = {"errors": {"yolo": {"hands": [6.0]}}, "values": {"marks": {"eyes_level": [3.0]}, "yolo": {"eyes_level": [4.0]}},
         "frames": 9, "found": {"yolo": 7}}
    # Merge.
    r = combine([a, b])
    # Mean of 2, 4 and 6 is 4, over 3 frames.
    assert r["point_error_cm"]["yolo"]["hands"] == (4.0, 3)
    # Mean of the marked eyes_level values 1 and 3 is 2.
    assert r["check_values"]["marks"]["eyes_level"] == (2.0, 2)
    # Detection: 16 of 18 frames.
    assert r["found"]["yolo"] == (16, 18)


def test_eye_tilt_error_in_degrees():
    # Error function under test.
    from batlab.evaluate import point_errors
    # Model eyes level; marked eyes tilted 45 degrees (10 px across, 10 px down).
    kp = _pose({"left_eye": (100.0, 10.0), "right_eye": (110.0, 10.0)})
    # Marks.
    marks = {"eye_left_in_picture": [100, 10], "eye_right_in_picture": [110, 20]}
    # Tilt difference is 45 degrees.
    assert point_errors(kp, marks, front_on_sides("right"), 1.0)["eye tilt (deg)"] == pytest.approx(45.0)


def test_stump_colour_found_even_next_to_bigger_yellow_shirt():
    # OpenCV draws the test image.
    import cv2
    # Function under test.
    from batlab.calibrate import find_stump_hue
    # Dark 1080 x 1920 frame.
    img = np.full((1920, 1080, 3), 40, dtype=np.uint8)
    # A big, bright yellow "shirt" (hue about 27), larger than all the stumps together.
    shirt = cv2.cvtColor(np.uint8([[[27, 180, 240]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    # Draw it.
    cv2.rectangle(img, (600, 700), (800, 900), shirt, -1)
    # Lime stump colour as measured on the 19 Nov clips (hue 35, saturation 100, brightness 230).
    lime = cv2.cvtColor(np.uint8([[[35, 100, 230]]]), cv2.COLOR_HSV2BGR)[0, 0].tolist()
    # Three 12 px wide, 150 px tall stumps, 22 px apart.
    for x in (550, 572, 594):
        # Draw one stump.
        cv2.rectangle(img, (x, 900), (x + 11, 1050), lime, -1)
    # The stump colour should be found near 35, not the shirt's 27.
    assert abs(find_stump_hue([img]) - 35) <= 3


def test_quiet_machine_sound_counts_when_bat_follows():
    # Fixed seed for repeatable noise.
    rng = np.random.default_rng(1)
    # Sample rate.
    sr = 48000
    # Eight seconds of faint background noise.
    audio = rng.normal(0, 1e-4, 8 * sr).astype(np.float32)

    def click(t: float, size: float) -> None:
        """Add a 2 ms burst of noise at time t with the given loudness."""
        # Start sample.
        s = int(t * sr)
        # Add the burst.
        audio[s:s + 96] += rng.normal(0, size, 96)

    # Delivery: a quiet machine sound (0.12) at 1.0 s and a much louder bat hit (0.3) 0.7 s later,
    # as in IMG_0551 delivery 4, where the machine reached just under half the bat's loudness.
    click(1.0, 0.12)
    # The bat hit.
    click(1.7, 0.3)
    # A stray knock of similar quietness at 5.0 s with nothing after it: not a delivery.
    click(5.0, 0.12)
    # Run detection.
    rel = find_releases(audio, sr)
    # One delivery, released at the quiet machine sound, not at the bat hit.
    assert len(rel) == 1 and abs(rel[0].time_s - 1.0) < 0.005


def test_model_head_tilt_compared_with_marked_grille():
    # Error function under test.
    from batlab.evaluate import point_errors
    # Model eyes level (0 degrees).
    kp = _pose({"left_eye": (100.0, 10.0), "right_eye": (130.0, 10.0)})
    # Marked grille bar 100 px wide and 10 px higher on the right: atan(10/100) = 5.7 degrees.
    marks = {"grille_left_end": [80, 30], "grille_right_end": [180, 20]}
    # The model's level eye line is 5.7 degrees off the helmet's true tilt.
    assert point_errors(kp, marks, front_on_sides("right"), 1.0)["head tilt vs helmet marks (deg)"] == pytest.approx(5.71, abs=0.01)


def test_helmet_tilt_finds_known_angle():
    # OpenCV draws the test image.
    import cv2
    # Function under test.
    from batlab.head_tilt import helmet_tilt
    # Grey 600 x 600 frame.
    img = np.full((600, 600, 3), 90, dtype=np.uint8)
    # Three bright bars 100 px wide (like a grille at 0.21 cm/px), 12 px apart, centred at x 300.
    for y in (310, 322, 334):
        # Draw one bar.
        cv2.rectangle(img, (250, y), (350, y + 3), (230, 230, 230), -1)
    # Tilt the whole picture by 4 degrees about the head point (300, 300).
    m = cv2.getRotationMatrix2D((300, 300), 4.0, 1.0)
    # Rotate; anticlockwise on screen raises the bars' right ends.
    img = cv2.warpAffine(img, m, (600, 600), borderValue=(90, 90, 90))
    # Raised right end = negative angle in the picture convention (y grows downward).
    assert helmet_tilt(img, (300, 300), 0.21) == pytest.approx(-4.0, abs=0.3)


def test_left_ball_with_faint_follow_up_still_counts():
    # Fixed seed for repeatable noise.
    rng = np.random.default_rng(2)
    # Sample rate.
    sr = 48000
    # Six seconds of faint background noise.
    audio = rng.normal(0, 1e-4, 6 * sr).astype(np.float32)

    def click(t: float, size: float) -> None:
        """Add a 2 ms burst of noise at time t with the given loudness."""
        # Start sample.
        s = int(t * sr)
        # Add the burst.
        audio[s:s + 96] += rng.normal(0, size, 96)

    # Loudest sound in the clip (sets the scale): a bat hit after a strong release elsewhere.
    click(0.5, 0.3)
    # A quiet machine sound (0.4 of the loudest) at 3.0 s...
    click(3.0, 0.12)
    # ...followed 0.75 s later by only a faint sound (the ball was left), as on IMG_0552.
    click(3.75, 0.004)
    # Run detection.
    rel = find_releases(audio, sr)
    # The quiet delivery at 3.0 s is found.
    assert any(abs(r.time_s - 3.0) < 0.005 for r in rel)
