"""Unit tests for the parts that do not need video: run with `pytest`."""

# numpy builds the fake keypoints and audio.
import numpy as np

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
    # Three stumps at x = 100, 125, 150 seen in several frames, plus one 3-px sliver.
    boxes = [(100, 0, 10, 80), (125, 0, 10, 80), (150, 0, 10, 80)] * 4 + [(118, 0, 3, 40)]
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
