"""Command line entry point: batlab analyze <session.toml>"""

# argparse reads the command line options.
import argparse
# Path builds file paths.
from pathlib import Path

# The pipeline's building blocks.
from .calibrate import batter_roi, calibrate_front
from .checks_front import front_checks, grade, head_stillness
from .geometry import front_on_sides
# Names of the 17 standard points, for saving keypoints in the report.
from .pose import NAMES
from .release import find_releases
from .report import draw_release, summarise, write_outputs
from .session import load_session, load_toml
from .video import probe, read_audio, read_frames

# Folder that holds config/ (two levels above this file: src/batlab -> project root).
ROOT = Path(__file__).resolve().parents[2]


def analyze(args: argparse.Namespace) -> None:
    """Run the Stage 1 front-on pipeline for one session."""
    # Load the session settings.
    session = load_session(args.session)
    # Load the target ranges.
    targets = load_toml(args.targets)
    # Load the WillowCoach reference values if the file exists.
    reference = load_toml(args.reference) if Path(args.reference).exists() else None
    # Stage 1 needs a front-on clip.
    if session.front_video is None:
        # Stop with a clear message.
        raise SystemExit("Stage 1 needs a front-on video in the session file.")
    # Stage 1 finds release from the machine sound only.
    if session.delivery_source != "machine":
        # Stop with a clear message.
        raise SystemExit("Release detection for bowler sessions is not built yet (Stage 2).")
    # Read the video's size, frame rate and HDR status.
    info = probe(session.front_video)
    # Report what was read.
    print(f"Video: {info.width}x{info.height} @ {info.fps:.2f} fps, {info.duration:.1f} s, HDR={info.hdr}")
    # Find releases from the machine's firing sound.
    releases = find_releases(read_audio(info.path), 48000)
    # Report how many deliveries were found.
    print(f"Deliveries found from machine sound: {len(releases)}")
    # Measure the stumps for scale and stump lines.
    cal = calibrate_front(info, session.stump_height_cm)
    # Report the calibration.
    print(f"Calibration: {cal.cm_per_px:.4f} cm/px from {cal.frames_used} frames, stumps at x={[round(x) for x in cal.stump_x]}")
    # Work out off side and front leg from the batting hand.
    sides = front_on_sides(session.batting_hand)
    # Pick the pose model. Imports happen here so only the chosen model's library must be installed.
    if args.pose == "mediapipe":
        # MediaPipe adapter (Apache 2.0).
        from .pose.mediapipe_pose import MediaPipePose
        # Load it; a missing model path means "download to the cache on first use".
        pose = MediaPipePose(args.pose_model)
    else:
        # YOLO11 adapter (AGPL-3.0).
        from .pose.yolo import YoloPose
        # Load it; ultralytics downloads "yolo11m-pose.pt" itself if no path is given.
        pose = YoloPose(args.pose_model or "yolo11m-pose.pt")
    # Report which model is in use.
    print(f"Pose model: {args.pose}")
    # Region around the batter, from the stumps (MediaPipe needs it; YOLO ignores it).
    roi = batter_roi(cal)
    # Number of frames in the head stillness window.
    window = max(2, int(round(args.stillness_window_s * info.fps)))
    # Output folder for this run.
    out_dir = Path(args.out)
    # Create it.
    out_dir.mkdir(parents=True, exist_ok=True)
    # Per-delivery results.
    deliveries = []
    # Analyse each delivery.
    for r in releases:
        # Frame number of release.
        release_frame = int(round(r.time_s * info.fps))
        # Read the stillness window ending at the release frame (window frames before, plus release).
        frames = read_frames(info, (release_frame - window) / info.fps, window + 1)
        # Run pose on every frame in the window, telling the model where the batter stands.
        kps = [pose.estimate(f, roi) for f in frames]
        # The release frame is the last one read.
        kp_release = kps[-1] if kps else None
        # Five single-frame checks, or none if no pose was found.
        checks = front_checks(kp_release, cal, sides) if kp_release is not None else {}
        # Sixth check, measured across the window.
        checks["head_stillness_front"] = head_stillness(kps, cal)
        # Count how many checks were measured.
        measured = sum(v is not None for v in checks.values())
        # Eligible if a pose was found and at least 4 of the 6 checks were measured.
        eligible = kp_release is not None and measured >= 4
        # Reason shown when a delivery is not eligible.
        reason = None if eligible else ("no batter found" if kp_release is None else f"only {measured}/6 checks measured")
        # Text lines drawn on the image.
        lines = [f"Delivery {r.index}  release {r.time_s:.2f}s  frame {release_frame}"]
        # One line per measured check.
        lines += [f"{k}: {v:+.1f}" for k, v in checks.items() if v is not None]
        # Save the annotated release frame.
        draw_release(frames[-1], kp_release, cal, sides, lines, out_dir / f"delivery_{r.index}_release.jpg")
        # Store this delivery's result.
        deliveries.append({
            # Delivery number.
            "index": r.index,
            # Release time in seconds.
            "release_s": r.time_s,
            # Release frame number.
            "release_frame": release_frame,
            # Bat sound time if heard.
            "impact_audio_s": r.impact_audio_s,
            # Release-to-bat-sound time, a check on the machine speed.
            "release_to_impact_s": None if r.impact_audio_s is None else r.impact_audio_s - r.time_s,
            # Eligibility and reason.
            "eligible": eligible, "reason": reason,
            # Check values.
            "checks": checks,
            # Every point at release as [x, y, confidence], so missing points can be diagnosed later.
            "keypoints_at_release": None if kp_release is None else {
                # The 17 standard points.
                **{n: [round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                   for n, (x, y), c in zip(NAMES, kp_release.xy, kp_release.conf)},
                # Extra points (heels and toes with MediaPipe).
                **{n: [round(float(p[0]), 1), round(float(p[1]), 1), round(float(c), 3)]
                   for n, (p, c) in kp_release.extra.items()},
            },
            # Grade of each check.
            "grades": {k: grade(k, v, targets) for k, v in checks.items()},
        })
    # Collect everything into one result.
    result = {
        # Session details, including which pose model produced these numbers.
        "session": {"player": session.player, "batting_hand": session.batting_hand,
                    "video": str(info.path), "fps": info.fps, "pose_model": args.pose},
        # Calibration values.
        "calibration": {"cm_per_px": cal.cm_per_px, "stump_x": cal.stump_x,
                        "stump_top_y": cal.stump_top_y, "stump_base_y": cal.stump_base_y},
        # Per-delivery results.
        "deliveries": deliveries,
        # Per-check summary.
        "summary": summarise(deliveries),
    }
    # Write the files and print the summary table.
    print(write_outputs(out_dir, result, reference, targets, grade))
    # Tell the user where the files are.
    print(f"Outputs written to {out_dir.resolve()}")


def main() -> None:
    """Parse the command line and run the chosen command."""
    # Top-level parser. allow_abbrev=False stops shortened option names (e.g. --pose being read as --pose-model).
    parser = argparse.ArgumentParser(prog="batlab", description="Batting technique analysis", allow_abbrev=False)
    # Sub-commands (only "analyze" for now).
    sub = parser.add_subparsers(dest="command", required=True)
    # The analyze command.
    a = sub.add_parser("analyze", help="Analyse one session", allow_abbrev=False)
    # Path to the session file (required).
    a.add_argument("session", help="Path to a sessions/*.toml file")
    # Output folder.
    a.add_argument("--out", default="runs/latest", help="Folder for outputs")
    # Target ranges file.
    a.add_argument("--targets", default=str(ROOT / "config" / "targets.toml"))
    # WillowCoach reference values file.
    a.add_argument("--reference", default=str(ROOT / "config" / "willowcoach_reference.toml"))
    # Which pose model to use.
    a.add_argument("--pose", choices=["yolo", "mediapipe"], default="yolo")
    # Optional path to the pose model file; without it, each model downloads its default on first use.
    a.add_argument("--pose-model", default=None)
    # Length of the head stillness window before release, in seconds.
    a.add_argument("--stillness-window-s", type=float, default=0.2)
    # Read the command line.
    args = parser.parse_args()
    # Run the chosen command.
    if args.command == "analyze":
        # Start the analysis.
        analyze(args)


# Allows "python -m batlab.cli analyze ..." as well as the "batlab" command.
if __name__ == "__main__":
    # Run main.
    main()
