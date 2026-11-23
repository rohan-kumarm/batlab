"""Command line entry point: batlab analyze | label | evaluate <session.toml>"""

# argparse reads the command line options.
import argparse
# Path builds file paths.
from pathlib import Path

# The pipeline's building blocks.
from .calibrate import batter_roi, calibrate_front
from .checks_front import _head, front_checks, grade, head_stillness
# Head tilt from the helmet.
from .head_tilt import helmet_tilt
from .geometry import front_on_sides
# Names of the 17 standard points, for saving keypoints in the report.
from .pose import NAMES
from .release import find_releases
from .report import draw_release, summarise, write_outputs
from .session import load_session, load_toml
from .video import probe, read_audio, read_frames

# Folder that holds config/ (two levels above this file: src/batlab -> project root).
ROOT = Path(__file__).resolve().parents[2]


def prepare(session_path: str):
    """Shared first steps: load the session, read the video, find releases, calibrate."""
    # Load the session settings.
    session = load_session(session_path)
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
    # Warn when the count differs from the balls the batter says they faced.
    if session.balls_faced is not None and session.balls_faced != len(releases):
        # Clear warning with both numbers.
        print(f"WARNING: session says {session.balls_faced} balls were faced, but {len(releases)} deliveries were found.")
    # Measure the stumps for scale and stump lines.
    cal = calibrate_front(info, session.stump_height_cm)
    # Report the calibration.
    print(f"Calibration: {cal.cm_per_px:.4f} cm/px from {cal.frames_used} frames, stumps at x={[round(x) for x in cal.stump_x]}")
    # Work out off side and front leg from the batting hand.
    sides = front_on_sides(session.batting_hand)
    # Return everything the commands need.
    return session, info, releases, cal, sides


def load_pose(name: str, model_path: str | None):
    """Load a pose adapter by name. Imports happen here so only that model's library is needed."""
    # MediaPipe adapter (Apache 2.0).
    if name == "mediapipe":
        # Import only when chosen.
        from .pose.mediapipe_pose import MediaPipePose
        # A missing model path means "download to the cache on first use".
        return MediaPipePose(model_path)
    # YOLO11 adapter (AGPL-3.0).
    from .pose.yolo import YoloPose
    # Ultralytics downloads "yolo11m-pose.pt" itself if no path is given.
    return YoloPose(model_path or "yolo11m-pose.pt")


def analyze(args: argparse.Namespace) -> None:
    """Run the Stage 1 front-on pipeline for one session."""
    # Shared setup.
    session, info, releases, cal, sides = prepare(args.session)
    # Load the target ranges.
    targets = load_toml(args.targets)
    # Load the WillowCoach reference values if the file exists.
    reference = load_toml(args.reference) if Path(args.reference).exists() else None
    # Load the chosen pose model.
    pose = load_pose(args.pose, args.pose_model)
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
        # Head point at release, used to place the helmet crop.
        head = _head(kp_release) if kp_release is not None else None
        # Helmet angle in the picture (left to right), or None.
        angle = helmet_tilt(frames[-1], head, cal.cm_per_px) if head is not None else None
        # Seventh check, in the eyes_level sign convention: positive when the head tips toward the off side.
        checks["head_tilt"] = None if angle is None else angle * sides.off_dir
        # Count how many checks were measured.
        measured = sum(v is not None for v in checks.values())
        # Eligible if a pose was found and at least 4 checks were measured.
        eligible = kp_release is not None and measured >= 4
        # Reason shown when a delivery is not eligible.
        reason = None if eligible else ("no batter found" if kp_release is None else f"only {measured}/{len(checks)} checks measured")
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


def label(args: argparse.Namespace) -> None:
    """Build the marking page for one session."""
    # Imported here: only this command needs it.
    from .label import build_label_page
    # Shared setup.
    _, info, releases, cal, _ = prepare(args.session)
    # Release frame numbers.
    release_frames = [int(round(r.time_s * info.fps)) for r in releases]
    # Write the page.
    page = build_label_page(info, cal, release_frames, Path(args.out), args.session)
    # Tell the user where it is.
    print(f"Marking page written to {page.resolve()}")
    # And what to do next.
    print("Open it in a browser, mark every frame, then click 'Download labels.json'.")


def evaluate_cmd(args: argparse.Namespace) -> None:
    """Score each pose model against the marks in one or more labels.json files."""
    # json reads the labels files.
    import json
    # Imported here: only this command needs them.
    from .evaluate import combine, evaluate, format_report
    # Load each requested model once, with its optional model file.
    paths = {"yolo": args.yolo_model, "mediapipe": args.mediapipe_model}
    # Name -> loaded adapter.
    models = {m: load_pose(m, paths[m]) for m in args.pose}
    # Raw results grouped by split ("tune" / "test").
    raws: dict[str, list] = {"tune": [], "test": []}
    # Go through every labels file.
    for lf in args.labels:
        # Read the marks.
        labels = json.loads(Path(lf).read_text())
        # Session from the file itself, or from --session for files made before sessions were saved.
        session_path = labels.get("session") or args.session
        # Without a session the clip and its settings are unknown.
        if not session_path:
            # Clear message.
            raise SystemExit(f"{lf} does not say which session it belongs to; pass --session.")
        # Show progress.
        print(f"\n== {lf}")
        # Shared setup for that clip (reads the video, calibrates).
        session, info, _, cal, sides = prepare(session_path)
        # Score this clip and file the raw results under its split.
        raws[session.split].append(evaluate(info, cal, sides, labels, models))
    # WillowCoach values for the last column, if the file exists.
    reference = load_toml(args.reference) if Path(args.reference).exists() else None
    # Accuracy targets for pass/fail.
    targets = load_toml(args.accuracy_targets) if Path(args.accuracy_targets).exists() else None
    # Report sections: the held-back test set is the one that counts.
    sections = []
    # Test set first, if any clips are in it.
    if raws["test"]:
        # Final accuracy: never used for tuning.
        sections.append(format_report(combine(raws["test"]), reference, targets, "Test set (held back: the real accuracy)"))
    # Tune set, if any.
    if raws["tune"]:
        # Used for improving the engine; expect it to look better than the test set.
        sections.append(format_report(combine(raws["tune"]), reference, targets, "Tune set"))
    # Join the sections.
    text = "\n".join(sections)
    # Output file.
    out = Path(args.out)
    # Make sure its folder exists.
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write the report.
    out.write_text(text)
    # Show it and say where it was saved.
    print("\n" + text + f"\nSaved to {out.resolve()}")


def new_session(args: argparse.Namespace) -> None:
    """Write a session file for a video, so many clips can be set up quickly."""
    # Imported here: only this command needs it.
    from .video import creation_date
    # Full path to the video, so the session works from any folder.
    video = Path(args.video).expanduser().resolve()
    # Stop early if the path is wrong.
    if not video.exists():
        # Clear message.
        raise SystemExit(f"Video not found: {video}")
    # Recording date from the video's metadata, or "undated".
    date = creation_date(video) or "undated"
    # File name: date plus the video's own name, e.g. 2026-11-10_IMG_0478.toml.
    out = Path(args.dir) / f"{date}_{video.stem}.toml"
    # Never overwrite an existing session (it may hold hand-edited settings).
    if out.exists():
        # Clear message.
        raise SystemExit(f"{out} already exists; edit it or delete it first.")
    # Machine speed line only for machine sessions.
    speed = f"machine_speed_mph = {args.speed}\n" if args.speed is not None else ""
    # The file's contents, in the same layout as the example session.
    text = (
        f'[player]\nname = "{args.player}"\nbatting_hand = "{args.hand}"\n\n'
        f'[session]\ndate = "{date}"\ndelivery_source = "{args.source}"\n{speed}split = "{args.split}"\n\n'
        f'[calibration]\nstump_height_cm = {args.stump_height}\n\n'
        f'[videos]\nfront = "{video}"\nside = ""\n'
    )
    # Make sure the folder exists.
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write the file.
    out.write_text(text)
    # Tell the user.
    print(f"Wrote {out}")


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
    # The label command: build the marking page.
    lb = sub.add_parser("label", help="Build a page for marking body points on release frames", allow_abbrev=False)
    # Session file.
    lb.add_argument("session", help="Path to a sessions/*.toml file")
    # Output folder for the page.
    lb.add_argument("--out", default="labels/latest", help="Folder for the marking page")
    # The evaluate command: score models against the marks.
    ev = sub.add_parser("evaluate", help="Compare pose models with your marks", allow_abbrev=False)
    # One or more labels files (e.g. labels/*/labels.json).
    ev.add_argument("--labels", nargs="+", required=True, help="labels.json files from the marking page")
    # Session for older labels files that do not record one.
    ev.add_argument("--session", default=None, help="Session file, only for labels.json files without one")
    # Which models to score (one or both).
    ev.add_argument("--pose", nargs="+", choices=["yolo", "mediapipe"], default=["yolo", "mediapipe"])
    # Optional model files for each.
    ev.add_argument("--yolo-model", default=None)
    ev.add_argument("--mediapipe-model", default=None)
    # Accuracy targets for pass/fail.
    ev.add_argument("--accuracy-targets", default=str(ROOT / "config" / "accuracy_targets.toml"))
    # Where to save the report.
    ev.add_argument("--out", default="labels/evaluation.md")
    # WillowCoach values for comparison.
    ev.add_argument("--reference", default=str(ROOT / "config" / "willowcoach_reference.toml"))
    # The new-session command: write a session file for one video.
    ns = sub.add_parser("new-session", help="Create a session file for a video", allow_abbrev=False)
    # The video.
    ns.add_argument("video", help="Path to the front-on video")
    # Player details.
    ns.add_argument("--player", default="Rohan")
    ns.add_argument("--hand", choices=["right", "left"], default="right")
    # Machine or bowler, and machine speed.
    ns.add_argument("--source", choices=["machine", "bowler"], default="machine")
    ns.add_argument("--speed", type=float, default=None, help="Machine speed in mph")
    # Tune or held-back test set.
    ns.add_argument("--split", choices=["tune", "test"], default="tune")
    # Stump height in cm.
    ns.add_argument("--stump-height", type=float, default=71.1)
    # Folder for session files.
    ns.add_argument("--dir", default="sessions")
    # Read the command line.
    args = parser.parse_args()
    # Run the chosen command.
    if args.command == "analyze":
        # Start the analysis.
        analyze(args)
    # Build the marking page.
    elif args.command == "label":
        # Run it.
        label(args)
    # Create a session file.
    elif args.command == "new-session":
        # Run it.
        new_session(args)
    # Score the models.
    elif args.command == "evaluate":
        # Run it.
        evaluate_cmd(args)


# Allows "python -m batlab.cli analyze ..." as well as the "batlab" command.
if __name__ == "__main__":
    # Run main.
    main()
