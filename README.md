# batlab

Batting technique analysis from net-session video. Stage 1 analyses a front-on clip from a
bowling-machine session: it finds each delivery's release from the machine's firing sound,
calibrates cm per pixel from the stumps, runs pose estimation and measures six setup checks.

- What is tracked, now and later: [SPEC.md](SPEC.md)
- How each check is measured, and known problems: [docs/check_definitions.md](docs/check_definitions.md)
- Pose model licensing: [docs/licensing.md](docs/licensing.md)

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
ffmpeg must be installed (`brew install ffmpeg` on a Mac).

## Run
1. Copy `sessions/2026-11-10_front.toml`, set the player, batting hand and video path.
2. Run:
```bash
batlab analyze sessions/2026-11-10_front.toml --out runs/2026-11-10_front
```
The first run downloads the YOLO11 pose model (about 40 MB).

Outputs in the `--out` folder:
- `delivery_N_release.jpg`: release frame with skeleton, off-stump line (red) and middle-stump line (yellow)
- `deliveries.csv`: every check for every delivery
- `summary.md`: per-check average, spread, grade and the WillowCoach value
- `report.json`: everything above plus calibration and timings

## Settings
- `config/targets.toml`: grading ranges (edit to match your coaching)
- `config/willowcoach_reference.toml`: values from the WillowCoach review, for comparison

## Tests
```bash
pytest
```
