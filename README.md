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

### MediaPipe pose model (Apache 2.0, gives heel and toe points)
```bash
pip install -e ".[mediapipe]"
batlab analyze sessions/2026-11-10_front.toml --pose mediapipe --out runs/2026-11-10_front_mp
```
The first MediaPipe run downloads its model (about 30 MB) to `~/.cache/batlab/`.
Use a different `--out` folder per model so the two sets of results can be compared.

Outputs in the `--out` folder:
- `delivery_N_release.jpg`: release frame with skeleton, off-stump line (red) and middle-stump line (yellow)
- `deliveries.csv`: every check for every delivery
- `summary.md`: per-check average, spread, grade and the WillowCoach value
- `report.json`: everything above plus calibration and timings

## Checking accuracy against your own marks
1. Create a session file for each clip (date comes from the video). Add `--split test` for about
   1 in 5 clips: those are held back and only used to measure final accuracy.
```bash
batlab new-session ~/Desktop/clips/IMG_0478.MOV --speed 62
```
2. Build a marking page per session (9 frames each: every release, plus 0.1 s and 0.2 s before):
```bash
batlab label sessions/2026-11-10_IMG_0478.toml --out labels/2026-11-10_IMG_0478
```
3. Open each `index.html`, click the 6 points on every frame (zoomed view helps; S = can't see it),
   click "Download labels.json" and move it into that clip's labels folder.
4. Score both models on every marked clip at once:
```bash
batlab evaluate --labels labels/*/labels.json
```
`labels/evaluation.md` shows, separately for the test and tune sets, each model's average point
error with PASS/FAIL against `config/accuracy_targets.toml`, how often each model found the batter,
and every check worked out from your marks next to each model and WillowCoach.

## Settings
- `config/targets.toml`: grading ranges (edit to match your coaching)
- `config/willowcoach_reference.toml`: values from the WillowCoach review, for comparison
- `config/accuracy_targets.toml`: how close the engine must be to your marks (agreed 20 Nov 2026)

## Tests
```bash
pytest
```
