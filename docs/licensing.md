# Pose model licensing

- **YOLO11-pose (Ultralytics):** AGPL-3.0. Fine for personal use and a public GitHub repo.
  If the app is ever hosted for paying users, AGPL requires publishing the full source of that
  service, unless an Ultralytics commercial license is bought.
- **MediaPipe Pose:** Apache 2.0. Free for commercial use with no source-sharing requirement.

Both adapters sit behind the same `PoseEstimator` interface (`src/batlab/pose/base.py`):
`--pose yolo` (default for now) or `--pose mediapipe`. Plan: switch the default to MediaPipe
once its results are checked, and before any paid academy use.
