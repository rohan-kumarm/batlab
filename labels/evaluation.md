# Test set (held back: the real accuracy)

Clips: 1. Batter found: yolo 21/21 frames, mediapipe 21/21 frames

## Point error vs your marks (lower is better)

| Point | Target | yolo | mediapipe |
|---|---|---|---|
| eye tilt (deg) | 2.0 | 6.8 (21) FAIL | 4.9 (21) FAIL |
| feet left edge (sideways) | 3.0 | 23.6 (21) FAIL | 2.4 (21) PASS |
| feet right edge (sideways) | 3.0 | 12.3 (21) FAIL | 8.3 (21) FAIL |
| front knee | 3.0 | 2.0 (21) PASS | 5.1 (21) FAIL |
| hands | 3.0 | 2.4 (21) PASS | 3.6 (21) FAIL |
| head (eye midpoint) | 2.0 | 1.0 (21) PASS | 3.8 (21) FAIL |
| head tilt (helmet detector, deg) | 2.0 | 2.0 (21) FAIL | 1.7 (21) PASS |
| head tilt vs helmet marks (deg) | 2.0 | 3.9 (21) FAIL | 3.1 (21) FAIL |

## Check values on release frames

| Check | Your marks | yolo | mediapipe | WillowCoach |
|---|---|---|---|---|
| eyes_level | +6.7 | -0.2 | +1.3 | 6.0 deg tilt left |
| head_tilt_helmet | +2.2 | +4.4 | +4.2 | 6.0 deg tilt left (eyes level) |
| head_on_middle_off | -7.9 | -7.5 | -4.7 | 0.0 cm |
| front_knee_inside_off_stump | +29.7 | +30.9 | +34.4 | 10.9 cm inside |
| hands_under_head | -1.3 | -3.1 | -6.5 | 4.1 cm |
| head_movement | +2.0 | +2.1 | +2.0 | 2.7 cm |

Numbers in brackets: how many frames the error is averaged over.
head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery (0.2 s before release to release), averaged over deliveries.

# Tune set

Clips: 1. Batter found: yolo 9/9 frames, mediapipe 2/9 frames

## Point error vs your marks (lower is better)

| Point | Target | yolo | mediapipe |
|---|---|---|---|
| eye tilt (deg) | 2.0 | 2.1 (9) FAIL | 6.2 (2) FAIL |
| feet left edge (sideways) | 3.0 | 19.0 (9) FAIL | 2.3 (2) PASS |
| feet right edge (sideways) | 3.0 | 12.5 (9) FAIL | 9.2 (2) FAIL |
| front knee | 3.0 | 6.0 (9) FAIL | 4.7 (2) FAIL |
| hands | 3.0 | 5.1 (9) FAIL | 11.6 (2) FAIL |
| head (eye midpoint) | 2.0 | 0.7 (9) PASS | 3.8 (2) FAIL |
| head tilt (helmet detector, deg) | 2.0 | 0.4 (9) PASS | 1.5 (2) PASS |
| head tilt vs helmet marks (deg) | 2.0 | 2.7 (9) FAIL | 2.5 (2) FAIL |

## Check values on release frames

| Check | Your marks | yolo | mediapipe | WillowCoach |
|---|---|---|---|---|
| eyes_level | +2.6 | +0.5 | +6.2 | 6.0 deg tilt left |
| head_tilt_helmet | +3.8 | +4.2 | +5.3 | 6.0 deg tilt left (eyes level) |
| head_on_middle_off | -1.0 | -0.8 | +0.0 | 0.0 cm |
| front_knee_inside_off_stump | +27.9 | +29.7 | +29.0 | 10.9 cm inside |
| hands_under_head | -4.4 | -6.7 | -13.1 | 4.1 cm |
| head_movement | +3.2 | +3.4 | - | 2.7 cm |

Numbers in brackets: how many frames the error is averaged over.
head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery (0.2 s before release to release), averaged over deliveries.
