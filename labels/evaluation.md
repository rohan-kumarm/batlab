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
| head tilt (helmet detector, deg) | 2.0 | 1.2 (21) PASS | 1.1 (21) PASS |
| head tilt vs helmet marks (deg) | 2.0 | 3.9 (21) FAIL | 3.1 (21) FAIL |

## Check values on release frames

| Check | Your marks | yolo | mediapipe | WillowCoach |
|---|---|---|---|---|
| eyes_level | +6.7 | -0.2 | +1.3 | 6.0 deg tilt left |
| head_tilt_helmet | +2.2 | +2.7 | +3.3 | 6.0 deg tilt left (eyes level) |
| head_on_middle_off | -7.9 | -7.5 | -4.7 | 0.0 cm |
| front_knee_inside_off_stump | +29.7 | +30.9 | +34.4 | 10.9 cm inside |
| hands_under_head | -1.3 | -3.1 | -6.5 | 4.1 cm |
| head_movement | +2.0 | +2.1 | +2.0 | 2.7 cm |

Numbers in brackets: how many frames the error is averaged over.
head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery (0.2 s before release to release), averaged over deliveries.

# Tune set

Clips: 2. Batter found: yolo 99/99 frames, mediapipe 92/99 frames

## Point error vs your marks (lower is better)

| Point | Target | yolo | mediapipe |
|---|---|---|---|
| eye tilt (deg) | 2.0 | 5.6 (39) FAIL | 4.0 (32) FAIL |
| feet left edge (sideways) | 3.0 | 23.6 (39) FAIL | 2.8 (32) PASS |
| feet right edge (sideways) | 3.0 | 12.3 (39) FAIL | 8.8 (32) FAIL |
| front knee | 3.0 | 2.9 (39) PASS | 5.1 (32) FAIL |
| hands | 3.0 | 3.3 (39) FAIL | 4.6 (32) FAIL |
| head (eye midpoint) | 2.0 | 1.0 (39) PASS | 3.9 (32) FAIL |
| head tilt (helmet detector, deg) | 2.0 | 0.9 (39) PASS | 0.9 (32) PASS |
| head tilt vs helmet marks (deg) | 2.0 | 3.8 (39) FAIL | 2.9 (32) FAIL |

## Check values on release frames

| Check | Your marks | yolo | mediapipe | WillowCoach |
|---|---|---|---|---|
| eyes_level | +4.3 | -2.1 | +1.8 | 6.0 deg tilt left |
| head_tilt_helmet | +2.1 | +3.5 | +3.3 | 6.0 deg tilt left (eyes level) |
| head_on_middle_off | -7.4 | -6.2 | -4.3 | 0.0 cm |
| front_knee_inside_off_stump | +31.3 | +31.8 | +35.2 | 10.9 cm inside |
| hands_under_head | -2.0 | -4.9 | -8.0 | 4.1 cm |
| head_movement | +2.1 | +2.0 | +2.0 | 2.7 cm |

Numbers in brackets: how many frames the error is averaged over.
head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery (0.2 s before release to release), averaged over deliveries.
