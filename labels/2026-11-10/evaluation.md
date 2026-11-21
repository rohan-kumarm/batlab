## Point error vs your marks (cm, lower is better)

| Point | yolo | mediapipe |
|---|---|---|
| feet left edge (sideways) | 19.7 (9) | 2.8 (5) |
| feet right edge (sideways) | 12.9 (9) | 10.6 (5) |
| front knee | 6.1 (9) | 3.0 (5) |
| hands | 5.3 (9) | 10.7 (5) |
| head (eye midpoint) | 0.7 (9) | 5.0 (5) |

## Check values on release frames

| Check | Your marks | yolo | mediapipe | WillowCoach |
|---|---|---|---|---|
| eyes_level | +2.6 | +0.5 | +5.6 | 6.0 deg tilt left |
| head_on_middle_off | -0.6 | -0.3 | +0.0 | 0.0 cm |
| front_knee_inside_off_stump | +28.8 | +30.6 | +28.9 | 10.9 cm inside |
| hands_under_head | -4.5 | -6.9 | -14.2 | 4.1 cm |
| head_movement | +3.3 | +3.6 | +2.0 | 2.7 cm |

Numbers in brackets: how many frames the error is averaged over.
head_movement: largest eye-midpoint movement across the 3 marked frames of each delivery (0.2 s before release to release), averaged over deliveries.
