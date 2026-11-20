# Front-on check definitions (Stage 1)

All measured on the release frame unless stated. "Off side" depends on batting hand.
cm values use the stump-height scale from `calibrate.py`.

| Check | Definition | Good when |
|---|---|---|
| eyes_level | Angle of the line between the eyes; positive = off-side eye lower | small |
| head_on_middle_off | 0 if the eye midpoint is between the off- and middle-stump lines; otherwise distance to the nearer line (+ outside off, − past middle) | 0 |
| balance_side_to_side | Distance of the centre of mass inside the outer ankle lines (negative = outside the base). Centre of mass from a 10-segment body model with standard mass shares | ≥ 0 |
| front_knee_inside_off_stump | Distance of the front knee on the leg side of the off-stump line | ≥ 0 |
| hands_under_head | Sideways distance of the wrist midpoint from a vertical line through the eye midpoint | small |
| head_stillness_front | Largest distance the eye midpoint moves from its position 0.2 s before release | small |

## How these were worked out
The check names and targets come from WillowCoach's review screens. Their screens draw the
reference lines for each check (stump lines, a head line, the outer foot lines and a centre-of-mass
marker), which showed what each one measures. None of their code was used; these are this
project's own implementations of the same measurements.

## Known differences from WillowCoach
- Their cm values on the same picture are about half of what the stump-height scale gives
  (for example, the hands-to-head gap they label 4.1 cm measures about 8 cm against the stumps).
  Either they use a different scale reference or one of the two scales is off. Measuring the
  real stump height would settle which.
- Ankles are used for the base of support; they may use toe and heel points.

## Known problems (Stage 1)
- **balance_side_to_side is not reliable yet.** From the front, the feet sit one behind the other,
  so the two ankles are only a few cm apart sideways and the centre of mass falls outside them.
  The base needs toe and heel points, which YOLO11 does not give and MediaPipe does. Fix: MediaPipe adapter.
- **front_knee_inside_off_stump is not reliable yet.** In the front-on view the legs overlap, and
  YOLO11 placed both knees on the back leg in all three test deliveries, missing the front pad
  beside the stumps. Fix options: MediaPipe (test whether it separates the legs), or take the
  front knee from the side-on camera once both phones are used.
