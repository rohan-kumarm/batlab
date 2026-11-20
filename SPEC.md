# batlab: what is tracked

Status key: **S1** = built in Stage 1, **later** = planned.

## A. Session details (entered once per session, `sessions/*.toml`)
- Player name, batting hand (right/left), standing height (optional, later)
- Delivery source (machine or bowler), pace or spin, machine speed
- Date, camera views, frame rate of each clip (read from the file)
- Stump height (default 71.1 cm)

## B. Calibration (per camera, per session)
- cm per pixel from stump height (S1, front-on)
- x position of each stump; off-stump and middle/off lines (S1, front-on)
- Crease line (later)
- Time offset between the two phones (later, from the machine sound or a clap)

## C. Events per delivery
- Release: machine sound (S1); bowler's arm (later)
- Trigger movement start, top of backlift, front foot landing, impact, end of follow-through (later)
- Eligible or not, with a reason (S1)

## D. Raw data per frame, per camera
- 17 body keypoints with confidence (S1)
- Bat handle and toe (later, needs a trained bat model)
- Ball position (later)

## E. Setup checks at release
Front-on (S1): eyes level, head on middle/off, side-to-side balance, front knee inside off stump,
hands under head, head stillness (front-on). Definitions are in `docs/check_definitions.md`.

Side-on (later): hands centred, stance width, stance height, forward-back balance, head stillness (side-on),
shoulder lead, head near front foot, head inside back toe, feet slightly open, front knee inside
front toe, back knee inside back toe, front knee inside back-knee line, front toe inside back-knee
line, shoulder-elbow alignment, backlift angle.

Timing (later): trigger movement (ms before release).

## F. Impact metrics (later)
Head position at impact, back heel at impact, hip-shoulder separation (3D from both phones),
backlift lag.

## G. Shot information (later)
Shot type and sub-type, bounce position (line and length).

## H. Session summary
Per check: average, spread, eligible count (S1); grade against `config/targets.toml` (S1);
faults ranked by distance from target and frequency (later).

## I. Progress across sessions (later)

## Filming rules
- Two phones on tripods, 60 fps, original files (not exports or "Rendered" copies).
- Front-on: directly behind the middle stump at the far end, so stumps and batter line up.
  An off-centre camera shifts the batter against the stumps and biases the sideways checks.
- Side-on: square of the batting crease, camera facing the batter's chest.
- Stumps fully visible in both views.
