"""Turns 'right-handed' or 'left-handed' into image directions and body sides."""

# dataclass for the Sides record.
from dataclasses import dataclass


# Which way is which for one batter in the front-on view.
@dataclass(frozen=True)
class Sides:
    # +1 if the off side is toward larger image x (right of the image), -1 if toward the left.
    off_dir: int
    # Prefix of the front leg's keypoint names: "left" or "right" (the player's own side).
    front: str
    # Prefix of the back leg's keypoint names.
    back: str
    # Index (0, 1 or 2, left to right in the image) of the off stump.
    off_stump: int


def front_on_sides(batting_hand: str) -> Sides:
    """Directions for a camera behind the far stumps, looking down the pitch at the batter.

    A right-hander faces the off side with the left shoulder toward the bowler, so from the
    bowler's end the off side is on the camera's left. A left-hander is the mirror image.
    """
    # Right-hander: off side is image-left, front leg is the player's left leg, off stump is leftmost.
    if batting_hand == "right":
        # Return the right-hander's directions.
        return Sides(off_dir=-1, front="left", back="right", off_stump=0)
    # Left-hander: everything mirrored.
    return Sides(off_dir=+1, front="right", back="left", off_stump=2)
