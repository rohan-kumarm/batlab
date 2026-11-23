"""Loads a session settings file and the target ranges."""

# tomllib reads .toml files; it is built into Python 3.11+.
import tomllib
# dataclass writes the __init__ method for simple data-holding classes.
from dataclasses import dataclass
# Path handles file paths the same way on Mac, Windows and Linux.
from pathlib import Path


# frozen=True makes a Session read-only after it is created, so no step can change it by accident.
@dataclass(frozen=True)
class Session:
    # Player's name, used in report file names.
    player: str
    # "right" or "left".
    batting_hand: str
    # "machine" or "bowler".
    delivery_source: str
    # Stump height in cm, used for the pixel-to-cm scale.
    stump_height_cm: float
    # Path to the front-on video, or None if not given.
    front_video: Path | None
    # Path to the side-on video, or None if not given.
    side_video: Path | None
    # Machine speed in mph, or None for bowler sessions.
    machine_speed_mph: float | None
    # "tune" (used to improve the engine) or "test" (held back, used only to measure final accuracy).
    split: str = "tune"
    # Number of balls the batter actually faced, counted by them; None if not recorded.
    balls_faced: int | None = None


def load_session(path: str | Path) -> Session:
    """Read a sessions/*.toml file into a Session."""
    # Open the file in binary mode, which tomllib requires.
    with open(path, "rb") as f:
        # Parse the TOML text into nested Python dictionaries.
        raw = tomllib.load(f)
    # Pull the batting hand and make it lowercase so "Right" also works.
    hand = raw["player"]["batting_hand"].lower()
    # Stop early with a clear message if the hand is not one of the two allowed values.
    if hand not in ("right", "left"):
        # Raising ValueError ends the run and shows this message.
        raise ValueError(f"batting_hand must be 'right' or 'left', got {hand!r}")
    # Read the video section, or an empty dict if the section is missing.
    videos = raw.get("videos", {})
    # Build and return the Session object from the parsed values.
    return Session(
        # Player name.
        player=raw["player"]["name"],
        # Checked batting hand.
        batting_hand=hand,
        # Machine or bowler, lowercased.
        delivery_source=raw["session"]["delivery_source"].lower(),
        # Stump height, defaulting to regulation 71.1 cm.
        stump_height_cm=float(raw.get("calibration", {}).get("stump_height_cm", 71.1)),
        # Front video path; an empty string becomes None.
        front_video=Path(videos["front"]) if videos.get("front") else None,
        # Side video path; an empty string becomes None.
        side_video=Path(videos["side"]) if videos.get("side") else None,
        # Machine speed if present, otherwise None.
        machine_speed_mph=raw["session"].get("machine_speed_mph"),
        # Tune or test set; sessions without the setting count as tune.
        split=raw["session"].get("split", "tune").lower(),
        # Optional count of balls faced, to check the release finder against.
        balls_faced=raw["session"].get("balls_faced"),
    )


def load_toml(path: str | Path) -> dict:
    """Read any TOML file (targets, reference values) into a dict."""
    # Open in binary mode for tomllib.
    with open(path, "rb") as f:
        # Return the parsed dictionary.
        return tomllib.load(f)
