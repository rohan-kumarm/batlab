"""Reads video information and frames using ffmpeg."""

# json parses ffprobe's output.
import json
# subprocess runs the ffmpeg and ffprobe programs.
import subprocess
# dataclass writes __init__ for the VideoInfo class.
from dataclasses import dataclass
# Path type for file arguments.
from pathlib import Path

# numpy holds frames as arrays of pixels.
import numpy as np

# iPhones record HDR video (HLG). Without converting it to normal (SDR) colour, frames look
# washed out and the stump colour mask fails. This ffmpeg filter chain does that conversion.
HDR_TO_SDR = (
    # Convert the HLG signal to linear light.
    "zscale=t=linear:npl=100,"
    # Switch to floating-point RGB so the next steps keep precision.
    "format=gbrpf32le,"
    # Convert the wide BT.2020 colours to normal BT.709 colours.
    "zscale=p=bt709,"
    # Compress bright highlights into the normal range (the "hable" curve).
    "tonemap=hable,"
    # Convert back to a normal BT.709 video signal.
    "zscale=t=bt709:m=bt709:r=tv"
)


# Read-only record of the facts about one video file.
@dataclass(frozen=True)
class VideoInfo:
    # Path to the file.
    path: Path
    # Width in pixels after rotation is applied (what you see on the phone).
    width: int
    # Height in pixels after rotation is applied.
    height: int
    # Frames per second.
    fps: float
    # Length in seconds.
    duration: float
    # True if the video is HDR and needs the HDR_TO_SDR conversion.
    hdr: bool


def probe(path: str | Path) -> VideoInfo:
    """Read size, frame rate, length and HDR status with ffprobe."""
    # Ask ffprobe for stream and format details as JSON.
    out = subprocess.run(
        # -v error hides log noise; -print_format json gives machine-readable output.
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        # Capture the output so Python can read it; check=True raises an error if ffprobe fails.
        capture_output=True, check=True, text=True,
    ).stdout
    # Parse the JSON text into a dict.
    meta = json.loads(out)
    # Pick the first video stream (iPhone files also have audio and metadata streams).
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    # avg_frame_rate is a fraction like "18600/311"; split it into its two parts.
    num, den = (int(x) for x in v["avg_frame_rate"].split("/"))
    # Default rotation is 0 degrees.
    rotation = 0
    # Phones store portrait video as landscape plus a rotation tag; look through side data for it.
    for sd in v.get("side_data_list", []):
        # The "Display Matrix" entry carries the rotation.
        if "rotation" in sd:
            # Keep the rotation value, for example -90.
            rotation = int(sd["rotation"])
    # Stored width and height before rotation.
    w, h = int(v["width"]), int(v["height"])
    # A 90-degree rotation swaps width and height (ffmpeg rotates frames automatically).
    if abs(rotation) % 180 == 90:
        # Swap the two values.
        w, h = h, w
    # HLG ("arib-std-b67") and PQ ("smpte2084") are the two HDR transfer types.
    hdr = v.get("color_transfer") in ("arib-std-b67", "smpte2084")
    # Return everything in one VideoInfo.
    return VideoInfo(Path(path), w, h, num / den, float(meta["format"]["duration"]), hdr)


def read_frames(info: VideoInfo, start_s: float, count: int, scale_width: int | None = None) -> np.ndarray:
    """Return `count` consecutive frames from `start_s` seconds as an array (count, H, W, 3), BGR."""
    # List of filters to apply; HDR conversion first if needed.
    filters = [HDR_TO_SDR] if info.hdr else []
    # Optional downscale to make scanning faster; -2 keeps the height even, as encoders need.
    if scale_width:
        # Add the scale filter.
        filters.append(f"scale={scale_width}:-2")
    # Final pixel format: 3 bytes per pixel in blue-green-red order, which OpenCV expects.
    filters.append("format=bgr24")
    # Work out the output size so the raw bytes can be split into frames.
    if scale_width:
        # Width is the requested width.
        out_w = scale_width
        # Height keeps the aspect ratio, rounded down to an even number.
        out_h = int(round(info.height * scale_width / info.width / 2)) * 2
    else:
        # Without scaling, the output is the rotated full size.
        out_w, out_h = info.width, info.height
    # Build the ffmpeg command.
    cmd = [
        # -v error hides logs.
        "ffmpeg", "-v", "error",
        # -ss before -i seeks quickly; ffmpeg still decodes up to the exact frame.
        "-ss", f"{max(start_s, 0):.4f}",
        # Input file.
        "-i", str(info.path),
        # Apply the filter chain.
        "-vf", ",".join(filters),
        # Stop after this many frames.
        "-frames:v", str(count),
        # Write raw pixels to standard output instead of a file.
        "-f", "rawvideo", "-",
    ]
    # Run ffmpeg and collect the raw bytes.
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    # Number of bytes in one frame.
    frame_bytes = out_w * out_h * 3
    # Number of whole frames received (fewer than asked if the clip ended).
    n = len(raw) // frame_bytes
    # Turn the bytes into a (frames, height, width, 3) array without copying.
    return np.frombuffer(raw[: n * frame_bytes], dtype=np.uint8).reshape(n, out_h, out_w, 3)


def read_audio(path: str | Path, sample_rate: int = 48000) -> np.ndarray:
    """Return the first audio track as mono float32 samples."""
    # Run ffmpeg to decode audio only.
    raw = subprocess.run(
        # -map 0:a:0 picks the first audio track (iPhones add a second spatial-audio track);
        # -ac 1 mixes to mono; -ar sets the sample rate; -f f32le outputs raw 32-bit floats.
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-vn",
         "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        # Capture the bytes; raise an error if ffmpeg fails.
        capture_output=True, check=True,
    ).stdout
    # Interpret the bytes as float32 samples.
    return np.frombuffer(raw, dtype=np.float32)
