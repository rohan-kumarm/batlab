"""Finds the release moment of each delivery from the bowling machine's firing sound."""

# dataclass for the settings and result records.
from dataclasses import dataclass

# numpy for array math.
import numpy as np
# scipy.signal has the filter design, filtering and peak-finding functions.
from scipy import signal


# Tunable settings for release detection. Defaults were set on the 10 Nov front-on clip.
@dataclass(frozen=True)
class ReleaseParams:
    # Ignore sound below this frequency (Hz); machine and bat clicks are sharp, high-pitched sounds.
    highpass_hz: float = 1500.0
    # Length of the loudness window in seconds (3 ms keeps the onset sharp).
    window_s: float = 0.003
    # A sound must be this many times louder than the clip's typical background level.
    min_ratio_to_floor: float = 20.0
    # A sound this loud (fraction of the clip's loudest sound) counts as a release on its own.
    strong_fraction_of_max: float = 0.5
    # Quieter sounds down to this fraction are considered too, but only accepted as a release if
    # a second sharp sound (the bat) follows within the pair window below. On the 19 Nov clip
    # IMG_0551 one machine sound reached 0.497 of the loudest (a bat hit), so a single 0.5 rule
    # took the bat as the release.
    min_fraction_of_max: float = 0.3
    # Machine-to-bat time range (s) that confirms a quieter sound as a release. Measured: 0.70-0.75 s
    # at 60-70 mph; the range allows roughly 50 to 80 mph.
    pair_min_s: float = 0.55
    # Upper end of the confirming range.
    pair_max_s: float = 0.9
    # The confirming sound only needs to be this many times the background level. A ball that is
    # left or edged makes only a faint sound (10-20x background on IMG_0552), far below the
    # release threshold; requiring a loud one missed 3 of 30 deliveries there.
    pair_min_ratio_to_floor: float = 10.0
    # Two sounds closer than this (s) are treated as one sound.
    min_peak_spacing_s: float = 0.15
    # After a release, sounds within this many seconds belong to the same delivery (bat, net).
    delivery_window_s: float = 2.0
    # A bat contact sound is expected this long after release (s); used to report audio impact.
    impact_min_s: float = 0.45
    # Upper limit for the bat contact sound after release (s).
    impact_max_s: float = 1.2


# Result for one delivery.
@dataclass(frozen=True)
class Release:
    # Delivery number, starting at 1.
    index: int
    # Time of release in seconds from the start of the clip (sound onset).
    time_s: float
    # Loudness of the release sound divided by the background level.
    strength: float
    # Time of a likely bat contact sound, or None if none was heard.
    impact_audio_s: float | None


def loudness_envelope(audio: np.ndarray, sr: int, p: ReleaseParams) -> np.ndarray:
    """Return the short-window loudness of the high-passed audio, one value per sample."""
    # Design a 4th-order Butterworth high-pass filter; the cutoff is given as a fraction of sr/2.
    b, a = signal.butter(4, p.highpass_hz / (sr / 2), btype="high")
    # filtfilt runs the filter forward and backward so sound onsets are not delayed.
    hp = signal.filtfilt(b, a, audio)
    # Number of samples in the loudness window.
    win = max(1, int(p.window_s * sr))
    # Root-mean-square loudness: average the squared signal over the window, then square root.
    return np.sqrt(np.convolve(hp**2, np.ones(win) / win, mode="same"))


def find_releases(audio: np.ndarray, sr: int, p: ReleaseParams = ReleaseParams()) -> list[Release]:
    """Return one Release per delivery found in the audio."""
    # Loudness over time.
    env = loudness_envelope(audio, sr, p)
    # Background level: the median ignores the few loud moments.
    floor = float(np.median(env))
    # The loudness a candidate must exceed: the larger of the two rules.
    threshold = max(floor * p.min_ratio_to_floor, float(env.max()) * p.min_fraction_of_max)
    # Loudness at which a sound counts as a release without needing a bat sound after it.
    strong = float(env.max()) * p.strong_fraction_of_max
    # Find local loudness peaks above the threshold, at least min_peak_spacing_s apart.
    peaks, _ = signal.find_peaks(env, height=threshold, distance=int(p.min_peak_spacing_s * sr))
    # All peaks above the much lower confirming level, used only to confirm quieter releases.
    faint, _ = signal.find_peaks(env, height=floor * p.pair_min_ratio_to_floor, distance=int(p.min_peak_spacing_s * sr))
    # Collected results.
    releases: list[Release] = []
    # Time (s) until which new peaks are treated as part of the current delivery.
    busy_until = -1.0
    # Walk through the peaks in time order.
    for i, pk in enumerate(peaks):
        # Peak time in seconds.
        t = pk / sr
        # Skip peaks that belong to the previous delivery (bat contact, ball hitting the net).
        if t < busy_until:
            # Go to the next peak.
            continue
        # A quieter sound needs a second sound 0.55-0.9 s later (machine, then bat) to count.
        if env[pk] < strong:
            # Is any faint-or-louder peak inside the machine-to-bat range?
            paired = any(p.pair_min_s <= (later - pk) / sr <= p.pair_max_s for later in faint)
            # Without one it is probably a stray knock, not a delivery.
            if not paired:
                # Go to the next peak.
                continue
        # Find the onset: step back from the peak while loudness stays above 20% of the peak.
        start = pk
        # Stop at the clip start or when loudness drops below 20% of this peak.
        while start > 0 and env[start - 1] > 0.2 * env[pk]:
            # Move one sample earlier.
            start -= 1
        # Look for a bat sound among the following peaks inside the expected window.
        impact = None
        # Check the peaks that come after this one.
        for later in peaks[i + 1:]:
            # Time between release peak and this later peak.
            gap = (later - pk) / sr
            # Stop searching once past the window.
            if gap > p.impact_max_s:
                # No need to check further peaks.
                break
            # Accept the first peak inside the window.
            if gap >= p.impact_min_s:
                # Record its time in seconds.
                impact = later / sr
                # Only the first match is used.
                break
        # Save this delivery's result.
        releases.append(Release(len(releases) + 1, start / sr, float(env[pk] / floor), impact))
        # Mark the next delivery_window_s seconds as belonging to this delivery.
        busy_until = t + p.delivery_window_s
    # Return all deliveries found.
    return releases
