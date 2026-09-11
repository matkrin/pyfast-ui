from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift
from skimage.registration import phase_cross_correlation

from pyfast_ui.pyfast_re.channels import Channels
from pyfast_ui.pyfast_re.data_mode import DataMode, reshape_data

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


log = logging.getLogger(__name__)

NUM_FRAMES_TO_CORRELATE = 10
"""How many frames the automatic detection averages over. One frame is enough
to get a number, not enough to get the same number twice."""

ROWS_PER_Y_PHASE = 4
"""A y-phase step rolls the series by two lines, and because the up and the
down frame move in opposite directions their relative offset changes by four
rows. Measured over -10 to +10 on three movies with no scatter."""


@dataclass
class PhaseCorrectionResult:
    data: NDArray[np.float32]
    applied_x_phase: float
    applied_y_phase: int
    updown_row_shift: int
    """Rows by which the down frames still have to be moved after the y-phase
    roll. The roll can only reach multiples of four rows, so an odd offset
    leaves one row behind."""


@final
class PhaseCorrection:
    def __init__(
        self,
        fast_movie: FastMovie,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int | None = None,
        auto_y_phase: bool = False,
        fractional_x_phase: bool = False,
    ):
        self.fast_movie = fast_movie
        self.auto_x_phase = auto_x_phase
        self.frame_index_to_correlate = frame_index_to_correlate
        self.sigma_gauss = sigma_gauss
        self.additional_x_phase = additional_x_phase
        self.manual_y_phase = manual_y_phase
        self.auto_y_phase = auto_y_phase
        self.fractional_x_phase = fractional_x_phase

    def _as_movie(self, series: NDArray[np.float32]) -> NDArray[np.float32]:
        metadata = self.fast_movie.metadata
        return reshape_data(
            series,
            Channels.UDI,
            metadata.num_images,
            metadata.scanner_x_points,
            metadata.scanner_y_points,
        )

    def correct_phase(self) -> PhaseCorrectionResult:
        """Determine and apply the x and y phase. Does not mutate the movie."""
        metadata = self.fast_movie.metadata
        num_x_points = metadata.scanner_x_points

        series: NDArray[np.float32] = (
            self.fast_movie.data.flatten()
            if self.fast_movie.mode == DataMode.MOVIE
            else self.fast_movie.data
        )

        x_phase: float = 0.0
        if self.auto_x_phase:
            movie = self._as_movie(series.copy())
            magnitude = x_phase_magnitude(
                movie, self.frame_index_to_correlate, self.sigma_gauss
            )
            x_phase = self._resolve_sign(series, magnitude)

        x_phase += self.additional_x_phase

        y_phase = metadata.acquisition_y_phase
        row_shift = 0
        if self.auto_y_phase:
            probe = self._as_movie(np.roll(series, int(round(x_phase))))
            offset = updown_row_offset(probe)
            y_phase = int(round(offset / ROWS_PER_Y_PHASE))
            row_shift = int(round(offset - ROWS_PER_Y_PHASE * y_phase))
            log.info(
                "Automatic y-phase: up/down offset %.1f rows -> y-phase %d, "
                "%d row(s) left for the frames",
                offset,
                y_phase,
                row_shift,
            )
        elif self.manual_y_phase is not None:
            y_phase = self.manual_y_phase

        whole = int(round(x_phase))
        remainder = x_phase - whole if self.fractional_x_phase else 0.0

        data = np.roll(series, whole + y_phase * num_x_points * 2)
        if remainder:
            # A fractional phase cannot be expressed as a roll, so the series is
            # interpolated. Worth about three to four percent of the residual
            # doubling on the movies measured, hence optional.
            data = ndshift(data, remainder, order=3, mode="nearest").astype(np.float32)

        log.info(
            "Phase correction applied: x %.3f samples%s, y %d",
            x_phase,
            "" if remainder else " (rounded)",
            y_phase,
        )

        return PhaseCorrectionResult(data, x_phase, y_phase, row_shift)

    def _resolve_sign(self, series: NDArray[np.float32], magnitude: float) -> float:
        """Decide whether the phase is positive or negative.

        The correlation gives the size of the forward to backward misalignment
        but not its direction, because `reshape_data` flips the up frames
        vertically and thereby swaps which row parity is the forward one. Which
        way round that comes out differs from movie to movie, so it is settled
        by trying both and keeping the one that leaves less doubling.
        """
        if magnitude < 0.05:
            return 0.0

        scores: dict[float, float] = {}
        for candidate in (magnitude, -magnitude):
            movie = self._as_movie(np.roll(series, int(round(candidate))))
            scores[candidate] = updown_line_mismatch(movie)

        best = min(scores, key=lambda c: scores[c])
        log.info(
            "Automatic x-phase: magnitude %.3f samples, %+.3f leaves %.1f and "
            "%+.3f leaves %.1f, taking %+.3f",
            magnitude,
            magnitude,
            scores[magnitude],
            -magnitude,
            scores[-magnitude],
            best,
        )
        return best


def _line_lag(frame: NDArray[np.float32]) -> float:
    """Displacement between neighbouring lines of one frame, in pixels.

    The correlation functions of all line pairs are summed and the peak is taken
    once, rather than taking a peak per line pair and averaging those. Averaging
    peak positions discards the evidence that lets a noisy line be outvoted, and
    it quantises every contribution to a whole pixel.

    Each correlation is divided by the number of overlapping samples. Without
    that the sum grows with the overlap and pulls the peak towards zero lag.
    """
    work = frame.astype(np.float64)
    work = work - work.mean(axis=1, keepdims=True)
    spread = work.std(axis=1, keepdims=True)
    spread[spread == 0.0] = 1.0
    work = work / spread

    num_rows, num_cols = work.shape
    overlap = np.concatenate(
        [np.arange(1, num_cols + 1), np.arange(num_cols - 1, 0, -1)]
    ).astype(np.float64)

    rows = np.linspace(2, num_rows - 3, min(60, max(num_rows - 4, 1))).astype(int)
    accumulated = np.zeros(2 * num_cols - 1)
    for row in rows:
        accumulated += np.correlate(work[row], work[row + 1], mode="full") / overlap

    peak = int(np.argmax(accumulated))
    left = accumulated[max(peak - 1, 0)]
    middle = accumulated[peak]
    right = accumulated[min(peak + 1, len(accumulated) - 1)]
    curvature = left - 2.0 * middle + right
    offset = (left - right) / (2.0 * curvature) if curvature != 0.0 else 0.0

    return (peak + offset) - (num_cols - 1)


def x_phase_magnitude(
    data: NDArray[np.float32],
    index_frame_to_correlate: int,
    sigma_gauss: int,
    num_frames: int = NUM_FRAMES_TO_CORRELATE,
) -> float:
    """Size of the x-phase in samples, without its sign.

    Args:
        data: Movie data, three dimensional.
        index_frame_to_correlate: First frame to use.
        sigma_gauss: Optional smoothing of the forward and backward lines.
        num_frames: How many consecutive frames to average over.

    Returns:
        Half the median absolute line misalignment, which is the phase, as a
        non-negative number. The sign is not observable here, see
        `PhaseCorrection._resolve_sign`.
    """
    if len(data.shape) != 3:
        raise ValueError("`data` must be 3 dimensional numpy array")

    start = max(0, min(index_frame_to_correlate, len(data) - 1))
    stop = min(len(data), start + max(1, num_frames))

    lags: list[float] = []
    for index in range(start, stop):
        frame = data[index].astype(np.float32, copy=True)
        if sigma_gauss != 0:
            frame[::2] = gaussian_filter(frame[::2], sigma_gauss)
            frame[1::2] = gaussian_filter(frame[1::2], sigma_gauss)
        lags.append(_line_lag(frame))

    magnitude = float(np.median(np.abs(lags))) / 2.0
    log.info(
        "Automatic x-phase: line misalignment over %d frames, median absolute "
        "%.3f px, spread %.3f px",
        stop - start,
        2.0 * magnitude,
        float(np.std(np.abs(lags))),
    )

    return magnitude


def updown_line_mismatch(
    data: NDArray[np.float32], num_frames: int = NUM_FRAMES_TO_CORRELATE
) -> float:
    """How much the forward and the backward lines of a frame disagree.

    This is the doubling artefact itself: the two sub-images sample the same
    surface, so at the correct x-phase their difference is smallest.
    """
    scores: list[float] = []
    for index in range(min(num_frames, len(data))):
        frame = data[index].astype(np.float64)
        forward, backward = frame[0::2], frame[1::2]
        rows = min(len(forward), len(backward))
        scores.append(
            float(np.sqrt(((forward[:rows] - backward[:rows]) ** 2).mean()))
        )

    return float(np.mean(scores)) if scores else float("inf")


def updown_row_offset(
    data: NDArray[np.float32], num_pairs: int = 6
) -> float:
    """Vertical offset between an up frame and the following down frame, in rows.

    A wrong y-phase puts the two sweeps of one image at different heights, which
    is what makes features double in y. The masked variant of the correlation is
    used because the plain one grows with the overlap and would favour zero.
    """
    if len(data) < 2:
        return 0.0

    valid = np.ones(data[0].shape, dtype=bool)
    offsets: list[float] = []
    for index in range(0, min(2 * num_pairs, len(data) - 1), 2):
        result = phase_cross_correlation(  # pyright: ignore[reportUnknownVariableType]
            data[index].astype(np.float64),
            data[index + 1].astype(np.float64),
            upsample_factor=1,
            reference_mask=valid,
            moving_mask=valid,
        )
        shift = result[0] if isinstance(result, tuple) else result
        offsets.append(float(np.asarray(shift)[0]))

    return float(np.median(offsets))


def get_x_phase_autocorrection(
    data: NDArray[np.float32], index_frame_to_correlate: int, sigma_gauss: int
) -> int:
    """Kept for callers outside the GUI. Returns a rounded, unsigned phase."""
    return int(round(x_phase_magnitude(data, index_frame_to_correlate, sigma_gauss)))
