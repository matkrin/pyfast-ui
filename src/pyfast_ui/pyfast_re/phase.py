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
from pyfast_ui.pyfast_re import updown
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

    updown_profile: NDArray[np.float64] | None = None
    """The up/down mismatch of the corrected series, per row block, measured on
    the interlaced movie. Kept because this is the last point in the pipeline
    where the image still carries its slow, non-periodic structure; the FFT
    filter takes that structure out and leaves a lattice movie whose
    registration is ambiguous by whole lattice periods. `None` unless the
    y-phase was determined automatically, since only then is it measured."""


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
        profile: NDArray[np.float64] | None = None
        if self.auto_y_phase:
            y_phase, row_shift, profile = self._search_y_phase(series, x_phase)
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

        return PhaseCorrectionResult(data, x_phase, y_phase, row_shift, profile)

    def _search_y_phase(
        self, series: NDArray[np.float32], x_phase: float, span: int = 2
    ) -> tuple[int, int, NDArray[np.float64]]:
        """Pick the y-phase that leaves the two sweeps closest together.

        A first estimate comes from dividing the measured offset by the rows a
        y-phase step is worth, but that step is not reliably four rows: measured
        on FS_251128_026 it came out anywhere between three and seven, so a
        movie tens of rows out lands two steps away from the best value. Each
        candidate around the estimate is therefore tried and measured. Every
        measurement that decides the outcome is then made close to zero offset,
        which is where the correlation is most trustworthy.

        Returns:
            The y-phase, the rows that are left for `reshape_data` to shift, and
            the mismatch profile of the chosen candidate.
        """
        num_x_points = self.fast_movie.metadata.scanner_x_points
        whole_x = int(round(x_phase))

        offset = updown_row_offset(self._as_movie(np.roll(series, whole_x)))
        estimate = int(round(offset / ROWS_PER_Y_PHASE))

        residuals: dict[int, float] = {}
        profiles: dict[int, NDArray[np.float64]] = {}
        for candidate in range(estimate - span, estimate + span + 1):
            probe = self._as_movie(
                np.roll(series, whole_x + candidate * num_x_points * 2)
            )
            profiles[candidate] = updown.mismatch_profile(probe, pairs=8)
            residuals[candidate] = float(updown.constant_offset(profiles[candidate])[0])

        best = min(residuals, key=lambda c: abs(residuals[c]))
        log.info(
            "Automatic y-phase: offset %.1f rows suggests %d; tried %s, taking %d "
            "with %.2f rows left",
            offset,
            estimate,
            ", ".join(f"{c}: {residuals[c]:+.1f}" for c in sorted(residuals)),
            best,
            residuals[best],
        )

        return best, int(round(residuals[best])), profiles[best]

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


def updown_row_offset(data: NDArray[np.float32]) -> float:
    """Vertical offset between an up frame and the following down frame, in rows.

    A wrong y-phase puts the two sweeps of one image at different heights, which
    is what makes features double in y.

    Only the constant part of the offset belongs to the y-phase. The two sweeps
    share their turning points, so at the first and the last line of a frame
    they are at the same height and what is measured there is the y-phase alone;
    the lines in between carry the delayed response of the piezo on top, which
    is creep and is the creep correction's business. Measuring over the whole
    frame, as this used to, returns the sum of the two and hands part of the
    creep signal to the y-phase, which then removes it.
    """
    if len(data) < 3:
        return 0.0

    profile = updown.mismatch_profile(data)
    log.info(updown.describe(profile))

    return float(updown.constant_offset(profile)[0])


def get_x_phase_autocorrection(
    data: NDArray[np.float32], index_frame_to_correlate: int, sigma_gauss: int
) -> int:
    """Kept for callers outside the GUI. Returns a rounded, unsigned phase."""
    return int(round(x_phase_magnitude(data, index_frame_to_correlate, sigma_gauss)))
