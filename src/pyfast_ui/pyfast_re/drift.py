"""
First try at movie stabilisation with
OpenCV. Parts of the code are inspired by:
learnopencv.com/video-stabilisation-using-point-feature-matching-in-opencv
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, final
import logging

import numpy as np
from numpy.typing import NDArray
from pyfast_ui.pyfast_re.tqdm_logging import TqdmLogger
import skimage
from pystackreg import StackReg  # pyright: ignore[reportMissingTypeStubs]
from scipy import sparse
from scipy.ndimage import convolve, gaussian_filter, shift as ndshift
from scipy.signal import correlate, medfilt
from scipy.sparse.linalg import lsqr
from skimage.registration import phase_cross_correlation

from pyfast_ui.pyfast_re.data_mode import DataMode

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


log = logging.getLogger(__name__)


class DriftMode(Enum):
    FULL = "full"
    COMMON = "common"


class KnownDriftType(Enum):
    INTEGRATED = "integrated"
    SEQUENTIAL = "sequential"


class StackRegReferenceType(Enum):
    PREVIOUS = "previous"
    FIRST = "first"
    MEAN = "mean"


# @final
# class Drift:
#     """
#     Initialise Drift class with Fast movie instance to then
#     drift correct the movie data.

#     Args:
#         FastmovieInstance: FastMovie object
#         stepsize: integer, the difference between frames that are correlated
#         corrspeed: int, the difference between two correlation windows
#         show_path: Parameter, if True rare and filter drift path are plotted.
#         boxcar: Parameter of the boxcar filter that applied to the
#             drift path. Set to 0 if no boxcar filter should be applied
#         median_filter: Paramter to decide if the drift path should be smoothed
#             via a median_filter of kernel size 3
#     """

#     def __init__(
#         self,
#         fast_movie: FastMovie,
#         stepsize: int = 40,
#         corrspeed: int = 1,
#         show_path: bool = False,
#         boxcar: int = 50,
#         median_filter: bool = True,
#     ):
#         if fast_movie.mode != DataMode.MOVIE:
#             raise ValueError(f"`FastMovie` instance must be in mode {DataMode.MOVIE}")
#         if fast_movie.channels is None:
#             raise ValueError("`FastMovie.channels must be set`")

#         self.data = fast_movie.data
#         self.file = fast_movie.filename.replace(".h5", ".drift.txt")
#         # self.processing_log = fast_movie.processing_log
#         self.channels = fast_movie.channels
#         self.stepsize = stepsize
#         self.corrspeed = corrspeed
#         self.n_frames = np.shape(self.data)[0]
#         self.img_width = np.shape(self.data)[2]
#         self.img_height = np.shape(self.data)[1]
#         self.boxcar = boxcar
#         self.median_filter = median_filter

#         if self.img_width > self.img_height:
#             self.im_size = 2 ** (int(np.log2(self.img_width)) + 1)
#         else:
#             self.im_size = 2 ** (int(np.log2(self.img_height)) + 1)

#         self.convdims = (self.im_size * 2 - 1, self.im_size * 2 - 1)
#         self.transformations = np.zeros((2, self.n_frames))
#         self.integrated_trans = None
#         self.show_path = show_path

#         # if self.stepsize is None:
#         #     self.stepsize = int(self.n_frames / 3)

#     def correct_correlation(
#         self,
#         mode: DriftMode = DriftMode.FULL,
#     ):
#         self._get_drift_correlation()
#         self._filter_drift()
#         self._write_drift()

#         match mode:
#             case DriftMode.FULL:
#                 return self._adjust_movie_buffered(), self.integrated_trans
#             case DriftMode.COMMON:
#                 return self._adjust_movie_common(), self.integrated_trans

#     def correct_stackreg(
#         self,
#         mode: DriftMode = DriftMode.FULL,
#         stackreg_reference: StackRegReferenceType = StackRegReferenceType.PREVIOUS,
#     ):
#         self._get_drift_stackreg(stackreg_reference)
#         self._filter_drift()
#         self._write_drift()

#         match mode:
#             case DriftMode.FULL:
#                 return self._adjust_movie_buffered(), self.integrated_trans
#             case DriftMode.COMMON:
#                 return self._adjust_movie_common(), self.integrated_trans

#     def correct_known(
#         self,
#         mode: DriftMode = DriftMode.FULL,
#         known_drift_type: KnownDriftType = KnownDriftType.INTEGRATED,
#     ):
#         driftfile = self.file.replace(".h5", ".drift.txt")

#         match known_drift_type:
#             case KnownDriftType.INTEGRATED:
#                 self.integrated_trans = np.loadtxt(driftfile).T[0:2, :]
#                 # self.processing_log.info("Known drift used: {}".format(known_drift_type))
#             case KnownDriftType.SEQUENTIAL:
#                 self.transformations = np.loadtxt(driftfile).T[2:4, :]
#                 self.integrated_trans = np.cumsum(self.transformations, axis=1)
#                 self._write_drift()
#                 # self.processing_log.info("Known drift used: {}".format(known_drift))

#         match mode:
#             case DriftMode.FULL:
#                 return self._adjust_movie_buffered(), self.integrated_trans
#             case DriftMode.COMMON:
#                 return self._adjust_movie_common(), self.integrated_trans

#     def _get_drift_correlation(self) -> None:
#         """Calculation of the drift by fft correlation."""
#         movie = np.zeros((self.n_frames, self.im_size, self.im_size))
#         hamm = np.sqrt(
#             np.outer(np.hamming(self.img_height), np.hamming(self.img_width))
#         )
#         for i in range(self.n_frames):
#             imag = self.data[i, :, :].copy()
#             imag /= imag.std()
#             imag -= imag.mean()
#             imag = hamm * imag
#             movie[i, :, :] = resize(
#                 imag, (self.im_size, self.im_size), anti_aliasing=True, order=0
#             )
#         for i in range(self.n_frames):
#             try:
#                 fftd = correlate(
#                     movie[self.corrspeed * i, :, :],
#                     movie[self.corrspeed * i + self.stepsize, :, :],
#                     method="fft",
#                 )
#                 maxind = np.argmax(fftd)
#                 indices = np.unravel_index(maxind, self.convdims)
#                 print("nocaling", indices)
#                 effektive_shift = np.asarray(
#                     [
#                         [(-(self.im_size - 1) + indices[0]) / self.stepsize],
#                         [(indices[1] - (self.im_size - 1)) / self.stepsize],
#                     ]
#                 )
#                 self.transformations[:, i] = effektive_shift.T
#             except Exception:
#                 pass
#         # print("last found correlation indices are {}".format(indices))

#     def _get_drift_stackreg(self, reference: StackRegReferenceType) -> None:
#         stackreg = StackReg(StackReg.TRANSLATION)
#         transformation_matrices = stackreg.register_stack(
#             self.data, reference=reference.value
#         )
#         x_path_integrated = []
#         y_path_integrated = []
#         for matrix in transformation_matrices:
#             x_path_integrated.append(-matrix[0, 2])
#             y_path_integrated.append(-matrix[1, 2])

#         # self.integrated_trans = np.array([y_path_integrated, x_path_integrated])
#         x_path = np.array(x_path_integrated)
#         x_path = np.diff(x_path, prepend=0)
#         y_path = np.array(y_path_integrated)
#         y_path = np.diff(y_path, prepend=0)

#         self.transformations = np.stack((y_path, x_path))

#     def _filter_drift(self):
#         """
#         smooth and filter drift path
#         """
#         boxwidth = self.boxcar boxcar = np.ones((1, boxwidth)) / boxwidth
#         boxcar = boxcar[0, :]

#         if self.median_filter:
#             self.transformations[0, :] = medfilt(self.transformations[0, :], 3)
#             self.transformations[1, :] = medfilt(self.transformations[1, :], 3)

#         self.integrated_trans = np.cumsum(self.transformations, axis=1)
#         # linear extrapolation
#         pos = np.linspace(0, self.n_frames - 1, self.n_frames)
#         k1, d1 = np.polyfit(
#             pos[: -self.stepsize], self.integrated_trans[0, : -self.stepsize], 1
#         )
#         k2, d2 = np.polyfit(
#             pos[: -self.stepsize], self.integrated_trans[1, : -self.stepsize], 1
#         )
#         self.integrated_trans[0, -self.stepsize :] = d1 + k1 * pos[-self.stepsize :]
#         self.integrated_trans[1, -self.stepsize :] = d2 + k2 * pos[-self.stepsize :]

#         if self.boxcar != 0:
#             # self.processing_log.info( "Boxcar filter used with boxsize: {}".format(boxwidth))
#             transformations_conv = np.zeros((2, self.n_frames))
#             transformations_conv[0, :] = convolve(self.integrated_trans[0], boxcar)
#             transformations_conv[1, :] = convolve(self.integrated_trans[1], boxcar)
#             self.integrated_trans = transformations_conv

#     def _write_drift(self):
#         """
#         Writes a drift.txt file
#         """
#         with open(self.file, "w") as fileobject:
#             fileobject.write(
#                 "# {0:>10}   {1:>12}  {2:>12}  {3:>12} \n".format(
#                     "y integrated", "x integrated", "y sequential", "x sequential"
#                 )
#             )
#             for i in range(self.transformations.shape[1]):
#                 fileobject.write(
#                     "{0:>14.5f}   {1:>12.5f}  {2:>12.5f}  {3:>12.5f} \n".format(
#                         self.integrated_trans[0, i],
#                         self.integrated_trans[1, i],
#                         self.transformations[0, i],
#                         self.transformations[1, i],
#                     )
#                 )

#     def _adjust_movie_buffered(self):
#         """embed movie frames into buffered background to
#         move freely according to drift path. The image ration
#         is changed back for interlace movies (2:1) to fit the
#         overall system architecture"""
#         maxy, maxx = np.max(self.integrated_trans, 1)
#         miny, minx = np.min(self.integrated_trans, 1)
#         buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
#         # print("Buffer values are {} in x and {} in y.".format(buffx, buffy))
#         ## This is to see effect of scaling

#         if self.channels.is_interlaced():
#             self.rescale_width = int(self.im_size / 2)
#             maxx = maxx / 2
#             minx = minx / 2
#         else:
#             self.rescale_width = self.im_size

#         buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

#         corr_movie = np.zeros(
#             (self.n_frames, self.im_size + int(buffy), self.rescale_width + int(buffx)),
#             dtype=np.float32,
#         )
#         for i in range(self.n_frames):
#             shift1, shift2 = self.integrated_trans[:, i]
#             shift1 = int(np.round(shift1))

#             if self.channels.is_interlaced():
#                 shift2 = int(np.round(shift2) / 2)
#             else:
#                 shift2 = int(np.round(shift2))

#             y_start = int(abs(miny)) + 1 + shift1
#             y_end = int(abs(miny)) + 1 + self.im_size + shift1
#             x_start = int(abs(minx)) + 1 + shift2
#             x_end = int(abs(minx)) + 1 + self.rescale_width + shift2
#             # possibly there is a +1 in the i for the frame to be taken.
#             corr_movie[i, y_start:y_end, x_start:x_end] = resize(
#                 self.data[i, :, :],
#                 (self.im_size, self.rescale_width),
#                 anti_aliasing=True,
#                 order=3,
#             )

#         print("drift correction finished")
#         return corr_movie

#     def _adjust_movie_common(self):
#         """cut out section from movie frames, which stays constant during
#         the entire movie."""
#         maxy, maxx = np.max(self.integrated_trans, 1)
#         miny, minx = np.min(self.integrated_trans, 1)
#         buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
#         # print(buffx, buffy)
#         ## This is to see effect of scaling

#         if self.channels.is_interlaced():
#             self.rescale_width = int(self.im_size / 2)
#             maxx = maxx / 2
#             minx = minx / 2
#         else:
#             self.rescale_width = self.im_size

#         buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

#         corr_movie = np.zeros(
#             (self.n_frames, self.im_size - int(buffy), self.rescale_width - int(buffx)),
#             dtype=np.float32,
#         )
#         for i in range(self.n_frames):
#             y_shift, x_shift = self.integrated_trans[:, -i + 1]
#             y_shift = int(np.round(y_shift))

#             if self.channels.is_interlaced():
#                 x_shift = int(np.round(x_shift) / 2)
#             else:
#                 x_shift = int(np.round(x_shift))

#             y_start = int(abs(miny)) + 1 + y_shift
#             y_end = int(abs(miny)) + 1 + self.im_size - int(buffy) + y_shift
#             x_start = int(abs(minx)) + 1 + x_shift
#             x_end = int(abs(minx)) + 1 + self.rescale_width - int(buffx) + x_shift

#             # possibly there is a +1 in the i for the frame to be taken.
#             corr_movie[i, :, :] = resize(
#                 self.data[i, :, :],
#                 (self.im_size, self.rescale_width),
#                 anti_aliasing=True,
#                 order=4,
#             )[y_start:y_end, x_start:x_end]

#         print("drift correction finished")
#         return corr_movie


##################################################################################################################


@dataclass
class DriftCorrectionResult:
    data: NDArray[np.float32]
    drift_path_sequential: NDArray[np.float32]
    drift_path_integrated: NDArray[np.float32]


#### Global drift estimation ###################################################
#
# The sequential methods above measure one displacement per frame and integrate
# it, so their errors accumulate into a random walk that no smoothing of the
# path can undo. The estimator below measures many redundant frame pairs and
# solves for all frame positions at once, which averages those errors instead of
# summing them.

_BAND_LO, _BAND_HI = 1.0, 8.0
"""Widths of the difference-of-Gaussians band pass, in pixels. The low cut
suppresses shot noise, the high cut the slowly varying background that would
otherwise dominate the correlation."""


def _hann_window(shape: tuple[int, int]) -> NDArray[np.float64]:
    return np.outer(
        np.hanning(shape[0] + 2)[1:-1], np.hanning(shape[1] + 2)[1:-1]
    )


def _bandpass(frame: NDArray[np.float32]) -> NDArray[np.float32]:
    return gaussian_filter(frame, _BAND_LO) - gaussian_filter(frame, _BAND_HI)


def _overlap(
    a: NDArray[np.float32], b: NDArray[np.float32], dy: int, dx: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The parts of two frames that coincide when b is displaced by (dy, dx).

    Both crops are empty when the displacement exceeds the frame, which happens
    while the path is still being estimated. The bounds are clamped into
    [0, size] first: an unclamped stop can come out negative, and numpy reads a
    negative stop as an offset from the end, so one of the two crops would come
    back non-empty and of a different shape than the other.
    """
    num_y, num_x = a.shape

    def span(size: int, start: int, stop: int) -> tuple[int, int]:
        low = min(max(start, 0), size)
        high = min(max(stop, low), size)
        return low, high

    ay0, ay1 = span(num_y, dy, num_y + dy)
    by0, by1 = span(num_y, -dy, num_y - dy)
    ax0, ax1 = span(num_x, dx, num_x + dx)
    bx0, bx1 = span(num_x, -dx, num_x - dx)

    return a[ay0:ay1, ax0:ax1], b[by0:by1, bx0:bx1]


_MIN_OVERLAP = 0.5
"""Smallest share of a frame that two frames must still share for a match to
count. The normalised correlation divides by the overlap, so a displacement
that leaves only a sliver in common is scored on a handful of pixels and can
beat the true peak. `skimage` allows down to 0.3 by default, which permits a
displacement of two thirds of a frame between neighbouring frames; that is far
outside anything a piezo drift produces and it is exactly where the spurious
peaks sit."""

_MAX_STEP = 0.2
"""Largest displacement between two neighbouring frames that the sequential
chain accepts, as a share of the frame. Beyond this the estimate is treated as
a failed match and the step is set to zero, which leaves that frame to the
redundant pairs of the global fit. A wrong jump of half a frame, in contrast,
is never recovered, because every later pair is measured around it."""


def _coarse_shift(
    a: NDArray[np.float32], b: NDArray[np.float32]
) -> NDArray[np.float64]:
    """Whole-pixel displacement, searched over the full range.

    Plain cross correlation sums over the overlap and therefore grows with it,
    which biases the result towards small displacements. The normalised variant
    divides that out, at some cost in speed, and needs a floor on the overlap
    so that it does not bias the other way.
    """
    ones = np.ones(a.shape, dtype=bool)
    result = phase_cross_correlation(  # pyright: ignore[reportUnknownVariableType]
        a,
        b,
        upsample_factor=1,
        reference_mask=ones,
        moving_mask=ones,
        overlap_ratio=_MIN_OVERLAP,
    )
    value = result[0] if isinstance(result, tuple) else result
    return np.asarray(value, dtype=float)


def _plausible_step(
    step: NDArray[np.float64], shape: tuple[int, int]
) -> bool:
    """Whether a displacement between neighbouring frames is small enough to be
    drift rather than a mismatched correlation peak."""
    return bool(
        abs(step[0]) <= _MAX_STEP * shape[0] and abs(step[1]) <= _MAX_STEP * shape[1]
    )


def _refine_shift(
    a: NDArray[np.float32],
    b: NDArray[np.float32],
    prediction: tuple[float, float] | NDArray[np.float64],
    upsample: int = 50,
    max_residual: float = 4.0,
    min_side: int = 24,
) -> tuple[NDArray[np.float64] | None, float]:
    """Sub-pixel displacement of b relative to a, close to `prediction`.

    The refinement runs on the overlapping part alone. Measuring on the whole
    frame instead leaves a bias of the order of a tenth of a pixel, because the
    non-overlapping margins contribute noise that pulls the correlation peak
    towards zero.

    Returns:
        The displacement and a confidence, which is the normalised correlation
        of the matched overlap. `(None, 0.0)` if the pair is unusable, either
        because the overlap is too small or because the measurement disagrees
        with the prediction by more than `max_residual`.
    """
    dy, dx = int(round(float(prediction[0]))), int(round(float(prediction[1])))
    a_crop, b_crop = _overlap(a, b, dy, dx)
    if a_crop.shape != b_crop.shape or min(a_crop.shape) < min_side:
        return None, 0.0

    window = _hann_window(a_crop.shape)
    residual, _, _ = phase_cross_correlation(  # pyright: ignore[reportUnknownVariableType]
        a_crop * window, b_crop * window, upsample_factor=upsample, normalization=None
    )
    if max(abs(residual[0]), abs(residual[1])) > max_residual:  # pyright: ignore[reportAny]
        return None, 0.0

    u = a_crop - a_crop.mean()
    v = b_crop - b_crop.mean()
    norm = np.sqrt((u * u).sum() * (v * v).sum())
    confidence = float((u * v).sum() / norm) if norm > 0 else 0.0
    if not np.isfinite(confidence) or confidence <= 0.05:
        return None, 0.0

    return np.array([dy + residual[0], dx + residual[1]], dtype=float), confidence


def _frame_pairs(
    num_frames: int,
    window: int,
    long_lags: tuple[int, ...],
    stride: int,
) -> list[tuple[int, int]]:
    """Every pair within `window` frames, plus a sparse set of distant pairs.

    The near pairs describe the fast frame-to-frame movement, the distant ones
    tie the slow drift down; without the latter the fit is again free to walk
    away over the length of the movie.
    """
    pairs = [
        (i, j)
        for i in range(num_frames)
        for j in range(i + 1, min(i + window + 1, num_frames))
    ]
    for lag in long_lags:
        pairs += [(i, i + lag) for i in range(0, num_frames - lag, stride)]
    return sorted(set(pairs))


def _solve_positions(
    pairs: list[tuple[int, int]],
    measured: NDArray[np.float64],
    confidence: NDArray[np.float64],
    num_frames: int,
    huber: float = 2.5,
    iterations: int = 5,
    scale_floor: float = 0.03,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Weighted least squares for all frame positions at once.

    Model `p_j - p_i = d_ij` for every measured pair, with the gauge
    `sum_i p_i = 0` to fix the free constant. Reweighted with a Huber rule, so
    that pairs spoiled by moving surface features lose their influence instead
    of dragging the whole path. The scale has a floor, so that a set of mutually
    consistent measurements is not over-rejected.
    """
    num_pairs = len(pairs)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for row, (i, j) in enumerate(pairs):
        rows += [row, row]
        cols += [i, j]
        vals += [-1.0, 1.0]
    rows += [num_pairs] * num_frames
    cols += list(range(num_frames))
    vals += [1.0] * num_frames
    matrix = sparse.csr_matrix(
        (vals, (rows, cols)), shape=(num_pairs + 1, num_frames)
    )

    weights = np.append(np.asarray(confidence, dtype=float), 10.0)
    positions = np.zeros((num_frames, 2))
    residual_norm = np.zeros(num_pairs)
    for _ in range(iterations):
        weighted = sparse.diags(weights) @ matrix
        residuals = np.empty((num_pairs, 2))
        for axis in range(2):
            rhs = np.append(measured[:, axis], 0.0)
            solution = lsqr(weighted, weights * rhs, atol=1e-12, btol=1e-12)[0]
            positions[:, axis] = solution
            residuals[:, axis] = matrix[:num_pairs] @ solution - measured[:, axis]
        residual_norm = np.linalg.norm(residuals, axis=1)
        deviation = np.median(np.abs(residual_norm - np.median(residual_norm)))
        scale = max(1.4826 * float(deviation), scale_floor)
        weights[:num_pairs] = confidence * np.minimum(
            1.0, huber * scale / np.maximum(residual_norm, 1e-9)
        )
        weights[num_pairs] = 10.0

    return positions, residual_norm, weights[:num_pairs]


def _updown_offset(path: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean offset between up frames and down frames, for the log only.

    Up and down frames of the same image are recorded with opposite y sweep, so
    a residual y phase shows up as an alternation of the path. The value depends
    on the frame rate and the image size, so it is measured rather than assumed.
    It needs no correcting: the fit gives every frame its own position anyway.
    """
    if len(path) < 3:
        return np.zeros(2)
    curvature = path[1:-1] - 0.5 * (path[:-2] + path[2:])
    parity = np.where(np.arange(1, len(path) - 1) % 2 == 0, 0.5, -0.5)
    return (curvature * parity[:, None]).mean(axis=0) * 2.0


@final
class Drift:
    """
    Initialise Drift class with Fast movie instance to then
    drift correct the movie data.

    Args:
        fast_movie: `FastMovie` instance.
        stepsize: Difference between frames that are correlated.
        corrspeed: Difference between two correlation windows.
        boxcar: Width of the boxcar filter that applied to the
            drift path. Set to 0 if no boxcar filter should be applied.
        median_filter: Parameter to decide if the drift path should be smoothed
            via a median filter of kernel size 3.
        subpixel: Apply the fractional part of the drift path by interpolation
            instead of rounding it away. Without it the correction is quantised
            to whole pixels, which leaves up to half a pixel per frame however
            accurate the path is. The pixel values become interpolated, so it is
            off by default.
    """

    def __init__(
        self,
        fast_movie: FastMovie,
        stepsize: int = 40,
        corrspeed: int = 1,
        boxcar: int = 50,
        median_filter: bool = True,
        subpixel: bool = False,
    ):
        if fast_movie.mode != DataMode.MOVIE:
            raise ValueError(f"`FastMovie` instance must be in mode {DataMode.MOVIE}")
        if fast_movie.channels is None:
            raise ValueError("`FastMovie.channels` must be set")

        self.data = fast_movie.data
        self.file = fast_movie.filename.replace(".h5", ".drift.txt")
        # self.processing_log = fast_movie.processing_log
        self.channels = fast_movie.channels
        self.stepsize = stepsize
        self.corrspeed = corrspeed
        self.boxcar = boxcar
        self.median_filter = median_filter
        self.subpixel = subpixel

        self.n_frames, self.img_height, self.img_width = self.data.shape
        self.transformations = np.zeros((2, self.n_frames), dtype=np.float32)
        self.integrated_trans: NDArray[np.float32] | None = None

        # if self.stepsize is None:
        #     self.stepsize = int(self.n_frames / 3)

    def correct_correlation(
        self,
        mode: DriftMode = DriftMode.FULL,
    ) -> DriftCorrectionResult:
        """Drift correction via cross correlation of two frames.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
        """
        self._get_drift_correlation()
        self._filter_drift()
        self._write_drift()

        assert self.integrated_trans is not None  # type assertion

        match mode:
            case DriftMode.FULL:
                return DriftCorrectionResult(
                    self._adjust_movie_buffered(),
                    self.transformations,
                    self.integrated_trans,
                )
            case DriftMode.COMMON:
                return DriftCorrectionResult(
                    self._adjust_movie_common(),
                    self.transformations,
                    self.integrated_trans,
                )

    def correct_phase_cross_correlation(
        self,
        mode: DriftMode = DriftMode.FULL,
    ) -> DriftCorrectionResult:
        """Drift correction via `scikit-image`'s
            [`phase_cross_correlation`][https://scikit-image.org/docs/0.23.x/api/skimage.registration.html#skimage.registration.phase_cross_correlation].

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
        """
        self._get_drift_phase_cross_correlation()
        self._filter_drift()
        self._write_drift()

        assert self.integrated_trans is not None  # type assertion

        match mode:
            case DriftMode.FULL:
                return DriftCorrectionResult(
                    self._adjust_movie_buffered(),
                    self.transformations,
                    self.integrated_trans,
                )
            case DriftMode.COMMON:
                return DriftCorrectionResult(
                    self._adjust_movie_common(),
                    self.transformations,
                    self.integrated_trans,
                )

    def correct_stackreg(
        self,
        mode: DriftMode = DriftMode.FULL,
        stackreg_reference: StackRegReferenceType = StackRegReferenceType.PREVIOUS,
    ) -> DriftCorrectionResult:
        """Drfit correction via [`pystackreg`][https://pystackreg.readthedocs.io/en/latest/].

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
        """
        self._get_drift_stackreg(stackreg_reference)
        self._filter_drift()
        self._write_drift()

        assert self.integrated_trans is not None  # type assertion

        match mode:
            case DriftMode.FULL:
                return DriftCorrectionResult(
                    self._adjust_movie_buffered(),
                    self.transformations,
                    self.integrated_trans,
                )
            case DriftMode.COMMON:
                return DriftCorrectionResult(
                    self._adjust_movie_common(),
                    self.transformations,
                    self.integrated_trans,
                )

    def correct_known(
        self,
        mode: DriftMode = DriftMode.FULL,
        known_drift_type: KnownDriftType = KnownDriftType.INTEGRATED,
    ) -> DriftCorrectionResult:
        """Drift correction from a known '.drift.txt' file.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
            known_drift_type: Whether to use values of integrated or sequential
                drift path.
        """
        driftfile = self.file.replace(".h5", ".drift.txt")

        match known_drift_type:
            case KnownDriftType.INTEGRATED:
                self.integrated_trans = np.loadtxt(driftfile, dtype=np.float32).T[
                    0:2, :
                ]
            case KnownDriftType.SEQUENTIAL:
                self.transformations = np.loadtxt(driftfile, dtype=np.float32).T[2:4, :]
                self.integrated_trans = np.cumsum(
                    self.transformations, axis=1, dtype=np.float32
                )
                self._write_drift()

        log.info(f"Known drift used: {known_drift_type}")

        match mode:
            case DriftMode.FULL:
                return DriftCorrectionResult(
                    self._adjust_movie_buffered(),
                    self.transformations,
                    self.integrated_trans,
                )
            case DriftMode.COMMON:
                return DriftCorrectionResult(
                    self._adjust_movie_common(),
                    self.transformations,
                    self.integrated_trans,
                )

    def correct_global(self, mode: DriftMode = DriftMode.FULL) -> DriftCorrectionResult:
        """Drift correction by a global fit over many redundant frame pairs.

        Unlike the sequential methods this does not integrate a per-frame
        displacement, so its error does not accumulate over the movie. The path
        is used as it comes out of the fit; the boxcar and median settings do
        not apply, because there is no integrated noise left for them to hide.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
        """
        self._get_drift_global()
        self._write_drift()

        assert self.integrated_trans is not None  # type assertion

        match mode:
            case DriftMode.FULL:
                data = self._adjust_movie_buffered()
            case DriftMode.COMMON:
                data = self._adjust_movie_common()

        return DriftCorrectionResult(
            data, self.transformations, self.integrated_trans
        )

    def _get_drift_global(
        self,
        window: int = 8,
        long_lags: tuple[int, ...] = (25, 50, 100, 200),
        stride: int = 5,
        upsample: int = 50,
        rounds: int = 2,
    ) -> None:
        """Determine the drift path by a global fit over redundant frame pairs.

        Three stages: a sequential chain to have a starting point, then the
        redundant pairs measured around the prediction that the current path
        gives, then a robust least squares over all of them. The prediction
        matters: measured blind, distant pairs pick the wrong correlation peak
        on a surface with repeating features.

        Falls back to the sequential chain if too few pairs can be measured, so
        that a difficult movie in a batch run yields a usable path instead of an
        exception.
        """
        num_frames = len(self.data)
        prepared = np.stack([_bandpass(frame) for frame in self.data])

        # Stage one: neighbouring frames only.
        shape = self.data.shape[1:]
        path = np.zeros((num_frames, 2))
        implausible = 0
        for i in TqdmLogger(range(1, num_frames), desc="Drift: frame chain"):
            prediction = _coarse_shift(prepared[i - 1], prepared[i])
            measured, _ = _refine_shift(prepared[i - 1], prepared[i], prediction)
            step = prediction if measured is None else measured
            if not _plausible_step(step, shape):
                implausible += 1
                step = np.zeros(2)
            path[i] = path[i - 1] + step

        if implausible:
            log.info(
                "Global drift: %d of %d frame steps looked implausible and were "
                "left to the global fit.",
                implausible,
                num_frames - 1,
            )

        pairs = _frame_pairs(num_frames, window, long_lags, stride)
        for round_index in range(rounds):
            kept: list[tuple[int, int]] = []
            measurements: list[NDArray[np.float64]] = []
            confidences: list[float] = []
            for i, j in TqdmLogger(pairs, desc="Drift: frame pairs"):
                measured, confidence = _refine_shift(
                    prepared[i], prepared[j], path[j] - path[i], upsample
                )
                if measured is not None:
                    kept.append((i, j))
                    measurements.append(measured)
                    confidences.append(confidence)

            if len(kept) < 2 * num_frames:
                log.warning(
                    "Global drift: only %d of %d frame pairs could be measured; "
                    "keeping the sequential path.",
                    len(kept),
                    len(pairs),
                )
                break

            previous = path
            path, residuals, weights = _solve_positions(
                kept, np.asarray(measurements), np.asarray(confidences), num_frames
            )
            rejected = float((weights < 0.5 * np.asarray(confidences)).mean())
            offset = _updown_offset(path)
            log.info(
                "Global drift round %d: %d of %d pairs used, "
                "pair residual RMS %.3f px, %.0f%% down-weighted, "
                "up/down offset (%.3f, %.3f) px",
                round_index,
                len(kept),
                len(pairs),
                float(np.sqrt((residuals**2).mean())),
                rejected * 100.0,
                offset[0],
                offset[1],
            )
            moved = np.abs(
                (path - path.mean(axis=0)) - (previous - previous.mean(axis=0))
            ).max()
            if round_index and moved < 0.01:
                break

        path = path - path[0]
        self.integrated_trans = path.T.astype(np.float32)
        self.transformations = np.diff(
            path, axis=0, prepend=path[:1]
        ).T.astype(np.float32)

    def _get_drift_correlation(self) -> None:
        """Calculation of the drift path by FFT cross correlation of two frames."""
        data = self.data.copy()
        num_frames, num_y_pixels, num_x_pixels = data.shape

        for i in TqdmLogger(range(num_frames), desc="Calculating drift path"):
            try:
                correlated = correlate(
                    data[self.corrspeed * i, :, :],
                    data[self.corrspeed * i + self.stepsize, :, :],
                    method="fft",
                )
                maxind = np.argmax(correlated)
                indices = np.unravel_index(maxind, correlated.shape)  # pyright: ignore[reportAny]
                effektive_shift = np.asarray(
                    [
                        [(-(num_y_pixels - 1) + indices[0]) / self.stepsize],
                        [(indices[1] - (num_x_pixels - 1)) / self.stepsize],
                    ]
                )
                self.transformations[:, i] = effektive_shift.T
            except Exception:
                pass
        log.info(f"Last found correlation indices are {indices}")

    def _get_drift_phase_cross_correlation(self) -> None:
        """Calculation of the drift by phase cross correlation of two frames."""
        data = self.data.copy()
        num_frames, _, _ = data.shape

        for i in TqdmLogger(range(num_frames), desc="Calculating drift path"):
            try:
                shift, _, _ = skimage.registration.phase_cross_correlation(  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
                    data[self.corrspeed * i, :, :],
                    data[self.corrspeed * i + self.stepsize, :, :],
                    upsample_factor=10,
                )

                effective_shift = np.asarray(  # pyright: ignore[reportUnknownVariableType]
                    [  # pyright: ignore[reportUnknownArgumentType]
                        [shift[0] / self.stepsize],
                        [shift[1] / self.stepsize],
                    ]
                )
                self.transformations[:, i] = effective_shift.T  # pyright: ignore[reportUnknownMemberType]
            except Exception:
                pass

    def _get_drift_stackreg(self, reference: StackRegReferenceType) -> None:
        """Calculation of the drift by pystackreg."""
        stackreg = StackReg(StackReg.TRANSLATION)
        transformation_matrices = stackreg.register_stack(  # pyright: ignore[reportUnknownMemberType]
            self.data, reference=reference.value
        )
        x_path_integrated: list[NDArray[np.float32]] = []
        y_path_integrated: list[NDArray[np.float32]] = []
        for matrix in transformation_matrices:  # pyright: ignore[reportAny]
            x_path_integrated.append(-matrix[0, 2])  # pyright: ignore[reportAny]
            y_path_integrated.append(-matrix[1, 2])  # pyright: ignore[reportAny]

        # self.integrated_trans = np.array([y_path_integrated, x_path_integrated])
        x_path = np.array(x_path_integrated)
        x_path = np.diff(x_path, prepend=0)
        y_path = np.array(y_path_integrated)
        y_path = np.diff(y_path, prepend=0)

        self.transformations = np.stack((y_path, x_path))
        self.stepsize = 1

    def _filter_drift(self) -> None:
        """Smoothing of drift path by median filter and/or boxcar filter."""
        boxwidth = self.boxcar
        boxcar = np.ones((1, boxwidth)) / boxwidth
        boxcar = boxcar[0, :]

        if self.median_filter:
            self.transformations[0, :] = medfilt(self.transformations[0, :], 3)
            self.transformations[1, :] = medfilt(self.transformations[1, :], 3)

        self.integrated_trans = np.cumsum(
            self.transformations, axis=1, dtype=np.float32
        )
        # linear extrapolation
        pos = np.linspace(0, self.n_frames - 1, self.n_frames)
        k1, d1 = np.polyfit(  # pyright: ignore[reportAny]
            pos[: -self.stepsize], self.integrated_trans[0, : -self.stepsize], 1
        )
        k2, d2 = np.polyfit(  # pyright: ignore[reportAny]
            pos[: -self.stepsize], self.integrated_trans[1, : -self.stepsize], 1
        )
        self.integrated_trans[0, -self.stepsize :] = d1 + k1 * pos[-self.stepsize :]
        self.integrated_trans[1, -self.stepsize :] = d2 + k2 * pos[-self.stepsize :]

        if self.boxcar != 0:
            transformations_conv = np.zeros((2, self.n_frames), dtype=np.float32)
            transformations_conv[0, :] = convolve(self.integrated_trans[0], boxcar)  # pyright: ignore[reportAny]
            transformations_conv[1, :] = convolve(self.integrated_trans[1], boxcar)  # pyright: ignore[reportAny]
            self.integrated_trans = transformations_conv

            log.info(f"Boxcar filter used with boxsize: {boxwidth}")

    def _write_drift(self):
        """Write a drift.txt file to disc."""
        if self.integrated_trans is None:
            raise ValueError("Drift path not determined yet.")

        with open(self.file, "w") as fileobject:
            _ = fileobject.write(
                "# {0:>10}   {1:>12}  {2:>12}  {3:>12} \n".format(
                    "y integrated", "x integrated", "y sequential", "x sequential"
                )
            )
            for i in range(self.transformations.shape[1]):
                _ = fileobject.write(
                    "{0:>14.5f}   {1:>12.5f}  {2:>12.5f}  {3:>12.5f} \n".format(
                        self.integrated_trans[0, i],  # pyright: ignore[reportAny]
                        self.integrated_trans[1, i],  # pyright: ignore[reportAny]
                        self.transformations[0, i],  # pyright: ignore[reportAny]
                        self.transformations[1, i],  # pyright: ignore[reportAny]
                    )
                )

    def _adjust_movie_buffered(self):
        """Embed movie frames into a buffered background so they can move freely
        according to the integrated drift path.
        """
        assert self.integrated_trans is not None  # type assertion

        y_translations = self.integrated_trans[0]
        x_translations = self.integrated_trans[1]

        y_min, y_max = np.min(y_translations), np.max(y_translations)
        x_min, x_max = np.min(x_translations), np.max(x_translations)

        y_padding = int(np.ceil(y_max - y_min))
        x_padding = int(np.ceil(x_max - x_min))

        corr_movie = np.zeros(
            (
                self.n_frames,
                self.img_height + y_padding,
                self.img_width + x_padding,
            ),
            dtype=np.float32,
        )

        base_y = int(np.round(-y_min))
        base_x = int(np.round(-x_min))

        for i in range(self.n_frames):
            y_shift = int(np.round(y_translations[i]))
            x_shift = int(np.round(x_translations[i]))

            y_start = base_y + y_shift
            y_end = y_start + self.img_height
            x_start = base_x + x_shift
            x_end = x_start + self.img_width

            frame = self.data[i]
            if self.subpixel:
                # Whole pixels by placement, the remainder by interpolation.
                # Done frame by frame, so no second copy of the movie is held.
                frame = ndshift(
                    frame,
                    (
                        y_translations[i] - y_shift,
                        x_translations[i] - x_shift,
                    ),
                    order=3,
                    mode="nearest",
                )

            corr_movie[i, y_start:y_end, x_start:x_end] = frame

        log.info("Drift correction finished")

        return corr_movie

    def _adjust_movie_common(self):
        """Apply drift correction and return only the region common to all frames."""
        buffered = self._adjust_movie_buffered()

        valid = buffered != 0
        common_mask = np.all(valid, axis=0)

        ys, xs = np.where(common_mask)
        if ys.size == 0 or xs.size == 0:
            raise ValueError("No common area exists")

        y_start, y_end = ys.min(), ys.max() + 1
        x_start, x_end = xs.min(), xs.max() + 1

        corr_movie = buffered[:, y_start:y_end, x_start:x_end]

        return corr_movie


# def meanfilter(data, kernel=3):
#     """
#     possible meanfilter.

#     Args:
#         data: 1D array
#         kernel: Size of values that are filtered

#     Returns:
#         filtered: adjusted array
#     """
#     filtered = np.zeros(len(data))
#     if kernel % 2 == 0:
#         kernel += 1
#     for i in range(len(data)):
#         down = i - int(kernel / 2)
#         up = i + int(kernel / 2) + 1
#         if down < 0:
#             down = 0
#         if up > len(data):
#             up = int(len(data))
#         filtered[i] = np.mean(data[down:up])
#     return filtered
