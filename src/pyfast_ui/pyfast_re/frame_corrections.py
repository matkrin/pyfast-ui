from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, cast
from scipy import sparse
from scipy.signal import convolve2d
from scipy.sparse.linalg import lsqr


AlignFunc = Callable[[NDArray[np.float32]], NDArray[np.float32]]

MIN_SELECTED = 3
"""Fewest selected pixels a row needs before its own fit is trusted. Below that
the row falls back to all of its pixels, which is better than a wild
extrapolation from two points."""

MULTILAG_LAGS = (1, 2, 3, 5, 8, 13)
"""Row or frame separations compared by the multi-lag estimators. A single
separation has to be integrated, and integrating its errors is a random walk;
several of them overdetermine the offsets so the errors average instead."""

MIN_SPAN = 0.25
"""Fraction of the row width the selected pixels must cover before a polynomial
is fitted to them alone. The row coordinate runs from -0.5 to 0.5."""


def selection_mask(
    frame: NDArray[np.float32],
    area_selection: str = "whole image",
    roi: tuple[int, int, int, int] | None = None,
) -> NDArray[np.bool_]:
    """Which pixels a background fit is allowed to use.

    Restricting the fit matters when the frame is not mostly background: a
    terrace covering half the image drags a plane or a row median towards
    itself, and the correction then tilts the flat parts instead of levelling
    them. The height based selections pick a band of the height histogram of
    that frame, so they follow the data rather than a fixed geometry.

    Args:
        frame: The frame the fit will run on.
        area_selection: One of "whole image", "lowest 50%", "middle 50%",
            "highest 50%", "exclude ROI".
        roi: Region to leave out, as (y_start, y_end, x_start, x_end). Only used
            for "exclude ROI".

    Returns:
        Boolean mask, True where the fit may use the pixel.
    """
    mask = np.ones(frame.shape, dtype=bool)

    match area_selection:
        case "lowest 50%":
            bounds = (0.0, 50.0)
        case "middle 50%":
            bounds = (25.0, 75.0)
        case "highest 50%":
            bounds = (50.0, 100.0)
        case "exclude ROI":
            if roi is not None:
                y_start, y_end, x_start, x_end = roi
                mask[y_start:y_end, x_start:x_end] = False
            return mask
        case _:
            return mask

    low, high = np.percentile(frame, bounds)
    return (frame >= low) & (frame <= high)


def _solve_offsets(
    count: int,
    pairs: list[tuple[int, int]],
    measured: list[float],
) -> NDArray[np.float64]:
    """Offsets that best explain the measured differences, gauged to zero mean.

    Least squares for o_j - o_i = d_ij over every measured pair. With one
    separation this reduces to a cumulative sum; with several it is
    overdetermined, which is the point.
    """
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for index, (i, j) in enumerate(pairs):
        rows += [index, index]
        cols += [i, j]
        vals += [-1.0, 1.0]

    gauge = len(pairs)
    rows += [gauge] * count
    cols += list(range(count))
    vals += [1.0] * count

    matrix = sparse.csr_matrix((vals, (rows, cols)), shape=(gauge + 1, count))
    right = np.append(np.asarray(measured, dtype=float), 0.0)

    return lsqr(matrix, right, atol=1e-10, btol=1e-10)[0]


def frame_level_offsets(
    movie: NDArray[np.float32], lags: tuple[int, ...] = MULTILAG_LAGS
) -> NDArray[np.float64]:
    """One height offset per frame, so that the frames sit at a common level.

    Every background correction fixes the level of a frame by some convention of
    its own, and "median of diff" fixes it by nothing at all: its offsets start
    at zero for the first row, so the whole frame hangs on the level of that one
    noisy row. The frames then pump up and down against each other.

    The level differences are measured as the median of the pixelwise difference
    between two frames, which a minority of the surface changing does not move,
    and solved over several frame separations so the estimate does not drift over
    a long movie.
    """
    count = len(movie)
    pairs: list[tuple[int, int]] = []
    measured: list[float] = []
    for lag in lags:
        if lag >= count:
            continue
        for index in range(count - lag):
            pairs.append((index, index + lag))
            measured.append(float(np.median(movie[index + lag] - movie[index])))

    if not pairs:
        return np.zeros(count)

    return _solve_offsets(count, pairs, measured)


def _row_background(
    row: NDArray[np.float32],
    selected: NDArray[np.bool_],
    align_type: str,
) -> NDArray[np.float32]:
    """Background of one row, fitted on the selected pixels only."""
    positions = np.linspace(-0.5, 0.5, row.shape[0]).astype(np.float32)
    degree = {"line": 1, "poly2": 2, "poly3": 3}.get(align_type, 0)

    if not _is_usable(positions, selected, degree):
        # Too few pixels, or all of them crowded into one part of the row. A
        # polynomial fitted to those extrapolates wildly over the rest, so the
        # whole row is used instead.
        selected = np.ones_like(selected)

    values = row[selected]
    match align_type:
        case "mean":
            return np.full(row.shape[0], values.mean(), dtype=np.float32)
        case "line" | "poly2" | "poly3":
            coeffs = np.polyfit(positions[selected], values, degree)
            return np.polyval(coeffs, positions).astype(np.float32)
        case _:
            return np.full(row.shape[0], np.median(values), dtype=np.float32)


def _is_usable(
    positions: NDArray[np.float32], selected: NDArray[np.bool_], degree: int
) -> bool:
    """Whether a fit of this degree can be trusted on the selected pixels."""
    count = int(selected.sum())
    if count < max(MIN_SELECTED, degree + 2):
        return False
    if degree == 0:
        return True

    chosen = positions[selected]
    span = float(chosen.max() - chosen.min())

    return span >= MIN_SPAN


def align_rows(
    frame: NDArray[np.float32],
    align_type: str = "median",
    mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float32]:
    """Background of a frame from a fit to each row separately.

    Args:
        frame: The frame used for background fitting.
        align_type: "median", "median of diff", "mean", "line", "poly2" or
            "poly3".
        mask: Pixels the fit may use, see `selection_mask`. `None` uses all.

    Returns:
        The background to subtract.
    """
    if mask is None:
        mask = np.ones(frame.shape, dtype=bool)

    # "median of diff" couples neighbouring rows and cannot be expressed as a
    # function of a single row.
    if align_type.startswith("median of diff"):
        lags = MULTILAG_LAGS if "multi-lag" in align_type else (1,)
        return _median_of_diff_background(frame, mask, lags)

    background = np.empty(frame.shape, dtype=np.float32)
    for index in range(frame.shape[0]):
        background[index] = _row_background(frame[index], mask[index], align_type)

    return cast(NDArray[np.float32], background)


def _median_of_diff_background(
    frame: NDArray[np.float32],
    mask: NDArray[np.bool_] | None = None,
    lags: tuple[int, ...] = (1,),
) -> NDArray[np.float32]:
    """Row offsets that drive the median of the vertical neighbour differences
    to zero.

    For each pair of neighbouring rows the median is taken over the pointwise
    differences, and those steps are accumulated into an offset per row. That is
    what makes the method robust: a feature covering less than half of a row
    leaves the median of the differences untouched, so large objects survive
    instead of being levelled away, which is the reason to prefer this over the
    plain row median.

    Taking the difference of the two row medians instead, as an earlier version
    did, is not the same quantity. It loses the robustness, and it is identically
    zero once the row medians have been removed, so a second correction appeared
    to do nothing.
    """
    num_rows = frame.shape[0]
    if mask is None:
        mask = np.ones(frame.shape, dtype=bool)

    pairs: list[tuple[int, int]] = []
    measured: list[float] = []
    for lag in lags:
        if lag >= num_rows:
            continue
        differences = frame[lag:] - frame[:-lag]
        # A difference is only meaningful where both of its rows are selected.
        usable = mask[lag:] & mask[:-lag]
        for index in range(num_rows - lag):
            chosen = usable[index]
            row = differences[index]
            pairs.append((index, index + lag))
            measured.append(
                float(np.median(row[chosen] if chosen.sum() >= MIN_SELECTED else row))
            )

    if not pairs:
        return np.zeros(frame.shape, dtype=np.float32)

    offsets = _solve_offsets(num_rows, pairs, measured).astype(np.float32)

    return np.repeat(offsets[:, None], frame.shape[1], axis=1)


def level_plane(
    frame: NDArray[np.float32], mask: NDArray[np.bool_] | None = None
) -> NDArray[np.float32]:
    """Fitting of a plane through the pixel intensities of a frame.

    A least squares fit of z = a + b x + c y over the selected pixels. On a full
    rectangular grid this is the same plane that ramps fitted to the two mean
    profiles give, but unlike those it also works when only part of the frame is
    selected.

    Args:
        frame: The frame used for background fitting.
        mask: Pixels the fit may use, see `selection_mask`. `None` uses all.

    Returns:
        The background that needs to be substracted from the image for plane leveling.
    """
    num_rows, num_cols = frame.shape
    y_positions = np.linspace(-0.5, 0.5, num_rows, dtype=np.float64)
    x_positions = np.linspace(-0.5, 0.5, num_cols, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(x_positions, y_positions)

    if mask is None or mask.sum() < 3:
        mask = np.ones(frame.shape, dtype=bool)

    design = np.column_stack(
        [np.ones(int(mask.sum())), grid_x[mask], grid_y[mask]]
    )
    coeffs, *_ = np.linalg.lstsq(design, frame[mask].astype(np.float64), rcond=None)
    background = coeffs[0] + coeffs[1] * grid_x + coeffs[2] * grid_y

    return cast(NDArray[np.float32], background.astype(np.float32))


def convolve_frame(
    frame: NDArray[np.float32], matrix: ArrayLike
) -> NDArray[np.float32]:
    """Convolution of a frame with matrix.

    Args:
        frame: The frame to be convolved with `matrix`.
        matrix: The convolving matrix.

    Returns:
        The convolution result of `frame` and `matrix`.
    """
    matrix = np.array(matrix)
    len_y = int((matrix.shape[0] - 1) / 2)
    len_x = int((matrix.shape[1] - 1) / 2)

    convolved = cast(NDArray[np.float32], convolve2d(frame, matrix, boundary="symm"))

    if len_y == 0:
        return convolved[:, len_x:-len_x]
    elif len_x == 0:
        return convolved[len_y:-len_y, :]
    else:
        return convolved[len_y:-len_y, len_x:-len_x]


def _poly_background(
    line: NDArray[np.float32], polynomial_degree: int
) -> NDArray[np.float32]:
    x = np.linspace(-0.5, 0.5, line.shape[0]).astype(np.float32)
    coeffs = np.polyfit(x, line, polynomial_degree)
    return np.polyval(coeffs, x)
