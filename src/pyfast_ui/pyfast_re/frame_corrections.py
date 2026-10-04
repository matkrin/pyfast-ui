from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, cast
from scipy.signal import convolve2d


AlignFunc = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def align_rows(
    frame: NDArray[np.float32],
    align_type: str = "median",
) -> NDArray[np.float32]:
    """
    Align rows of a 2D frame based on the selected method.
    """

    # "median of diff" couples neighbouring rows and cannot be expressed as a
    # function of a single row.
    if align_type == "median of diff":
        return _median_of_diff_background(frame)

    match align_type:
        case "median":
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731
        case "mean":
            func: AlignFunc = lambda row: np.full(row.shape[0], np.mean(row))  # noqa: E731
        case "poly2":
            func: AlignFunc = lambda row: _poly_background(row, 2)  # noqa: E731
        case "poly3":
            func: AlignFunc = lambda row: _poly_background(row, 3)  # noqa: E731
        case _:
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731

    background = np.apply_along_axis(func, axis=1, arr=frame)
    return cast(NDArray[np.float32], background)


def _median_of_diff_background(frame: NDArray[np.float32]) -> NDArray[np.float32]:
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
    offsets = np.zeros(num_rows, dtype=np.float32)
    if num_rows > 1:
        steps = np.median(frame[1:] - frame[:-1], axis=1)
        offsets[1:] = np.cumsum(steps)

    return np.repeat(offsets[:, None], frame.shape[1], axis=1)


def level_plane(frame: NDArray[np.float32]) -> NDArray[np.float32]:
    """Fitting of a plane through the pixel intensities of a frame.

    Args:
        frame: The frame used for background fitting.

    Returns:
        The background that needs to be substracted from the image for plane leveling.
    """
    # One ramp per direction, from the profile averaged over the other one.
    background_x = _poly_background(frame.mean(axis=0), 1)  # pyright: ignore[reportAny]
    background_y = _poly_background(frame.mean(axis=1), 1)  # pyright: ignore[reportAny]

    # The x ramp varies along the columns and the y ramp along the rows, so they
    # are broadcast along different axes. Writing both along axis 1, as an
    # earlier version did, put the y ramp across the image and produced an array
    # of shape (rows, rows), which is silently wrong on a square frame and
    # raises on any other.
    background = background_x[None, :] + background_y[:, None]

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
