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

    match align_type:
        case "median":
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731
        case "median of diff":
            # placeholder (actual implementation handled before apply_along_axis)
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731
        case "mean":
            func: AlignFunc = lambda row: np.full(row.shape[0], np.mean(row))  # noqa: E731
        case "poly2":
            func: AlignFunc = lambda row: _poly_background(row, 2)  # noqa: E731
        case "poly3":
            func: AlignFunc = lambda row: _poly_background(row, 3)  # noqa: E731
        case _:
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731

    # Special handling: "median of diff" depends on adjacent rows, not just a single row.
    if align_type == "median of diff":
        med = np.median(frame, axis=1).astype(np.float32)
        diff = np.zeros_like(med, dtype=np.float32)
        if med.shape[0] > 1:
            diff[1:] = med[1:] - med[:-1]
        background = np.repeat(diff[:, None], frame.shape[1], axis=1)
        return cast(NDArray[np.float32], background)

    background = np.apply_along_axis(func, axis=1, arr=frame)
    return cast(NDArray[np.float32], background)


def level_plane(frame: NDArray[np.float32]) -> NDArray[np.float32]:
    """Fitting of a plane through the pixel intensities of a frame.

    Args:
        frame: The frame used for background fitting.

    Returns:
        The background that needs to be substracted from the image for plane leveling.
    """
    background_x = _poly_background(frame.mean(axis=0), 1)  # pyright: ignore[reportAny]
    background_y = _poly_background(frame.mean(axis=1), 1)  # pyright: ignore[reportAny]

    background_xx = np.apply_along_axis(
        lambda _: background_x,
        axis=1,
        arr=frame,
    )
    background_yy = np.apply_along_axis(lambda _: background_y, axis=1, arr=frame)
    background = background_xx + background_yy

    return background


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
