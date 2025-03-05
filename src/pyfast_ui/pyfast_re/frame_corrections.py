from typing import Callable, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import convolve2d


AlignFunc: TypeAlias = Callable[[NDArray[np.float32]], NDArray[np.float32]]


def align_rows(
    frame: NDArray[np.float32], align_type: Literal["median", "mean", "poly2", "poly3"]
) -> NDArray[np.float32]:
    """Align the rows of a frame.

    Args:
        frame: The frame of which the rows will be aligned.
        align_type: Correction algorithm for row alignment.

    Returns:
        The background that needs to be substracted from the image for row alignment.
    """
    match align_type:
        case "median":
            func: AlignFunc = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731
        case "mean":
            func = lambda row: np.full(row.shape[0], np.median(row))  # noqa: E731
        case "poly2":
            func = lambda row: _poly_background(row, 2)  # noqa: E731
        case "poly3":
            func = lambda row: _poly_background(row, 3)  # noqa: E731

    background = np.apply_along_axis(
        func,
        axis=1,
        arr=frame,
    )

    return background


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
