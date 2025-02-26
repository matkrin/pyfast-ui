"""Exact interpolation for interlacing"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, lil_matrix
from scipy.spatial import Delaunay
from tqdm import tqdm

from pyfast_ui.pyfast_re.data_mode import DataMode

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie

log = logging.getLogger(__name__)


@dataclass
class InterpolationResult:
    interpolation_matrix_up: csr_matrix
    interpolation_matrix_down: csr_matrix
    x_coords_measured: NDArray[np.float32]
    y_coords_measured: NDArray[np.float32]
    x_coords_target: NDArray[np.float32]
    y_coords_target: NDArray[np.float32]


def _output_y_grid(num_y_pixels: int, num_x_pixels: int):
    """Returns the y grid of the data points; not corrected for probe creep.

    Args:
        num_y_pixels: Number of pixels in y direction.
        num_x_pixels: Number of pixels in x direction.

    Returns:
        y_meshgrid:  y meshgrid of measured datapoints.
        y_grid_1d: equidistant 1D grid; contains the centers of each line in y.

    """

    # equidistant grid (1D array)
    y_grid_1d = np.linspace(-(num_y_pixels / 2), num_y_pixels / 2, num_y_pixels)

    # meshgrid of measured datapoints
    y_meshgrid = np.linspace(
        -(num_y_pixels / 2), num_y_pixels / 2, num_y_pixels * num_x_pixels
    ).reshape(num_y_pixels, num_x_pixels)

    for i in range(num_y_pixels):
        y_meshgrid[i, :] = y_meshgrid[i, :: (-1) ** i]

    return y_meshgrid, y_grid_1d


def _output_x_grid(num_y_pixels: int, num_x_pixels: int):
    """Returns the x grid of the data points.

    Args:
        num_y_pixels: Number of pixels in y direction.
        num_x_pixels: Number of pixels in x direction.

    Returns:
        x_meshgrid:  x meshgrid of measured datapoints.
        x_grid_1d: equidistant 1D grid.

    """

    # equidistant grid (1D array)
    x_grid_1d = np.linspace(-(num_x_pixels / 2), num_x_pixels / 2, num_x_pixels)

    # hysteresis in x direction
    x_hysteresis = (
        num_x_pixels
        / 2.0
        * np.sin(x_grid_1d * np.pi / (num_x_pixels + abs(x_grid_1d[0] - x_grid_1d[1])))
    )

    # meshgrid of measured datapoints
    x_meshgrid = np.array([x_hysteresis for _ in range(num_y_pixels)])

    return x_meshgrid, x_grid_1d


def get_interpolation_matrix(
    points_to_triangulate: list[tuple[float, float]],
    grid_points: list[tuple[float, float]],
):
    """
    Creates matrix containing all the relevant information for interpolating
    values measured at the same relative postion to a grid.

    The matix is constructed in a sparse format to limit memory usage.
    Construction is done in the lil sparse format, which is late converted to csr format for
    faster matrix vector dot procut.

    The matrix is of the form  (number of grid points at which to interpolate the frame) x (number of measured points within one frame).
    For this reason the number of interpolation and measured points do not have to be the same.

    Frames are interpolated within one matrix vecor dot product. For this reason the frame arrays
    need to be flattened (vectorized) before multiplication.

    Args:
        points_to_triangulate: list of tuples! represnting the points of measurement
        grid_points: list of tuples! represnting the grid points at which to interpolate

    The input formatting is slightly unusuall if you are used to numpy etc.
    Here we do not give two list, each containing values for one dimension.
    Insted we give tuples of values for each point i.e. (x,y) - See QHull documentation.

    Steps:
        1) Perform delaunay triangulation on grid of measurement positions.
        2) Find dealaunay triangles which contain new grid points.
        3) For each triangle get indices of corner poinst.
        4) At corresponding point in matrix insert barycentric coordinates.
    """

    triangulation = Delaunay(points_to_triangulate)
    triangles_containing_gridpoints = triangulation.find_simplex(grid_points)
    interpolation_matrix = lil_matrix((len(grid_points), len(points_to_triangulate)))

    for i in tqdm(
        range(len(grid_points)), desc="Building interpolation matrix", unit="lines"
    ):
        triangle_corners = triangulation.simplices[triangles_containing_gridpoints[i]]
        barycentric_coords = triangulation.transform[
            triangles_containing_gridpoints[i], :2
        ].dot(
            grid_points[i]
            - triangulation.transform[triangles_containing_gridpoints[i], 2]
        )
        barycentric_coords = np.append(
            barycentric_coords, 1 - np.sum(barycentric_coords)
        )

        if triangles_containing_gridpoints[i] == -1:
            for j in range(3):
                interpolation_matrix[i, triangle_corners[j]] = np.nan
        else:
            for j in range(3):
                interpolation_matrix[i, triangle_corners[j]] = barycentric_coords[j]

    interpolation_matrix = csr_matrix(interpolation_matrix)
    return interpolation_matrix


def determine_interpolation(
    fast_movie: FastMovie,
    offset: float = 0.0,
    grid=None,
    # image_range=None,
    # interpolation_matrix_up=None,
    # interpolation_matrix_down=None,
    # give_grid=False,
):
    """Interpolates the pixels in a FAST movie using the analytic positions of the probe.
    Currently only available for interlaced movies.

    Args:
        fast_movie: FastMovie object
        offset: y offset of interpolation points; defaults to 0.0
        grid: precomputed grid of the actual STM tip movement,
            this supersedes the in place calculation of Bezier curves.
        image_range: range of images to be interpolated.
        interpolation_matrix_up: precomputed interpolation matrix for up frames.
        interpolation_matrix_up: precomputed interpolation matrix for down frames. This prevents
            the matrix from beeing constructed multiple times.
        give_grid: if this option is set to True, the function returns
            the interpolation matrix directly after its construction.

    Returns:
        nothing
    """

    if fast_movie.mode != DataMode.MOVIE:
        raise ValueError("you must first reshape your data in movie mode.")
    if fast_movie.channels is None:
        raise ValueError("`FastMovie.channels` must be set")

    num_x_pixels = fast_movie.data.shape[2]
    num_y_pixels = fast_movie.data.shape[1]

    # meshgrids
    if fast_movie.channels.is_interlaced():
        # Computing only the grids which are actually need might be more efficient, but this does not seem to be a bottleneck
        y_coords_measured, y_grid_1d = _output_y_grid(num_y_pixels, num_x_pixels)

        # Leaving the general structure this way to make it easier to adapt to other types of grids
        x_coords_measured, x_grid_1d = _output_x_grid(num_y_pixels, num_x_pixels)
    else:
        # Computing only the grids which are actually need might be more efficient, but this does not seem to be a bottleneck
        y_coords_measured, y_grid_1d = _output_y_grid(2 * num_y_pixels, num_x_pixels)
        x_coords_measured, x_grid_1d = _output_x_grid(2 * num_y_pixels, num_x_pixels)

    # correct creep in y direction
    if grid is None:
        y_up = y_coords_measured
        y_down = y_coords_measured[:, ::-1]
    else:
        y_up, y_down = grid

    x_coords_target, y_coords_target = np.meshgrid(x_grid_1d, y_grid_1d)

    y_coords_target += offset

    if fast_movie.channels.is_forward():
        y_coords_target = y_coords_target[0::2].copy()
        # y_target_down = y_coords_target[0::2].copy()
        x_coords_target = x_coords_target[0::2].copy()
        y_up = y_up[0::2].copy()
        y_down = y_down[1::2].copy()
        x_coords_measured = x_coords_measured[0::2].copy()
        y_coords_measured = y_coords_measured[0::2].copy()

    elif fast_movie.channels.is_backward():
        y_coords_target = y_coords_target[1::2].copy()
        # y_target_down = y_coords_target[1::2].copy()
        x_coords_target = x_coords_target[1::2].copy()
        y_up = y_up[1::2].copy()
        y_down = y_down[0::2].copy()
        x_coords_measured = x_coords_measured[1::2].copy()
        y_coords_measured = y_coords_measured[1::2].copy()

    points_up = list(zip(y_up.flatten(), x_coords_measured.flatten()))
    points_down = list(zip(y_down.flatten(), x_coords_measured.flatten()))
    grid_points_up = list(zip(y_coords_target.flatten(), x_coords_target.flatten()))
    grid_points_down = list(zip(y_coords_target.flatten(), x_coords_target.flatten()))

    interpolation_matrix_up = get_interpolation_matrix(points_up, grid_points_up)
    interpolation_matrix_down = get_interpolation_matrix(points_down, grid_points_down)

    # return interpolation_matrix_up, interpolation_matrix_down
    return InterpolationResult(
        interpolation_matrix_up=interpolation_matrix_up,
        interpolation_matrix_down=interpolation_matrix_down,
        x_coords_measured=x_coords_measured,
        y_coords_measured=y_coords_measured,
        x_coords_target=x_coords_target,
        y_coords_target=y_coords_target,
    )


def apply_interpolation(
    fast_movie: FastMovie,
    interpolation_matrix_up: csr_matrix,
    interpolation_matrix_down: csr_matrix,
):
    num_x_pixels = fast_movie.data.shape[2]
    num_y_pixels = fast_movie.data.shape[1]
    assert fast_movie.channels is not None  # type assertion

    for i in range(fast_movie.data.shape[0]):
        is_up_frame = False
        if fast_movie.channels.is_up_and_down() and i % 2 == 0:
            is_up_frame = True

        if fast_movie.channels.is_up_not_down():
            is_up_frame = True

        frame_flattened: NDArray[np.float32] = fast_movie.data[i].flatten()  # pyright: ignore[reportAny]
        if is_up_frame:
            fast_movie.data[i] = interpolation_matrix_up.dot(frame_flattened).reshape(  # pyright: ignore[reportAny, reportUnknownMemberType]
                num_y_pixels, num_x_pixels
            )
        else:
            fast_movie.data[i] = interpolation_matrix_down.dot(frame_flattened).reshape(  # pyright: ignore[reportAny, reportUnknownMemberType]
                num_y_pixels, num_x_pixels
            )

    fast_movie.data = np.nan_to_num(fast_movie.data)
