from enum import Enum, auto

import numpy as np
from numpy.typing import NDArray

from pyfast_ui.pyfast_re.channels import Channels


class DataMode(Enum):
    """Indicator in which mode `FastMovie.data` is.

    TIMESERIES: Data as 1d numpy array.
    MOVIE: Data as 3d numpy array of shape (frames, y, x).
    """

    TIMESERIES = 0
    MOVIE = auto()


def reshape_data(
    time_series_data: NDArray[np.float32],
    channels: Channels,
    num_images: int,
    x_points: int,
    y_points: int,
) -> NDArray[np.float32]:
    """Reshape timeseries data into movie data.

    Args:
        time_series_data: 1D FAST data as timeseries.
        channels: The channel for which frames are extracted.
        num_images: Number of images (__not__ frames!).
        x_points: Number of points in x dimension.
        y_points: Number of points in y dimension.

    Returns:
        3D numpy array of the shape (frame, y, x).

    Raises:
        ValueError: If the movie is not in timeseries (1D array) mode.
    """
    if len(time_series_data.shape) != 1:
        raise ValueError("FastMovie must be in timeseries mode")

    data: NDArray[np.float32] = np.reshape(
        time_series_data, (num_images, y_points * 4, x_points)
    )
    num_frames = num_images * 4

    match channels:
        case Channels.UDF:
            data = data[:, 0 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            slc = slice(0, num_frames * 2 - 1, 2)
            data[slc, :, :] = data[slc, ::-1, :]

        case Channels.UDB:
            data = data[:, 1 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            slc = slice(0, num_frames * 2, 2)
            data[slc, :, :] = data[slc, ::-1, :]
            # flip backwards frames horizontally
            data[0 : num_frames * 2, :, :] = data[0 : num_frames * 2, :, ::-1]

        case Channels.UF:
            data = data[:, 0 : (2 * y_points) : 2, :]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]

        case Channels.UB:
            data = data[:, 1 : (2 * y_points) : 2, :]
            # flip backwards frames horizontally
            data[0:num_frames, :, :] = data[0:num_frames, :, ::-1]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]

        case Channels.DF:
            data = data[:, (2 * y_points) : (4 * y_points) : 2, :]

        case Channels.DB:
            data = data[:, (2 * y_points + 1) : (4 * y_points) : 2, :]
            # flip backwards frames horizontally
            data[0:num_frames, :, :] = data[0:num_frames, :, ::-1]

        case Channels.UDI:
            data = np.resize(data, (num_images * 2, y_points * 2, x_points))
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]
            # flip every up frame upside down
            slc = slice(0, num_frames * 2 - 1, 2)
            data[slc, :, :] = data[slc, ::-1, :]

        case Channels.UI:
            data = data[:, : (2 * y_points), :]
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]
            # flip every up frame upside down
            data[0:num_frames, :, :] = data[0:num_frames, ::-1, :]

        case Channels.DI:
            data = data[:, (2 * y_points) :, :]
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]

    return data
