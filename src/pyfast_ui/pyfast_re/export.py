from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Hashable, Literal, TypeAlias, final

from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from pyfast_ui.pyfast_re.channels import FrameChannelType
from pyfast_ui.pyfast_re.data_mode import DataMode

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


@final
class MovieExport:
    def __init__(self, fast_movie: FastMovie) -> None:
        if fast_movie.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")
        if fast_movie.channels is None:
            raise ValueError("FastMovie.channels must be set")

        self.fast_movie = fast_movie

        save_folder = fast_movie.path.resolve().parent
        basename = fast_movie.path.stem
        self.export_filepath = (
            f"{save_folder / basename}_{fast_movie.cut_range[0]}-"
            + f"{fast_movie.cut_range[1]}_{fast_movie.channels.value}"
        )

    def export_mp4(self, fps_factor: int, color_map: str, label_frames: bool) -> None:
        assert self.fast_movie.channels is not None

        num_frames, num_y_pixels, num_x_pixels = self.fast_movie.data.shape

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            figsize=(num_x_pixels * px, num_y_pixels * px),
            frameon=False,
        )
        img_plot = ax.imshow(self.fast_movie.data[0], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
        # TODO Adjustable scale
        img_plot.set_clim(self.fast_movie.data.min(), self.fast_movie.data.max())  # pyright: ignore[reportAny]
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()
        fps = self.fast_movie.fps()

        if label_frames:
            text_left = f"0{next(frame_channel_iterator)}"
            right_text = f"{0 / fps:.3f}s"

            padding = 0.02
            fontsize = 0.05 * num_y_pixels

            label_left = ax.text(
                num_x_pixels * padding,
                num_x_pixels * padding,
                text_left,
                fontsize=fontsize,
                color="white",
                alpha=0.8,
                horizontalalignment="left",
                verticalalignment="top",
            )
            label_right = ax.text(
                num_x_pixels - (num_x_pixels * padding),
                num_x_pixels * padding,
                right_text,
                fontsize=fontsize,
                color="white",
                alpha=0.8,
                horizontalalignment="right",
                verticalalignment="top",
            )

        def update(frame_index: int) -> None:
            """Closure for `FuncAnimation`."""
            img_plot.set_data(self.fast_movie.data[frame_index])  # pyright: ignore[reportAny]

            if label_frames:
                frame_id = (
                    frame_index // 2
                    if self.fast_movie.channels.is_up_and_down()
                    else frame_index
                )
                frame_id += self.fast_movie.cut_range[0]
                channel_id = next(frame_channel_iterator)
                frame_text = f"{frame_id}{channel_id}"
                time_text = f"{frame_index / fps:.3f}s"

                label_left.set_text(frame_text)
                label_right.set_text(time_text)

        interval = 1 / (fps_factor * self.fast_movie.fps()) * 1000
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=num_frames, interval=interval
        )
        ani.save(f"{self.export_filepath}.mp4")

    def export_tiff(self) -> None:
        frame_stack = [Image.fromarray(frame) for frame in self.fast_movie.data]  # pyright: ignore[reportAny]
        frame_stack[0].save(
            f"{self.export_filepath}.tiff",
            save_all=True,
            append_images=frame_stack[1:],
            compression=None,
            tiffinfo={277: 1},
        )


FrameExportFormat: TypeAlias = Literal[
    "eps",
    "jpeg",
    "jpg",
    "pdf",
    "pgf",
    "png",
    "ps",
    "raw",
    "rgba",
    "svg",
    "svgz",
    "tif",
    "tiff",
    "webp",
]


@final
class FrameExport:
    def __init__(self, fast_movie: FastMovie, frame_range: tuple[int, int]) -> None:
        if fast_movie.mode != DataMode.MOVIE or len(fast_movie.data.shape) != 3:
            raise ValueError("Data must be reshaped into movie mode")
        if fast_movie.channels is None:
            raise ValueError("FastMovie.channels must be set")
        if frame_range[1] > fast_movie.cut_range[1]:
            raise ValueError(
                f"Frame number {frame_range[1]} does not exist, "
                + f"the movie ends at frame {fast_movie.cut_range[1]}"
            )

        self.fast_movie = fast_movie
        self.frame_range = frame_range
        self.export_path = fast_movie.path.resolve().parent / fast_movie.path.stem

    def export_txt(self) -> None:
        """"""
        assert self.fast_movie.channels is not None
        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()

        frame_start, frame_end = self.frame_range
        data = self.fast_movie.data[frame_start:frame_end, :, :]

        for i in range(data.shape[0]):
            frame: NDArray[np.float32] = data[i]
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.fast_movie.cut_range[0]
            channel_id = next(frame_channel_iterator)
            frame_name = f"{self.fast_movie.path.stem}_{frame_id}{channel_id}"
            header = f"Channel: {frame_name}\nWidth: 1 m\nHeight: 1 m\nValue units: m"
            export_filename = f"{self._export_filename(frame_id, channel_id)}.txt"
            np.savetxt(
                export_filename, frame, delimiter="\t", header=header, fmt="%.4e"
            )

    def export_image(self, image_format: FrameExportFormat, color_map: str) -> None:
        """frame_range: from inclusive, end exclusive, (0, -1) for all frames"""
        assert self.fast_movie.channels is not None
        _, num_y_pixels, num_x_pixels = self.fast_movie.data.shape
        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()

        frame_start, frame_end = self.frame_range
        if self.fast_movie.channels.is_up_and_down():
            frame_end *= 2

        data = self.fast_movie.data[frame_start:frame_end, :, :]

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        for i in range(data.shape[0]):
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.fast_movie.cut_range[0]
            channel_id = next(frame_channel_iterator)
            fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                figsize=(num_x_pixels * px, num_y_pixels * px),
                frameon=False,
            )
            img_plot = ax.imshow(data[i], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
            # TODO Adjustable scale
            img_plot.set_clim(data.min(), data.max())  # pyright: ignore[reportAny]
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            export_filename = (
                f"{self._export_filename(frame_id, channel_id)}.{image_format}"
            )
            fig.savefig(export_filename)  # pyright: ignore[reportUnknownMemberType]

    def export_gwy(self, gwy_type: Literal["images", "volume"]) -> None:
        """"""
        export_filename = (
            f"{self.export_path}_{self.frame_range[0]}-"
            + f"{self.frame_range[1]}_{self.fast_movie.channels.value}"
        )

        match gwy_type:
            case "images":
                export_filename += ".imgs.gwy"
                _gwy_writer_images(self.fast_movie, export_filename, self.frame_range)
            case "volume":
                export_filename += ".vol.gwy"
                _gwy_writer_volume(self.fast_movie, export_filename, self.frame_range)

    def _export_filename(self, frame_id: int, channel_id: FrameChannelType):
        """No fileextension yet. <Path to original movie><Name of original movie>_<Frame ID><Channels of Frame>"""
        return f"{self.export_path}_{frame_id}{channel_id}"


def _gwy_writer_images(
    fast_movie: FastMovie,
    filename: str,
    frame_range: tuple[int, int],
) -> None:
    """Writes frames within `image_range` to disc as a .gwy file where
    all images can be accessed separately.

    Args:
        fast_movie: A `FastMovie` instance which contains the image data
        channel: Channel to export (up/down, forward/backward)
        filename: Name of the exported file (full path)
        image_range: The range of frames that will be included in the written
            file
        scaling: Scaling to apply to each frame in xy dimensions
        metadata: Metadata that will be included in the .gwy file
    """
    assert fast_movie.channels is not None
    GWY_HEADER = b"GWYP"
    top_level_container_content: bytes = b""

    frame_channel_iterator = fast_movie.channels.frame_channel_iterator()

    frame_start, frame_end = frame_range
    if fast_movie.channels.is_up_and_down():
        frame_end *= 2
    data: NDArray[np.float32] = fast_movie.data[frame_start:frame_end, :, :]

    for i in range(data.shape[0]):
        frame_id = i // 2 if fast_movie.channels.is_up_and_down() else i
        frame_id += fast_movie.cut_range[0]
        channel_id = next(frame_channel_iterator)

        channel_title = bytes(f"{frame_id}-{channel_id}\0", "utf-8")
        channel_key = bytes(f"/{i}/data\0o", "utf-8")
        title_key = bytes(f"/{i}/data/title\0s", "utf-8")

        top_level_container_content += (
            channel_key
            + _gwy_make_datafield(data[i])  # pyright: ignore[reportAny]
            + title_key
            + channel_title
            + _gwy_make_datafield_meta(i, fast_movie.metadata.as_dict())
        )

    container_size = struct.pack("<i", len(top_level_container_content))
    result: bytes = b"GwyContainer\0" + container_size + top_level_container_content
    content = GWY_HEADER + result

    with open(filename, "wb") as f:
        _ = f.write(content)


def _gwy_writer_volume(
    fast_movie: FastMovie,
    filename: str,
    frame_range: tuple[int, int],
) -> None:
    """Writes frames within `image_range` to disc as a .gwy file where
    all images are contained in one volume data.

    Args:
        fast_movie: A `FastMovie` instance which contains the image data
        channel: Channel to export (up/down, forward/backward)
        filename: Name of the exported file (full path)
        image_range: The range of frames that will be included in the written
            file
        scaling: Scaling to apply to each frame in xy dimensions
        metadata: Metadata that will be included in the .gwy file
    """
    frame_start, frame_end = frame_range
    if fast_movie.channels.is_up_and_down():
        frame_end *= 2
    data: NDArray[np.float32] = fast_movie.data[frame_start:frame_end, :, :]

    preview = b"/brick/0/preview\0o" + _gwy_make_datafield(data[0])

    GWY_HEADER = b"GWYP"
    top_level_container: bytes = b""
    channel_key = b"/brick/0\0o"
    title = bytes(
        f"/brick/0/title\0s{frame_start}-{frame_end}{fast_movie.channels.value}\0",
        "utf-8",
    )

    top_level_container += (
        channel_key
        + _gwy_make_brick(data)
        + preview
        + title
        + _gwy_make_brick_meta(fast_movie.metadata.as_dict())
    )

    container_size = struct.pack("<i", len(top_level_container))
    result: bytes = b"GwyContainer\0" + container_size + top_level_container
    content = GWY_HEADER + result

    with open(filename, "wb") as f:
        _ = f.write(content)


def _gwy_make_datafield(frame: NDArray[np.float32]) -> bytes:
    """Constructs a GwyDatafield as defined by the .gwy file format
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html)

    Args:
        frame: The frame (2D array) for which the datafield bytes are created

    Returns:
        GwyDataField as bytes
    """
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D array, got {frame.ndim}")

    yres, xres = frame.shape

    datafield = b"".join(
        [
            b"xreal\0d" + struct.pack("<d", 1.0),
            b"yreal\0d" + struct.pack("<d", 1.0),
            b"xoff\0d" + struct.pack("<d", 0),
            b"yoff\0d" + struct.pack("<d", 0),
            b"xres\0i" + struct.pack("<i", xres),
            b"yres\0i" + struct.pack("<i", yres),
        ]
    )

    datafield += _gwy_make_si_unit("xy")
    datafield += _gwy_make_si_unit("z")

    frame[0:10, :] = 0.0
    data_arr = frame.flatten().astype(np.float64)
    data_arr_size = len(data_arr)

    datafield += b"data\0D" + struct.pack("<i", data_arr_size)
    datafield += data_arr.tobytes()

    datafield_size = struct.pack("<i", len(datafield))

    return b"GwyDataField\0" + datafield_size + datafield


def _gwy_make_brick(movie: NDArray[np.float32]) -> bytes:
    """Constructs a GwyBrick as defined by the .gwy file format.
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html)

    Args:
        movie: The movie (3D array) for which the brick bytes are created.

    Returns:
        GwyBrick as bytes.
    """
    if movie.ndim != 3:
        raise ValueError(f"movie must be 3D array, got {movie.ndim}D")

    zres, yres, xres = movie.shape

    brick = b"".join(
        [
            b"xreal\0d" + struct.pack("<d", 1.0),
            b"yreal\0d" + struct.pack("<d", 1.0),
            b"xoff\0d" + struct.pack("<d", 0),
            b"yoff\0d" + struct.pack("<d", 0),
            b"xres\0i" + struct.pack("<i", xres),
            b"yres\0i" + struct.pack("<i", yres),
            b"zres\0i" + struct.pack("<i", zres),
        ]
    )

    brick += _gwy_make_si_unit("xy")
    brick += _gwy_make_si_unit("z")

    data_arr = movie.flatten().astype(np.float64)
    data_arr_size = len(data_arr)

    brick += b"data\0D" + struct.pack("<i", data_arr_size)
    brick += data_arr.tobytes()

    brick_size = struct.pack("<i", len(brick))

    return b"GwyBrick\0" + brick_size + brick


def _gwy_make_si_unit(dimension: str) -> bytes:
    """Constructs GwySIUnit.

    Args:
        dimension: Spacial dimension of the SI Unit (either xy or z).

    Returns:
        GwySIUnit as bytes.
    """
    si_name = b"si_unit_" + bytes(dimension, "utf-8") + b"\0o" + b"GwySIUnit\0"
    unit = b"unitstr\0s" + b"m\0"
    unit_size = struct.pack("<i", len(unit))

    return si_name + unit_size + unit


def _gwy_make_datafield_meta(
    channel_num: int, metadata: dict[Hashable, int | float | str]
) -> bytes:
    """Constructs a GwyContainer containing the meta-data of an image.

    Args:
        channel_num: Number of channel the metadata belongs to
            (a contiguous number within the .gwy format).
        metadata: Metadata for the corresponding datafield.

    Returns:
        Metadata as bytes for usage within a .gwy file.
    """
    meta_name = bytes(f"/{channel_num}/meta\0o", "utf-8")
    meta_name += b"GwyContainer\0"
    meta_data = _gwy_make_meta(metadata)
    meta_data_size = struct.pack("<i", len(meta_data))
    return meta_name + meta_data_size + meta_data


def _gwy_make_brick_meta(metadata: dict[Hashable, int | float | str]) -> bytes:
    meta_name = b"/brick/0/meta\0o"
    meta_name += b"GwyContainer\0"
    meta_data = _gwy_make_meta(metadata)
    meta_data_size = struct.pack("<i", len(meta_data))
    return meta_name + meta_data_size + meta_data


def _gwy_make_meta(metadata: dict[Hashable, int | float | str]) -> bytes:
    """Constructs a GwyContainer containing the metadata of an image.

    Args:
        metadata: The metadata dict that gets converted to bytes.

    Return:
        GwyContainer containing metadata in bytes.
    """

    return b"".join(
        [bytes(f"{key}\0s{val}\0", "utf-8", "replace") for key, val in metadata.items()]
    )
