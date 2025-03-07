from __future__ import annotations

import struct
from typing import TYPE_CHECKING, Annotated, Hashable, Literal, TypeAlias, final

from PIL import Image
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray
from pyfast_ui.pyfast_re.channels import FrameChannelType
from pyfast_ui.pyfast_re.data_mode import DataMode
from pyfast_ui.pyfast_re.tqdm_logging import TqdmLogger

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


@final
class MovieExport:
    """Handles export of a complete movie.

    Args:
        fast_movie: `FastMovie` instance to be exported.
    """

    def __init__(self, fast_movie: FastMovie) -> None:
        if fast_movie.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")
        if fast_movie.channels is None:
            raise ValueError("FastMovie.channels must be set")

        self.fast_movie = fast_movie

        save_folder = fast_movie.path.resolve().parent
        basename = fast_movie.path.stem
        frame_start, frame_end = fast_movie.cut_range()
        self.export_filepath = (
            f"{save_folder / basename}_{frame_start}-"
            + f"{frame_end}_{fast_movie.channels.value}"
        )

    def export_mp4(
        self,
        fps_factor: int,
        contrast: tuple[float, float],
        color_map: str,
        label_frames: bool,
    ) -> None:
        """Export of a `FastMovie` as MP4.

        Args:
            fps_factor: Multiplication factor applied on frames per second from meta data.
            contrast: The lower and upper percentile bound of the color map (0 to 1).
            color_map: Matplotlib color map.
            label_frames: Render labels with information on each frame.
        """
        assert self.fast_movie.channels is not None  # type assertion

        num_frames, num_y_pixels, num_x_pixels = self.fast_movie.data.shape

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            figsize=(num_x_pixels * px, num_y_pixels * px),
            frameon=False,
        )
        img_plot = ax.imshow(self.fast_movie.data[0], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
        # Adjust color scale
        min_absolute = float(np.percentile(self.fast_movie.data, contrast[0] * 100))
        max_absolute = float(np.percentile(self.fast_movie.data, contrast[1] * 100))
        img_plot.set_clim(min_absolute, max_absolute)

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()
        fps = self.fast_movie.fps()
        frame_offset, _ = self.fast_movie.cut_range()

        if label_frames:
            text_left = f"{frame_offset}{next(frame_channel_iterator)}"
            right_text = f"{frame_offset / fps:.3f}s"

            padding = 0.02
            fontsize = 0.05 * num_y_pixels

            label_left = ax.text(  # pyright: ignore[reportUnknownMemberType]
                num_x_pixels * padding,
                num_x_pixels * padding,
                text_left,
                fontsize=fontsize,
                color="white",
                alpha=0.8,
                horizontalalignment="left",
                verticalalignment="top",
            )
            label_right = ax.text(  # pyright: ignore[reportUnknownMemberType]
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
            """Closure passed to `FuncAnimation`'s `func` paramter."""
            img_plot.set_data(self.fast_movie.data[frame_index])  # pyright: ignore[reportAny]

            if label_frames:
                assert self.fast_movie.channels is not None  # type assertion
                frame_id = (
                    frame_index // 2
                    if self.fast_movie.channels.is_up_and_down()
                    else frame_index
                )
                frame_id += frame_offset
                channel_id = next(frame_channel_iterator)
                frame_text = f"{frame_id}{channel_id}"
                time_text = f"{(frame_index + frame_offset) / fps:.3f}s"

                label_left.set_text(frame_text)  # pyright: ignore[reportPossiblyUnboundVariable]
                label_right.set_text(time_text)  # pyright: ignore[reportPossiblyUnboundVariable]

        #
        interval = 1 / (fps_factor * self.fast_movie.fps()) * 1000
        ani = animation.FuncAnimation(
            fig=fig,
            func=update,  # pyright: ignore[reportArgumentType]
            frames=num_frames,
            interval=interval,
        )
        ani.save(f"{self.export_filepath}.mp4")

    def export_tiff(self) -> None:
        """Export the movie as multipage TIFF file."""
        frame_stack = [Image.fromarray(frame) for frame in self.fast_movie.data]  # pyright: ignore[reportAny]
        frame_stack[0].save(
            f"{self.export_filepath}.tiff",
            save_all=True,
            append_images=frame_stack[1:],
            compression=None,
            tiffinfo={277: 1},
        )


FrameExportFormat: TypeAlias = Annotated[
    Literal[
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
    ],
    "Export formats that can be used with `FrameExport.export_image()`",
]


@final
class FrameExport:
    """Handles export of single frames or a range of frames.

    Args:
        fast_movie: `FastMovie` instance.
        frame_range: Start (inclusive) and end (exclusive) of frames to be exported.
    """

    def __init__(self, fast_movie: FastMovie, frame_range: tuple[int, int]) -> None:
        if fast_movie.mode != DataMode.MOVIE or len(fast_movie.data.shape) != 3:
            raise ValueError("Data must be reshaped into movie mode")
        if fast_movie.channels is None:
            raise ValueError("FastMovie.channels must be set")
        if frame_range[1] > fast_movie.cut_range()[1]:
            raise ValueError(
                f"Frame number {frame_range[1]} does not exist, "
                + f"the movie ends at frame {fast_movie.cut_range()[1]}"
            )

        self.fast_movie = fast_movie
        self.frame_range = frame_range
        self.frame_offset, _ = self.fast_movie.cut_range()
        self.export_path = fast_movie.path.resolve().parent / fast_movie.path.stem

    def export_txt(self) -> None:
        """Export of frames as .txt files (Gwyddion ASCII matrix)."""
        assert self.fast_movie.channels is not None  # type assertion
        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()

        frame_start, frame_end = self.frame_range
        data = self.fast_movie.data[frame_start:frame_end, :, :]

        for i in TqdmLogger(range(data.shape[0]), desc="Exporting frames"):
            frame: NDArray[np.float32] = data[i]
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.frame_offset
            channel_id = next(frame_channel_iterator)
            frame_name = f"{self.fast_movie.path.stem}_{frame_id}{channel_id}"
            header = f"Channel: {frame_name}\nWidth: 1 m\nHeight: 1 m\nValue units: m"
            export_filename = f"{self._export_filename(frame_id, channel_id)}.txt"
            np.savetxt(
                export_filename, frame, delimiter="\t", header=header, fmt="%.4e"
            )

    def export_image(
        self,
        image_format: FrameExportFormat,
        contrast: tuple[float, float],
        color_map: str,
    ) -> None:
        """Export of frames as image.

        Args:
            image_format: Image format to be exported.
            contrast: The lower and upper percentile bound of the color map.
            color_map: Any matplotlib color map to be applied on the exported frame(s).
        """
        assert self.fast_movie.channels is not None  #  type assertion
        _, num_y_pixels, num_x_pixels = self.fast_movie.data.shape
        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()

        frame_start, frame_end = self.frame_range
        if self.fast_movie.channels.is_up_and_down():
            frame_end *= 2

        data = self.fast_movie.data[frame_start:frame_end, :, :]
        min_absolute = float(np.percentile(self.fast_movie.data, contrast[0] * 100))
        max_absolute = float(np.percentile(self.fast_movie.data, contrast[1] * 100))

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        for i in TqdmLogger(range(data.shape[0]), desc="Exporting frames"):
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.frame_offset
            channel_id = next(frame_channel_iterator)
            fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                figsize=(num_x_pixels * px, num_y_pixels * px),
                frameon=False,
            )
            img_plot = ax.imshow(data[i], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
            img_plot.set_clim(min_absolute, max_absolute)

            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            export_filename = (
                f"{self._export_filename(frame_id, channel_id)}.{image_format}"
            )
            fig.savefig(export_filename)  # pyright: ignore[reportUnknownMemberType]

    def export_gwy(self, gwy_type: Literal["images", "volume"]) -> None:
        """Export of frames as .gwy files (Gwyddion native format).

        Args:
            gwy_type: Decides if frames are saved as single images or a
                volume brick inside the .gwy file.
        """
        assert self.fast_movie.channels is not None  # type assertion

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
        """Create the filename for exported file. No fileextension yet.
        <Path to original movie><Name of original movie>_<Frame ID><Channels of Frame>

        Args:
            frame_id: Frame id/number (in e.g., UDI, there is a '0' up and a '0' down frame).
            channel_id: Channel id (in e.g, UDI, the 'u' or 'd')

        Returns:
            The filename without file extentions
            (e.g., for the 0th frame of a UDI movie: '<path>_Ou').
        """
        return f"{self.export_path}_{frame_id}{channel_id}"


def _gwy_writer_images(
    fast_movie: FastMovie,
    filename: str,
    frame_range: tuple[int, int],
) -> None:
    """Writes frames within `frame_range` to disc as a .gwy file where
    all images can be accessed separately.

    Args:
        fast_movie: A `FastMovie` instance which contains the image data.
        filename: Name of the exported file (full path).
        frame_range: The range of frames that will be included in the written
            file.
    """
    assert fast_movie.channels is not None  # type assertion
    GWY_HEADER = b"GWYP"
    top_level_container_content: bytes = b""

    frame_channel_iterator = fast_movie.channels.frame_channel_iterator()

    frame_start, frame_end = frame_range
    if fast_movie.channels.is_up_and_down():
        frame_end *= 2
    data: NDArray[np.float32] = fast_movie.data[frame_start:frame_end, :, :]

    for i in range(data.shape[0]):
        frame_id = i // 2 if fast_movie.channels.is_up_and_down() else i
        frame_id += fast_movie.cut_range()[0]
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
        fast_movie: A `FastMovie` instance which contains the image data.
        filename: Name of the exported file (full path).
        frame_range: The range of frames that will be included in the written
            file.
    """
    assert fast_movie.channels is not None  # type assertion

    frame_start, frame_end = frame_range
    if fast_movie.channels.is_up_and_down():
        frame_end *= 2
    data: NDArray[np.float32] = fast_movie.data[frame_start:frame_end, :, :]

    preview = b"/brick/0/preview\0o" + _gwy_make_datafield(data[0])  # pyright: ignore[reportAny]

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
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html).

    Args:
        frame: The frame (2D array) for which the datafield bytes are created.

    Returns:
        GwyDataField as bytes.
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
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html).

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
