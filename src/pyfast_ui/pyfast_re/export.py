from __future__ import annotations

import logging
import shutil
import struct
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Hashable, Literal, TypeAlias, final

from PIL import Image
import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import skimage

from numpy.typing import NDArray
from pyfast_ui.pyfast_re.channels import FrameChannelType
from pyfast_ui.pyfast_re.data_mode import DataMode
from pyfast_ui.pyfast_re.tqdm_logging import TqdmLogger

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


def drift_mode_suffix(fast_movie: FastMovie) -> str:
    """Marker for the drift mode, to be put just before the file extension.

    "_c" for the common area, "_f" for the full padded frame, nothing if no
    drift correction was applied. Without it, running a movie through both
    modes overwrites the first result with the second.
    """
    match getattr(fast_movie, "drift_mode", None):
        case "common":
            return "_c"
        case "full":
            return "_f"
        case _:
            return ""


def _square_pixel_size(
    fast_movie: FastMovie, num_y_pixels: int, num_x_pixels: int, even: bool = False
) -> tuple[int, int]:
    """Output size in square pixels that reproduces the physical proportions of
    a frame.

    Raster formats have no notion of non-square pixels, so the frame has to be
    stretched along one axis on the way out. Only upsampling is used, so no
    information is lost, and the stretching is left to the renderer instead of
    resampling the movie.

    Args:
        fast_movie: The movie to be exported.
        num_y_pixels: Number of rows of the frames.
        num_x_pixels: Number of columns of the frames.
        even: Force both results to be even, as required by H.264.

    Returns:
        Number of output rows and output columns.
    """
    assert fast_movie.channels is not None  # type assertion

    pixel_aspect = fast_movie.metadata.pixel_aspect(fast_movie.channels)
    out_y = int(round(num_y_pixels * max(1.0, pixel_aspect)))
    out_x = int(round(num_x_pixels * max(1.0, 1.0 / pixel_aspect)))

    if even:
        out_y += out_y % 2
        out_x += out_x % 2

    return out_y, out_x


MOVIE_QUALITY = 12
"""Constant rate factor handed to x264. Lower is better and larger.

Measured against a mathematically lossless encode of the same forty frames of
FS_250903_037, in peak signal to noise ratio and file size:

    23 (the encoder's default)  27.4 dB   1.00 MB
    18                          31.2 dB   1.47 MB
    15                          34.0 dB   1.81 MB
    12                          36.7 dB   2.21 MB
    10                          38.6 dB   2.48 MB
    0  (lossless)                  inf    4.65 MB

At the encoder's own default the eight by eight blocks of the transform are
visible on the atomic corrugation and the labels are smeared; the worst pixel
is 95 grey levels out of 255. 12 costs twice the file size and brings that to
48, with little left to gain below it."""

MOVIE_FALLBACK_BITRATE = 16000
"""Kilobits per second for an encoder that takes no constant rate factor. A
generous number: the point is not to lose the corrugation."""

MOVIE_PRESET = "slow"
"""How hard x264 works for that quality. A movie is written once and watched
many times, and the difference against the default is seconds."""



def _ffmpeg_has(encoder: str) -> bool:
    """Whether the ffmpeg on this machine can encode with `encoder`.

    Not every build carries x264: the licence keeps it out of some
    distributions, and a build without it fails on the flags meant for it, at
    the point where the frames are already being piped in.
    """
    path = mpl.rcParams["animation.ffmpeg_path"]
    if shutil.which(path) is None and not Path(path).is_file():
        return False

    try:
        result = subprocess.run(
            [path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    return f" {encoder} " in result.stdout


def _movie_writer(fps: int) -> animation.AbstractMovieWriter:
    """The writer for a scientific movie, or the plain one if it cannot be had.

    Returns:
        An FFMpegWriter set to a quality worth keeping, or the default writer
        when this ffmpeg has no x264 to set it on.
    """
    if _ffmpeg_has("libx264"):
        return animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-crf",
                str(MOVIE_QUALITY),
                "-preset",
                MOVIE_PRESET,
                # 4:2:0 halves the colour resolution, which is what every
                # player can read; the structure rides on the brightness.
                "-pix_fmt",
                "yuv420p",
                # so that ffmpeg's own complaint reaches the log if it has one
                "-loglevel",
                "warning",
            ],
        )

    log.warning(
        "This ffmpeg has no libx264, so the movie is written with whatever "
        "encoder it defaults to and at that encoder's own quality. The frames "
        "will be more strongly compressed than intended."
    )

    return animation.FFMpegWriter(fps=fps, bitrate=MOVIE_FALLBACK_BITRATE)


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
            + drift_mode_suffix(fast_movie)
        )

    def export_mp4(
        self,
        fps_factor: int,
        contrast: tuple[float, float],
        color_map: str,
        label_frames: bool,
        scaling: int = 1,
    ) -> None:
        """Export of a `FastMovie` as MP4.

        Args:
            fps_factor: Multiplication factor applied on frames per second from meta data.
            contrast: The lower and upper percentile bound of the color map (0 to 1).
            color_map: Matplotlib color map.
            label_frames: Render labels with information on each frame.
            scaling: How many output pixels per measured pixel. The frames are
                not resampled for it: the axes are drawn larger and the image
                is enlarged by nearest neighbour, so no value is invented, and
                the labels and the colour scale get the room to be legible.
        """
        assert self.fast_movie.channels is not None  # type assertion

        num_frames, num_y_pixels, num_x_pixels = self.fast_movie.data.shape

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        # Video pixels are square, so the frame is stretched to its physical
        # proportions by the figure size; `aspect="auto"` lets the frame fill
        # the axes. The data itself is not resampled.
        out_y_pixels, out_x_pixels = _square_pixel_size(
            self.fast_movie, num_y_pixels, num_x_pixels, even=True
        )
        scaling = max(int(scaling), 1)
        out_y_pixels *= scaling
        out_x_pixels *= scaling

        fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            figsize=(out_x_pixels * px, out_y_pixels * px),
            frameon=False,
        )
        img_plot = ax.imshow(  # pyright: ignore[reportUnknownMemberType]
            self.fast_movie.data[0],  # pyright: ignore[reportAny]
            cmap=color_map,
            interpolation="none",
            aspect="auto",
        )

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
            # Proportional to the frame, but never so small that the encoder
            # has nothing to work with: below about fourteen points the glyphs
            # come out of the compression as a grey smudge.
            fontsize = max(0.05 * out_y_pixels, 14.0)

            label_left = ax.text(  # pyright: ignore[reportUnknownMemberType]
                num_x_pixels * padding,
                num_y_pixels * padding,
                text_left,
                fontsize=fontsize,
                color="white",
                alpha=0.8,
                horizontalalignment="left",
                verticalalignment="top",
            )
            label_right = ax.text(  # pyright: ignore[reportUnknownMemberType]
                num_x_pixels - (num_x_pixels * padding),
                num_y_pixels * padding,
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

        # The encoder's own default quality is meant for video of the world,
        # not for a frame whose information sits in a few grey levels.
        writer = _movie_writer(max(int(fps_factor * self.fast_movie.fps()), 1))
        target = f"{self.export_filepath}.mp4"
        try:
            ani.save(target, writer=writer)
        except Exception as error:
            # Whatever this ffmpeg objected to, a movie written plainly beats
            # no movie at all, and the reason belongs in the log.
            log.warning(
                "Writing %s with the chosen quality settings failed (%s). "
                "Falling back to the plain writer; the movie will be more "
                "strongly compressed.",
                Path(target).name,
                error,
            )
            ani.save(target)

    def export_tiff(self, double_x_pixels: bool = False) -> None:
        """Export the movie as multipage TIFF file.

        By default the pixel values are written unchanged and the physical
        proportions are carried by the resolution tags, so viewers can display
        the frames correctly without the data being resampled.

        Args:
            double_x_pixels: Interpolate interlaced frames to twice the number
                of columns. Interlaced channels hold the forward and the
                backward line of every scan line, so they have twice as many
                rows as columns for the same scan area; doubling the columns
                makes the pixels square. Use this for programs that ignore the
                resolution tags. Frames of non-interlaced channels are never
                touched.
        """
        assert self.fast_movie.channels is not None  # type assertion

        data = self.fast_movie.data
        pixel_aspect = self.fast_movie.metadata.pixel_aspect(self.fast_movie.channels)

        if double_x_pixels and self.fast_movie.channels.is_interlaced():
            data = np.array(
                [
                    skimage.transform.rescale(frame, (1, 2))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
                    for frame in data  # pyright: ignore[reportAny]
                ],
                dtype=np.float32,
            )
            # Same physical width over twice the columns halves the column
            # width, so a row is twice as tall relative to a column.
            pixel_aspect *= 2.0

        # Resolution is pixels per unit length, so a row that is `pixel_aspect`
        # times taller than a column is wide means a correspondingly lower
        # resolution in y. ResolutionUnit 1 means "no absolute unit", which is
        # the standard way of expressing an aspect ratio only.
        x_resolution = 1000.0
        y_resolution = x_resolution / pixel_aspect

        frame_stack = [Image.fromarray(frame) for frame in data]  # pyright: ignore[reportAny]
        frame_stack[0].save(
            f"{self.export_filepath}.tiff",
            save_all=True,
            append_images=frame_stack[1:],
            compression=None,
            tiffinfo={
                277: 1,  # SamplesPerPixel
                282: x_resolution,  # XResolution
                283: y_resolution,  # YResolution
                296: 1,  # ResolutionUnit: none
            },
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
            frame_range = (frame_range[0], fast_movie.cut_range()[1] + 1)

        self.fast_movie = fast_movie
        self.frame_range = frame_range
        self.frame_offset, _ = self.fast_movie.cut_range()
        self.export_path = fast_movie.path.resolve().parent / fast_movie.path.stem
        self.drift_suffix = drift_mode_suffix(fast_movie)

    def export_txt(self) -> None:
        """Export of frames as .txt files (Gwyddion ASCII matrix)."""
        assert self.fast_movie.channels is not None  # type assertion
        frame_channel_iterator = self.fast_movie.channels.frame_channel_iterator()

        frame_start, frame_end = self.frame_range
        frame_end += 1
        if self.fast_movie.channels.is_up_and_down():
            frame_end *= 2

        data = self.fast_movie.data[frame_start:frame_end, :, :]

        # Gwyddion takes the proportions of an ASCII matrix from the Width and
        # Height fields, so they carry the pixel aspect instead of the data
        # being resampled. See `_gwy_real_dimensions` for the choice of scale.
        pixel_aspect = self.fast_movie.metadata.pixel_aspect(self.fast_movie.channels)
        width, height = _gwy_real_dimensions(
            pixel_aspect, data.shape[1], data.shape[2]
        )

        for i in TqdmLogger(range(data.shape[0]), desc="Exporting frames"):
            frame: NDArray[np.float32] = data[i]
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.frame_offset
            channel_id = next(frame_channel_iterator)
            frame_name = f"{self.fast_movie.path.stem}_{frame_id}{channel_id}"
            header = (
                f"Channel: {frame_name}\nWidth: {width:.6g} m\n"
                f"Height: {height:.6g} m\nValue units: m"
            )
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
        frame_end += 1
        if self.fast_movie.channels.is_up_and_down():
            frame_end *= 2

        data = self.fast_movie.data[frame_start:frame_end, :, :]
        min_absolute = float(np.percentile(self.fast_movie.data, contrast[0] * 100))
        max_absolute = float(np.percentile(self.fast_movie.data, contrast[1] * 100))

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        # Image pixels are square, so the frame is stretched to its physical
        # proportions by the figure size instead of by resampling the data.
        out_y_pixels, out_x_pixels = _square_pixel_size(
            self.fast_movie, num_y_pixels, num_x_pixels
        )

        for i in TqdmLogger(range(data.shape[0]), desc="Exporting frames"):
            frame_id = i // 2 if self.fast_movie.channels.is_up_and_down() else i
            frame_id += self.frame_offset
            channel_id = next(frame_channel_iterator)
            fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                figsize=(out_x_pixels * px, out_y_pixels * px),
                frameon=False,
            )
            img_plot = ax.imshow(  # pyright: ignore[reportAny, reportUnknownMemberType]
                data[i], cmap=color_map, interpolation="none", aspect="auto"
            )
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
            + self.drift_suffix
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
        return f"{self.export_path}_{frame_id}{channel_id}{self.drift_suffix}"


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
    pixel_aspect = fast_movie.metadata.pixel_aspect(fast_movie.channels)

    for i in range(data.shape[0]):
        frame_id = i // 2 if fast_movie.channels.is_up_and_down() else i
        frame_id += fast_movie.cut_range()[0]
        channel_id = next(frame_channel_iterator)

        channel_title = bytes(f"{frame_id}-{channel_id}\0", "utf-8")
        channel_key = bytes(f"/{i}/data\0o", "utf-8")
        title_key = bytes(f"/{i}/data/title\0s", "utf-8")

        top_level_container_content += (
            channel_key
            + _gwy_make_datafield(data[i], pixel_aspect)  # pyright: ignore[reportAny]
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
    pixel_aspect = fast_movie.metadata.pixel_aspect(fast_movie.channels)

    preview = b"/brick/0/preview\0o" + _gwy_make_datafield(data[0], pixel_aspect)  # pyright: ignore[reportAny]

    GWY_HEADER = b"GWYP"
    top_level_container: bytes = b""
    channel_key = b"/brick/0\0o"
    title = bytes(
        f"/brick/0/title\0s{frame_start}-{frame_end}{fast_movie.channels.value}\0",
        "utf-8",
    )

    top_level_container += (
        channel_key
        + _gwy_make_brick(data, pixel_aspect)
        + preview
        + title
        + _gwy_make_brick_meta(fast_movie.metadata.as_dict())
    )

    container_size = struct.pack("<i", len(top_level_container))
    result: bytes = b"GwyContainer\0" + container_size + top_level_container
    content = GWY_HEADER + result

    with open(filename, "wb") as f:
        _ = f.write(content)


def _gwy_real_dimensions(pixel_aspect: float, yres: int, xres: int) -> tuple[float, float]:
    """Lateral dimensions to declare in a .gwy file.

    Gwyddion derives the displayed proportions and all lateral measurements from
    `xreal` and `yreal`, so the correct ratio has to be written there rather than
    the data being resampled. The x dimension is kept at 1.0, as before, because
    the FAST metadata states the scan amplitudes without a unit; only the ratio
    is well defined. Substitute the amplitudes here once their unit is known.

    Args:
        pixel_aspect: Physical height of one row over physical width of one
            column.
        yres: Number of rows.
        xres: Number of columns.

    Returns:
        The values for `xreal` and `yreal`.
    """
    return 1.0, yres * pixel_aspect / xres


def _gwy_make_datafield(
    frame: NDArray[np.float32], pixel_aspect: float = 1.0
) -> bytes:
    """Constructs a GwyDatafield as defined by the .gwy file format
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html).

    Args:
        frame: The frame (2D array) for which the datafield bytes are created.
        pixel_aspect: Physical height of one row over physical width of one
            column, see `Metadata.pixel_aspect`.

    Returns:
        GwyDataField as bytes.
    """
    if frame.ndim != 2:
        raise ValueError(f"frame must be 2D array, got {frame.ndim}")

    yres, xres = frame.shape
    xreal, yreal = _gwy_real_dimensions(pixel_aspect, yres, xres)

    datafield = b"".join(
        [
            b"xreal\0d" + struct.pack("<d", xreal),
            b"yreal\0d" + struct.pack("<d", yreal),
            b"xoff\0d" + struct.pack("<d", 0),
            b"yoff\0d" + struct.pack("<d", 0),
            b"xres\0i" + struct.pack("<i", xres),
            b"yres\0i" + struct.pack("<i", yres),
        ]
    )

    datafield += _gwy_make_si_unit("xy")
    datafield += _gwy_make_si_unit("z")

    data_arr = frame.flatten().astype(np.float64)
    data_arr_size = len(data_arr)

    datafield += b"data\0D" + struct.pack("<i", data_arr_size)
    datafield += data_arr.tobytes()

    datafield_size = struct.pack("<i", len(datafield))

    return b"GwyDataField\0" + datafield_size + datafield


def _gwy_make_brick(movie: NDArray[np.float32], pixel_aspect: float = 1.0) -> bytes:
    """Constructs a GwyBrick as defined by the .gwy file format.
    (http://gwyddion.net/documentation/user-guide-en/gwyfile-format.html).

    Args:
        movie: The movie (3D array) for which the brick bytes are created.
        pixel_aspect: Physical height of one row over physical width of one
            column, see `Metadata.pixel_aspect`.

    Returns:
        GwyBrick as bytes.
    """
    if movie.ndim != 3:
        raise ValueError(f"movie must be 3D array, got {movie.ndim}D")

    zres, yres, xres = movie.shape
    xreal, yreal = _gwy_real_dimensions(pixel_aspect, yres, xres)

    brick = b"".join(
        [
            b"xreal\0d" + struct.pack("<d", xreal),
            b"yreal\0d" + struct.pack("<d", yreal),
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
    unit = b"unitstr\0s" + b" \0"
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
