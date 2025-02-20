from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeAlias, final

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

    def _export_filename(self, frame_id: int, channel_id: FrameChannelType):
        return f"{self.export_path}_{frame_id}{channel_id}"

    def export_gwy(self, gwy_type: Literal["images", "volume"]) -> None:
        """"""
        ...
        # TODO
