from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Self, TypeAlias, final

from PIL import Image
import h5py as h5
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from pyfast_ui.pyfast_re.channels import Channels
import skimage

from pyfast_ui.pyfast_re.creep import Creep, CreepMode
from pyfast_ui.pyfast_re.data_mode import DataMode, reshape_data
from pyfast_ui.pyfast_re.drift import Drift, DriftMode, StackRegReferenceType
from pyfast_ui.pyfast_re.fft_filter import FftFilter
from pyfast_ui.pyfast_re.interpolation import (
    apply_interpolation,
    determine_interpolation,
)
from pyfast_ui.pyfast_re.phase import PhaseCorrection


@final
class FastMovie:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.path = Path(filename)
        self.basename = str(self.path.stem)
        self.parent_dir = str(self.path.resolve().parent)

        with h5.File(filename, mode="r") as f:
            self.data: NDArray[np.float32] = f["data"][()].astype(np.float32)  # pyright: ignore
            num_pixels = len(self.data)
            self.metadata = Metadata(f["data"].attrs, num_pixels)

        self.channels: Channels | None = None
        self.mode: DataMode = DataMode.TIMESERIES
        self.grid = None
        self.num_images = self.metadata.num_images
        self.cut_range = (0, self.num_images)

        # Initial phase correction from file metadata
        y_phase_roll = (
            self.metadata.scanner_x_points * self.metadata.acquisition_y_phase * 2
        )
        self.data = np.roll(self.data, self.metadata.acquisition_x_phase + y_phase_roll)

    def fps(self) -> int:
        if self.channels is not None and self.channels.is_up_and_down():
            return self.metadata.scanner_y_frequency * 2

        return self.metadata.scanner_y_frequency

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def set_channels(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ):
        self.channels = Channels(channels.lower())

    def to_movie_mode(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ):
        self.channels = Channels(channels.lower())
        data = reshape_data(
            self.data,
            self.channels,
            self.num_images,
            self.metadata.scanner_x_points,
            self.metadata.scanner_y_points,
        )
        self.data = data
        self.mode = DataMode.MOVIE

    def rescale(self, scaling_factor: tuple[int, int]) -> None:
        scaled: list[NDArray[np.float32]] = []
        for i in range(self.data.shape[0]):
            scaled.append(skimage.transform.rescale(self.data[i], scaling_factor))  # pyright: ignore[reportAny, reportUnknownMemberType, reportUnknownArgumentType]

        self.data = np.array(scaled)

    def cut(self, cut_range: tuple[int, int]) -> None:
        self.cut_range = cut_range

    def crop(self, x_range: tuple[int, int], y_range: tuple[int, int]) -> None: ...

    def correct_phase(
        self,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int = 0,
    ) -> None:
        phase_correction = PhaseCorrection(
            fast_movie=self,
            auto_x_phase=auto_x_phase,
            frame_index_to_correlate=frame_index_to_correlate,
            sigma_gauss=sigma_gauss,
            additional_x_phase=additional_x_phase,
            manual_y_phase=manual_y_phase,
        )
        result = phase_correction.correct_phase()
        _applied_x_phase = result.applied_x_phase
        _applied_y_phase = result.applied_y_phase
        # Mutate data
        self.data = result.data

    def fft_filter(
        self,
        filter_config: FftFilterConfig,
        filter_broadness: float,
        num_x_overtones: int,
        num_pump_overtones: int,
        pump_freqs: list[int],
        high_pass_params: tuple[float, float],
    ) -> None:
        fft_filtering = FftFilter(
            fast_movie=self,
            filter_config=filter_config,
            filter_broadness=filter_broadness,
            num_x_overtones=num_x_overtones,
            num_pump_overtones=num_pump_overtones,
            pump_freqs=pump_freqs,
            high_pass_params=high_pass_params,
        )
        filtered_data = fft_filtering.filter_movie()
        # Mutate data
        self.data = filtered_data

    def correct_creep_non_bezier(
        self,
        creep_mode: Literal["sin", "root"],
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
    ) -> None:
        # must be movie mode now
        mode = CreepMode(creep_mode.lower())
        creep = Creep(self, index_to_linear=guess_ind, creep_mode=mode)
        self.grid = creep.fit_creep((initial_guess,), known_params=known_params)

    def correct_creep_bezier(
        self,
        weight_boundary: float,
        creep_num_cols: int,
        guess_ind: float,
        known_input: tuple[float, float, float] | None,
    ) -> None:
        # must be movie mode now
        creep = Creep(self, index_to_linear=guess_ind)
        col_inds = np.linspace(
            self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, creep_num_cols
        ).astype(int)
        _opt, self.grid = creep.fit_creep_bez(
            col_inds=col_inds, w=weight_boundary, known_input=known_input
        )

    def interpolate(self) -> None:
        interpolation_matrix_up, interpolation_matrix_down = determine_interpolation(
            self, offset=0.0, grid=self.grid
        )
        # Mutates data
        apply_interpolation(self, interpolation_matrix_up, interpolation_matrix_down)

    def correct_drift_correlation(
        self,
        # fft_drift: bool,
        mode: Literal["common", "full"],
        stepsize: int,
        boxcar: int,
        median_filter: bool,
    ) -> None:
        driftmode = DriftMode(mode.lower())
        drift = Drift(
            self, stepsize=stepsize, boxcar=boxcar, median_filter=median_filter
        )
        # Mutate data
        self.data, _drift_path = drift.correct_correlation(driftmode)

        # if self.show_path is True:
        #     plt.plot(self.integrated_trans[0, :], self.integrated_trans[1, :])
        #     plt.plot(transformations_conv[0, :], transformations_conv[1, :])
        #     plt.title("Drift path both raw and smoothed")
        #     plt.show()

    def correct_drift_stackreg(
        self,
        drifttype: Literal["common", "full"],
        # stepsize: int,
        stackreg_reference: Literal["previous", "first", "mean"],
        boxcar: int,
        median_filter: bool,
    ):
        mode = DriftMode(drifttype)
        reference = StackRegReferenceType(stackreg_reference)
        drift = Drift(self, stepsize=1, boxcar=boxcar, median_filter=median_filter)
        # Mutate data
        self.data, _drift_path = drift.correct_stackreg(mode, reference)

    def correct_drift_known(
        self,
        drifttype: Literal["common", "full"],
    ):
        mode = DriftMode(drifttype)
        drift = Drift(self)
        # Mutate data
        self.data, _drift_path = drift.correct_known(mode)

    def remove_streaks(self) -> None: ...

    # def correct_frames(
    #     self,
    #     correction_type: Literal["align", "plane", "fixzero"],
    #     align_type: Literal["median", "mean", "poly2", "poly3"],
    # ) -> None: ...
    def align_rows(self):
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

        # def _median_bkg(line):
        #     return np.full(line.shape[0], np.median(line))

        for i in range(self.data.shape[0]):
            # fast_movie.data[i], _ = _align_img(frame, baseline=align_type, axis=1)
            # # _align_img
            # bkg = np.apply_along_axis(_median_bkg, axis=1, self.data[i])

            background = np.apply_along_axis(
                lambda row: np.full(row.shape[0], np.median(row)),
                axis=1,
                arr=self.data[i],  # pyright: ignore[reportAny]
            )
            # Mutate data
            self.data[i] -= background

    def filter_frames(
        self,
        filter_type: Literal["gauss", "median", "mean"],
        kernel_size: int,
    ) -> None: ...

    def export_mp4(
        self, fps_factor: int = 1, color_map: str = "bone", label_frames: bool = True
    ) -> None:
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")
        if self.channels is None:
            raise ValueError("FastMovie.channels must be set")

        num_frames, num_y_pixels, num_x_pixels = self.data.shape

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
            figsize=(num_x_pixels * px, num_y_pixels * px),
            frameon=False,
        )
        img_plot = ax.imshow(self.data[0], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
        # TODO Adjustable scale
        img_plot.set_clim(self.data.min(), self.data.max())  # pyright: ignore[reportAny]
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        frame_channel_iterator = self.channels.frame_channel_iterator()
        fps = self.fps()

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
            frame_id = (
                frame_index // 2 if self.channels.is_up_and_down() else frame_index
            )
            frame_id += self.cut_range[0]
            channel_id = next(frame_channel_iterator)
            frame_text = f"{frame_id}{channel_id}"
            time_text = f"{frame_index / fps:.3f}s"

            img_plot.set_data(self.data[frame_index])  # pyright: ignore[reportAny]

            if label_frames:
                label_left.set_text(frame_text)
                label_right.set_text(time_text)

        interval = 1 / (fps_factor * self.fps()) * 1000
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=num_frames, interval=interval
        )
        ani.save(f"test-{self.channels.value}.mp4")

    def export_tiff(self) -> None:
        if self.mode != DataMode.MOVIE or len(self.data.shape) != 3:
            raise ValueError("Data must be reshaped into movie mode")

        if self.channels is None:
            raise ValueError("FastMovie.channels must be set")

        num_frames, num_y_pixels, num_x_pixels = self.data.shape
        save_folder = self.path.resolve().parent
        basename = self.path.stem
        save_path = f"{save_folder / basename}_{self.cut_range[0]}-{self.cut_range[1]}_{self.channels.value}.tiff"

        frame_stack = [Image.fromarray(frame) for frame in self.data]
        frame_stack[0].save(
            save_path,
            save_all=True,
            append_images=frame_stack[1:],
            compression=None,
            tiffinfo={277: 1},
        )

    def export_frames_txt(self, frame_range: tuple[int, int]) -> None:
        if self.mode != DataMode.MOVIE or len(self.data.shape) != 3:
            raise ValueError("Data must be reshaped into movie mode")

        if self.channels is None:
            raise ValueError("FastMovie.channels must be set")

        save_folder = self.path.resolve().parent
        basename = self.path.stem
        frame_channel_iterator = self.channels.frame_channel_iterator()

        frame_start, frame_end = frame_range
        data = self.data[frame_start:frame_end, :, :]

        for i in range(data.shape[0]):
            frame: NDArray[np.float32] = data[i]
            frame_id = i // 2 if self.channels.is_up_and_down() else i
            frame_id += self.cut_range[0]
            channel_id = next(frame_channel_iterator)
            frame_name = f"{basename}_{frame_id}{channel_id}.txt"
            save_path = save_folder / frame_name
            header = f"Channel: {frame_name}\nWidth: 1 m\nHeight: 1 m\nValue units: m"
            np.savetxt(save_path, frame, delimiter="\t", header=header, fmt="%.4e")

    def export_frames_gwy(
        self, gwy_type: Literal["images", "volume"], frame_range: tuple[int, int]
    ) -> None: ...

    def export_frames_image(
        self,
        image_format: FrameExportFormat,
        frame_range: tuple[int, int],
        color_map: str,
    ) -> None:
        """frame_range: from inclusive, end exclusive, (0, -1) for all frames"""
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

        if self.channels is None:
            raise ValueError("FastMovie.channels must be set")

        num_frames, num_y_pixels, num_x_pixels = self.data.shape

        save_folder = self.path.resolve().parent
        basename = self.path.stem
        frame_channel_iterator = self.channels.frame_channel_iterator()

        frame_start, frame_end = frame_range
        data = self.data[frame_start:frame_end, :, :]

        px: float = 1 / plt.rcParams["figure.dpi"]  # pixel in inches

        for i in range(data.shape[0]):
            frame: NDArray[np.float32] = data[i]
            frame_id = i // 2 if self.channels.is_up_and_down() else i
            frame_id += self.cut_range[0]
            channel_id = next(frame_channel_iterator)
            frame_name = f"{basename}_{frame_id}{channel_id}.{image_format}"
            save_path = save_folder / frame_name
            fig, ax = plt.subplots(  # pyright: ignore[reportUnknownMemberType]
                figsize=(num_x_pixels * px, num_y_pixels * px),
                frameon=False,
            )
            img_plot = ax.imshow(self.data[i], cmap=color_map)  # pyright: ignore[reportAny, reportUnknownMemberType]
            # TODO Adjustable scale
            img_plot.set_clim(self.data.min(), self.data.max())  # pyright: ignore[reportAny]
            fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
            fig.savefig(save_path)


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


@dataclass
class FftFilterConfig:
    filter_x: bool
    filter_y: bool
    filter_x_overtones: bool
    filter_high_pass: bool
    filter_pump: bool
    filter_noise: bool


class FrameCorrectionType(Enum):
    ALIGN = "align"
    PLANE = "plane"
    FIXZERO = "fixzero"


class AlignType(Enum):
    MEDIAN = "median"
    MEAN = "mean"
    POLY2 = "poly2"
    POLY3 = "poly3"


class FrameFilterType(Enum):
    GAUSS = "gauss"
    MEDIAN = "median"
    MEAN = "mean"


class ExportImageFormat(Enum):
    PNG = "png"
    JPG = "jpg"
    BMP = "bmp"


@final
class Metadata:
    def __init__(self, meta_attrs: h5.AttributeManager, num_pixels: int) -> None:
        self.meta_attrs = meta_attrs
        self._correct_missspelled_keys()
        self.acquisition_x_phase = int(meta_attrs["Acquisition.X_Phase"])  # pyright: ignore[reportArgumentType]
        self.acquisition_y_phase = int(meta_attrs["Acquisition.Y_Phase"])  # pyright: ignore[reportArgumentType]
        self.scanner_x_points = int(meta_attrs["Scanner.X_Points"])  # pyright: ignore[reportArgumentType]
        self.scanner_y_points = int(meta_attrs["Scanner.Y_Points"])  # pyright: ignore[reportArgumentType]
        self.acquisition_adc_samplingrate = float(
            meta_attrs["Acquisition.ADC_SamplingRate"]  # pyright: ignore[reportArgumentType]
        )
        self.scanner_y_frequency = int(meta_attrs["Scanner.Y_Frequency"])  # pyright: ignore[reportArgumentType]
        self.num_images: int = int(meta_attrs["Acquisition.NumImages"])  # pyright: ignore[reportArgumentType]
        self.num_images = self._get_correct_num_images(num_pixels)
        self.num_frames = self.num_images * 4

    def _get_correct_num_images(self, num_pixels: int) -> int:
        num_x_points: int = int(self.meta_attrs["Scanner.X_Points"])  # pyright: ignore[reportArgumentType]
        num_y_points: int = int(self.meta_attrs["Scanner.Y_Points"])  # pyright: ignore[reportArgumentType]
        return int(num_pixels / (num_x_points * num_y_points * 4))

    def _correct_missspelled_keys(self) -> None:
        # remove misspelled key, if present, and add the correct one
        try:
            self.meta_attrs["Acquisition.X_Phase"] = self.meta_attrs.pop(
                "Acquisiton.X_Phase"
            )
            self.meta_attrs["Acquisition.Y_Phase"] = self.meta_attrs.pop(
                "Acquisiton.Y_Phase"
            )
        except KeyError:
            pass

        try:
            self.meta_attrs["Acquisition.LogAmp"] = self.meta_attrs.pop(
                "Acquisition.LogAmp."
            )
        except KeyError:
            pass
