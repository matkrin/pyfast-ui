from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, Self, final

import h5py as h5
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import skimage

from pyfast_ui.pyfast_re.creep import Creep, CreepMode
from pyfast_ui.pyfast_re.data_mode import DataMode
from pyfast_ui.pyfast_re.drift import Drift, DriftMode, StackRegReferenceType
from pyfast_ui.pyfast_re.fft_filter import FftFilter
from pyfast_ui.pyfast_re.interpolation import (
    apply_interpolation,
    determine_interpolation,
)
from pyfast_ui.pyfast_re.phase import PhaseCorrection


class Channels(Enum):
    UDI = "udi"
    UDF = "udf"
    UDB = "udb"
    UF = "uf"
    UB = "ub"
    DF = "df"
    DB = "db"
    UI = "ui"
    DI = "di"

    def is_interlaced(self) -> bool:
        return "i" in self.value

    def is_forward(self) -> bool:
        return "f" in self.value

    def is_backward(self) -> bool:
        return "b" in self.value

    def is_up_and_down(self) -> bool:
        return "u" in self.value and "d" in self.value

    def is_up_not_down(self) -> bool:
        return "u" in self.value and "d" not in self.value

    def is_down_not_up(self) -> bool:
        return "d" in self.value and "u" not in self.value

    def frame_channel_iterator(self) -> itertools.cycle[str]:
        cycle_list = []
        match self:
            case Channels.UDI:
                cycle_list = ["ui", "di"]
            case Channels.UDF:
                cycle_list = ["uf", "df"]
            case Channels.UDB:
                cycle_list = ["ub", "db"]
            case _:
                cycle_list = [self.value]

        return itertools.cycle(cycle_list)


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
        self.channels = Channels(channels)

    def to_movie_mode(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ):
        self.channels = Channels(channels)
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
        mode = CreepMode(creep_mode)
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
        drifttype: Literal["common", "full"],
        stepsize: int,
        boxcar: int,
        median_filter: bool,
    ) -> None:
        mode = DriftMode(drifttype)
        drift = Drift(
            self, stepsize=stepsize, boxcar=boxcar, median_filter=median_filter
        )
        # Mutate data
        self.data, _drift_path = drift.correct_correlation(mode)

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
        self.data, _drift_path = drift.correct_stackreg(mode, reference)

    def correct_drift_known(
        self,
        drifttype: Literal["common", "full"],
    ):
        mode = DriftMode(drifttype)
        drift = Drift(self)
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
        self, fps_factor: int = 1, color_map: str = "bone", auto_label: bool = True
    ) -> None:
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

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
            label_left.set_text(frame_text)
            label_right.set_text(time_text)

        interval = 1 / (fps_factor * self.fps()) * 1000
        ani = animation.FuncAnimation(
            fig=fig, func=update, frames=num_frames, interval=interval
        )
        ani.save(f"test-{self.channels.value}.mp4")

    def export_tiff(self) -> None: ...

    def export_frames_txt(self, frame_range: tuple[int, int]) -> None:
        if self.mode != DataMode.MOVIE:
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
            frame_name = f"{basename}_{frame_id}-{channel_id}.txt"
            save_path = save_folder / frame_name
            header = f"Channel: {frame_name}\nWidth: 1 m\nHeight: 1 m\nValue units: m"
            np.savetxt(save_path, frame, delimiter="\t", header=header, fmt="%.4e")

    def export_frames_gwy(
        self, gwy_type: Literal["images", "volume"], frame_range: tuple[int, int]
    ) -> None: ...

    def export_frames_image(
        self,
        image_format: Literal["png", "jpg", "bmp"],
        frame_range: tuple[int, int],
        color_map: str,
    ) -> None: ...


def reshape_data(
    time_series_data: NDArray[np.float32],
    channels: Channels,
    num_images: int,
    x_points: int,
    y_points: int,
):
    """
    Returns a 3D numpy array from an HDF5 file containing (image number, the 4 channels, rows).

    Args:
        time_series (1darray): the FAST data in timeseries format
        channels: a string specifying the channels to extract
        x_points (int): the number of x points
        y_points (int): the number of y points
        num_images (int): number of images
        num_frames (int): number of frames

    Returns:
        ndarray: the reshaped data as (image number, the 4 channels, rows)

    """

    data: NDArray[np.float32] = np.reshape(
        time_series_data, (num_images, y_points * 4, x_points)
    )
    num_frames = num_images * 4

    match channels:
        case Channels.UDF:
            data = data[:, 0 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]

        case Channels.UDB:
            data = data[:, 1 : (4 * y_points) : 2, :]
            data = np.resize(data, (num_images * 2, y_points, x_points))
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]
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
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]

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
