from __future__ import annotations

import copy
from enum import Enum
from pathlib import Path
from typing import Any, Hashable, Literal, Mapping, Self, final

import h5py as h5
import numpy as np
import skimage
from numpy.typing import NDArray

from pyfast_ui.pyfast_re.channels import Channels
from pyfast_ui.pyfast_re.creep import Creep, CreepMode
from pyfast_ui.pyfast_re.data_mode import DataMode, reshape_data
from pyfast_ui.pyfast_re.drift import Drift, DriftMode, StackRegReferenceType
from pyfast_ui.pyfast_re.export import FrameExport, FrameExportFormat, MovieExport
from pyfast_ui.pyfast_re.fft_filter import FftFilter, FftFilterConfig
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
        self.cut_range = (0, self.num_images - 1)

        # Initial phase correction from file metadata
        y_phase_roll = (
            self.metadata.scanner_x_points * self.metadata.acquisition_y_phase * 2
        )
        self.data = np.roll(self.data, self.metadata.acquisition_x_phase + y_phase_roll)

    def fps(self) -> int:
        """"""
        if self.channels is not None and self.channels.is_up_and_down():
            return self.metadata.scanner_y_frequency * 2

        return self.metadata.scanner_y_frequency

    def clone(self) -> Self:
        """"""
        return copy.deepcopy(self)

    def set_channels(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ):
        """"""
        self.channels = Channels(channels.lower())

    def to_movie_mode(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ):
        """"""
        self.channels = Channels(channels.lower())
        data = reshape_data(
            self.data,
            self.channels,
            self.num_images,
            self.metadata.scanner_x_points,
            self.metadata.scanner_y_points,
        )
        # Mutate data
        self.data = data
        self.mode = DataMode.MOVIE

    def rescale(self, scaling_factor: tuple[int, int]) -> None:
        """"""
        scaled: list[NDArray[np.float32]] = []
        for i in range(self.data.shape[0]):
            scaled.append(skimage.transform.rescale(self.data[i], scaling_factor))  # pyright: ignore[reportAny, reportUnknownMemberType, reportUnknownArgumentType]

        self.data = np.array(scaled)

    def cut(self, cut_range: tuple[int, int]) -> None:
        """"""
        self.cut_range = cut_range
        # TODO

    def crop(self, x_range: tuple[int, int], y_range: tuple[int, int]) -> None:
        """"""
        # TODO
        ...

    def correct_phase(
        self,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int = 0,
    ) -> None:
        """"""
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
        """"""
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
        """"""
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
        """"""
        # must be movie mode now
        creep = Creep(self, index_to_linear=guess_ind)
        col_inds = np.linspace(
            self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, creep_num_cols
        ).astype(int)
        _opt, self.grid = creep.fit_creep_bez(
            col_inds=col_inds, w=weight_boundary, known_input=known_input
        )

    def interpolate(self) -> None:
        """"""
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
        """"""
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
        """"""
        mode = DriftMode(drifttype)
        reference = StackRegReferenceType(stackreg_reference)
        drift = Drift(self, stepsize=1, boxcar=boxcar, median_filter=median_filter)
        # Mutate data
        self.data, _drift_path = drift.correct_stackreg(mode, reference)

    def correct_drift_known(
        self,
        drifttype: Literal["common", "full"],
    ):
        """"""
        mode = DriftMode(drifttype)
        drift = Drift(self)
        # Mutate data
        self.data, _drift_path = drift.correct_known(mode)

    def remove_streaks(self) -> None:
        """"""
        # TODO

    # def correct_frames(
    #     self,
    #     correction_type: Literal["align", "plane", "fixzero"],
    #     align_type: Literal["median", "mean", "poly2", "poly3"],
    # ) -> None: ...
    def align_rows(self):
        """"""
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
    ) -> None:
        """"""
        ...
        # TODO

    def export_mp4(
        self, fps_factor: int = 1, color_map: str = "bone", label_frames: bool = True
    ) -> None:
        """"""
        movie_export = MovieExport(self)
        movie_export.export_mp4(fps_factor, color_map, label_frames)

    def export_tiff(self) -> None:
        """"""
        export = MovieExport(self)
        export.export_tiff()

    def export_frames_txt(self, frame_range: tuple[int, int]) -> None:
        """"""
        export = FrameExport(self, frame_range)
        export.export_txt()

    def export_frames_gwy(
        self, gwy_type: Literal["images", "volume"], frame_range: tuple[int, int]
    ) -> None:
        """"""
        export = FrameExport(self, frame_range)
        export.export_gwy(gwy_type)

    def export_frames_image(
        self,
        image_format: FrameExportFormat,
        frame_range: tuple[int, int],
        color_map: str,
    ) -> None:
        """"""
        export = FrameExport(self, frame_range)
        export.export_image(image_format, color_map)


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


@final
class Metadata:
    def __init__(self, meta_attrs: h5.AttributeManager, num_pixels: int) -> None:
        # If scalars are numpy types, convert to python types
        self._meta_attrs: dict[Hashable, int | float | str] = {
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in meta_attrs.items()
        }
        self._correct_missspelled_keys()
        self.acquisition_x_phase = int(self._meta_attrs["Acquisition.X_Phase"])
        self.acquisition_y_phase = int(self._meta_attrs["Acquisition.Y_Phase"])
        self.scanner_x_points = int(self._meta_attrs["Scanner.X_Points"])
        self.scanner_y_points = int(self._meta_attrs["Scanner.Y_Points"])
        self.acquisition_adc_samplingrate = float(
            self._meta_attrs["Acquisition.ADC_SamplingRate"]
        )
        self.scanner_y_frequency = int(self._meta_attrs["Scanner.Y_Frequency"])
        self.num_images = int(self._meta_attrs["Acquisition.NumImages"])
        self.num_images = self._get_correct_num_images(num_pixels)
        self.num_frames = self.num_images * 4

    def as_dict(self) -> dict[Hashable, int | float | str]:
        return self._meta_attrs

    def _get_correct_num_images(self, num_pixels: int) -> int:
        num_x_points: int = int(self._meta_attrs["Scanner.X_Points"])
        num_y_points: int = int(self._meta_attrs["Scanner.Y_Points"])
        return int(num_pixels / (num_x_points * num_y_points * 4))

    def _correct_missspelled_keys(self) -> None:
        """Remove misspelled key, if present, and add the correct one."""
        try:
            self._meta_attrs["Acquisition.X_Phase"] = self._meta_attrs.pop(
                "Acquisiton.X_Phase"
            )
            self._meta_attrs["Acquisition.Y_Phase"] = self._meta_attrs.pop(
                "Acquisiton.Y_Phase"
            )
        except KeyError:
            pass

        try:
            self._meta_attrs["Acquisition.LogAmp"] = self._meta_attrs.pop(
                "Acquisition.LogAmp."
            )
        except KeyError:
            pass
