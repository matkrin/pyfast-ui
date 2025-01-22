from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Literal, Self, final

import h5py as h5
import numpy as np
from numpy.typing import NDArray


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


class DataMode(Enum):
    TIMESERIES = 0
    MOVIE = auto()


@final
class FastMovie:
    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.basename = str(Path(filename).stem)
        self.parent_dir = str(Path(filename).resolve().parent)

        with h5.File(filename, mode="r") as f:
            self.data: NDArray[np.float32] = f["data"][()].astype(np.float32)  # pyright: ignore
            num_pixels = len(self.data)
            self.metadata = Metadata(f["data"].attrs, num_pixels)

        self.channels: Channels | None = None
        self.mode: DataMode = DataMode.TIMESERIES

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def set_channels(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ): ...

    def rescale(self, scaling_factor: tuple[int, int]) -> None: ...

    def cut(self, cut_range: tuple[int, int]) -> None: ...

    def crop(self, x_range: tuple[int, int], y_range: tuple[int, int]) -> None: ...

    def correct_phase(
        self,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int = 0,
    ) -> None: ...

    def fft_filter(
        self,
        filter_config: FftFilterConfig,
        filter_broadness: float,
        num_x_overtones: int,
        num_pump_overtones: int,
        pump_freqs: list[int],
        high_pass_params: tuple[float, float],
    ) -> None: ...

    def correct_creep_non_bezier(
        self,
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
    ) -> None: ...

    def correct_creep_bezier(
        self,
        weight_boundary: float,
        creep_num_cols: int,
        known_input: tuple[float, float, float] | None,
    ) -> None: ...

    def interpolate(self) -> None: ...

    def remove_streaks(self) -> None: ...

    def correct_drift(
        self,
        fft_drift: bool,
        drifttype: Literal["common", "full"],
        stepsize: int,
        known_drift: Literal["integrated", "sequential"] | None,
    ) -> None: ...

    def correct_frames(
        self,
        correction_type: Literal["align", "plane", "fixzero"],
        align_type: Literal["median", "mean", "poly2", "poly3"],
    ) -> None: ...

    def filter_frames(
        self,
        filter_type: Literal["gauss", "median", "mean"],
        kernel_size: int,
    ) -> None: ...

    def export_mp4(self, fps: int, color_map: str, auto_label: bool) -> None: ...

    def export_tiff(self) -> None: ...

    def export_gwy(
        self, gwy_type: Literal["images", "volume"], frame_range: tuple[int, int]
    ) -> None: ...

    def export_frames(
        self,
        image_format: Literal["png", "jpg", "bmp"],
        frame_range: tuple[int, int],
        color_map: str,
    ) -> None: ...


@dataclass
class FftFilterConfig:
    filter_x: bool
    filter_y: bool
    filter_x_overtones: bool
    filter_high_pass: bool
    filter_pump: bool
    filter_noise: bool


class CreepMode(Enum):
    SIN = "sin"
    BEZIER = "bezier"
    ROOT = "root"


class DriftType(Enum):
    FULL = "full"
    COMMON = "common"


class KnownDriftType(Enum):
    INTEGREATED = "integrated"
    SEQUENTIAL = "sequential"


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
        self.acquisition_x_phase: int = int(meta_attrs["Acquisition_X_phase"])  # pyright: ignore[reportArgumentType]
        self.acquisition_y_phase: int = int(meta_attrs["Acquisition_Y_phase"])  # pyright: ignore[reportArgumentType]
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
