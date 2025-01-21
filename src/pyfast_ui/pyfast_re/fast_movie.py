from __future__ import annotations
from collections.abc import Hashable
from enum import Enum, auto
from pathlib import Path
from typing import Any, Self, cast, final

import numpy as np
from numpy.typing import NDArray
import h5py as h5


class Channels(Enum):
    UDI = 0
    UDF = auto()
    UDB = auto()
    UF = auto()
    UB = auto()
    DF = auto()
    DB = auto()
    UI = auto()
    DI = auto()


class Mode(Enum):
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

    def clone(self) -> Self:
        return copy.deepcopy(self)

    def correct_phase(self) -> None:
        pass

    def correct_creep(self) -> None:
        pass

    def correct_drift(self) -> None:
        pass

    def interpolate(self) -> None:
        pass



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
        self.channels: Channels | None = None
        self.mode: Mode = Mode.TIMESERIES

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
