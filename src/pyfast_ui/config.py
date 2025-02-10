import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Self, TypeAlias

import tomli_w
import tomllib
from pydantic import BaseModel

ChannelType: TypeAlias = Literal[
    "udi",
    "udf",
    "udb",
    "uf",
    "ub",
    "df",
    "db",
    "ui",
    "di",
]

FrameExportChannel: TypeAlias = Literal[
    "ui",
    "di",
    "uf",
    "ub",
    "df",
    "db",
]


class GeneralConfig(BaseModel):
    channel: ChannelType = "udi"
    colormap: str = "bone"


class PhaseConfig(BaseModel):
    apply_auto_xphase: bool = True
    additional_x_phase: int = 0
    manual_y_phase: int = 0
    index_frame_to_correlate: int = 0
    sigma_gauss: int = 0


class FftFilterConfig(BaseModel):
    filter_x: bool = True
    filter_y: bool = True
    filter_x_overtones: bool = False
    filter_high_pass: bool = True
    filter_pump: bool = True
    filter_noise: bool = False
    display_spectrum: bool = False
    filter_broadness: int | None = None
    num_x_overtones: int = 10
    high_pass_params: tuple[float, float] = (1000.0, 600.0)
    num_pump_overtones: int = 3
    pump_freqs: tuple[float, float] = (
        1500.0,
        1000.0,
    )
    fft_display_range: tuple[int, int] = (0, 40_000)


class CreepConfig(BaseModel):
    creep_mode: Literal["sin", "root", "bezier"] = "sin"
    weight_boundry: float = 0.0
    creep_num_cols: int = 3
    known_input: tuple[float, float, float] | None = None
    initial_guess: float = 0.3
    guess_ind: float = 0.2
    known_params: float | None = None


class DriftConfig(BaseModel):
    drift_algorithm: Literal["correlation", "stackreg", "known"] = "correlation"
    fft_drift: bool = True
    drifttype: Literal["common", "full"] = "common"
    stepsize: int = 10
    known_drift: bool = False
    stackreg_reference: Literal["previous", "first", "mean"] = "previous"
    boxcar: int = 50
    median_filter: bool = True


class ImageCorrectionConfig(BaseModel):
    correction_type: Literal["align", "plane", "fixzero"] = "align"
    align_type: Literal["median", "mean", "poly2", "poly3"] = "median"


class ImageFilterConfig(BaseModel):
    filter_type: Literal["gaussian2d", "median2d", "mean2d"] = "gaussian2d"
    pixel_width: int = 3


class ExportConfig(BaseModel):
    export_movie: bool = True
    export_tiff: bool = True
    export_frames: bool = False
    scaling: tuple[float, float] = (2.0, 2.0)
    fps_factor: int = 5
    auto_label: bool = True
    frame_export_images: tuple[int, int] = (0, 1)
    frame_export_channel: FrameExportChannel = "ui"
    frame_export_format: Literal["gwy", "png", "jpg", "bmp"] = "gwy"


class Config(BaseModel):
    general: GeneralConfig = field(default_factory=GeneralConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    fft_filter: FftFilterConfig = field(default_factory=FftFilterConfig)
    creep: CreepConfig = field(default_factory=CreepConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    image_correction: ImageCorrectionConfig = field(
        default_factory=ImageCorrectionConfig
    )
    image_filter: ImageFilterConfig = field(default_factory=ImageFilterConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def save_toml(self, tomlfile: Path) -> None:
        config = self.model_dump()
        for key, value in config.items():
            if value is None:
                config[key] = "None"
            if isinstance(value, dict):
                for k, v in value.items():
                    if v is None:
                        config[key][k] = "None"

        with open(tomlfile, "wb") as f:
            tomli_w.dump(config, f)

    @classmethod
    def load_toml(cls, tomlfile: Path) -> Self:
        with open(tomlfile, "rb") as f:
            config_dict = tomllib.load(f)
        fft_filter = config_dict["fft_filter"]
        export = config_dict["export"]

        # Change some values to tuples
        fft_filter["fft_display_range"] = (
            fft_filter["fft_display_range"][0],
            fft_filter["fft_display_range"][1],
        )
        fft_filter["fft_display_range"] = (
            fft_filter["fft_display_range"][0],
            fft_filter["fft_display_range"][1],
        )
        export["scaling"] = (export["scaling"][0], export["scaling"][1])
        export["frame_export_images"] = (
            export["frame_export_images"][0],
            export["frame_export_images"][1],
        )

        # Replace "None" strings to None
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if v == "None":
                        config_dict[key][k] = None

        return cls(
            general=GeneralConfig(**config_dict["general"]),
            phase=PhaseConfig(**config_dict["phase"]),
            fft_filter=FftFilterConfig(**config_dict["fft_filter"]),
            creep=CreepConfig(**config_dict["creep"]),
            drift=DriftConfig(**config_dict["drift"]),
            image_correction=ImageCorrectionConfig(**config_dict["image_correction"]),
            image_filter=ImageFilterConfig(**config_dict["image_filter"]),
            export=ExportConfig(**config_dict["export"]),
        )


def init_config() -> Config:
    home_dir = Path.home()
    config_dir = home_dir / ".pyfast-ui"
    config_file = config_dir / "config.toml"

    if not config_dir.exists():
        config_dir.mkdir()

    if not config_file.exists():
        config = Config()
        config.save_toml(config_file)

    return Config.load_toml(config_file)
