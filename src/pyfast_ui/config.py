import logging
from dataclasses import field
from pathlib import Path
from typing import Any, Literal, Self, TypeAlias, TypeVar

import tomli_w
import tomllib
from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)

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
    histogram_percentile: tuple[float, float] = (0.0, 1.0)


class PhaseConfig(BaseModel):
    apply_auto_xphase: bool = True
    additional_x_phase: int = 0
    manual_y_phase: int = 0
    index_frame_to_correlate: int = 0
    sigma_gauss: int = 0
    apply_auto_yphase: bool = False
    fractional_x_phase: bool = False


class FftFilterConfig(BaseModel):
    filter_x: bool = True
    filter_y: bool = True
    filter_x_overtones: bool = False
    filter_high_pass: bool = True
    filter_pump: bool = True
    filter_noise: bool = False
    display_spectrum: bool = False
    filter_broadness: float = 0.0
    num_x_overtones: int = 10
    high_pass_params: tuple[float, float] = (1000.0, 600.0)
    num_pump_overtones: int = 3
    pump_freqs: list[float] = [
        1500.0,
        1000.0,
    ]
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
    drift_algorithm: Literal["correlation", "stackreg", "global", "known"] = (
        "correlation"
    )
    fft_drift: bool = True
    drifttype: Literal["common", "full"] = "common"
    stepsize: int = 10
    known_drift: bool = False
    stackreg_reference: Literal["previous", "first", "mean"] = "previous"
    boxcar: int = 50
    median_filter: bool = True
    subpixel: bool = False


class ImageCorrectionConfig(BaseModel):
    correction_type: Literal["align", "plane", "fixzero"] = "align"
    align_type: Literal["median", "median of diff", "mean", "poly2", "poly3"] = "median"


class ImageFilterConfig(BaseModel):
    filter_type: Literal["gauss", "median", "mean"] = "gauss"
    gauss_sigma: float = 1.0
    pixel_width: int = 3


class ExportConfig(BaseModel):
    export_movie: bool = True
    export_tiff: bool = True
    export_frames: bool = False
    double_x_pixels_tiff: bool = False
    scaling: int = 2
    fps_factor: int = 5
    auto_label: bool = True
    frame_export_images: tuple[int, int] = (0, 1)
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
        # A section may be absent in a file written by another version, so
        # nothing here may assume a key exists.
        fft_filter = config_dict.get("fft_filter") or {}
        export = config_dict.get("export") or {}

        # Change some values to tuples
        if "fft_display_range" in fft_filter:
            fft_filter["fft_display_range"] = (
                fft_filter["fft_display_range"][0],
                fft_filter["fft_display_range"][1],
            )
        if "frame_export_images" in export:
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
            general=_lenient(GeneralConfig, config_dict, "general"),
            phase=_lenient(PhaseConfig, config_dict, "phase"),
            fft_filter=_lenient(FftFilterConfig, config_dict, "fft_filter"),
            creep=_lenient(CreepConfig, config_dict, "creep"),
            drift=_lenient(DriftConfig, config_dict, "drift"),
            image_correction=_lenient(
                ImageCorrectionConfig, config_dict, "image_correction"
            ),
            image_filter=_lenient(ImageFilterConfig, config_dict, "image_filter"),
            export=_lenient(ExportConfig, config_dict, "export"),
        )


SectionT = TypeVar("SectionT", bound=BaseModel)


def _lenient(
    model: type[SectionT],
    config_dict: dict[str, Any],
    section: str,
) -> SectionT:
    """Build one config section, dropping entries the model cannot accept.

    A config file written by a newer version can contain a value this version
    does not know, for instance a correction algorithm added since. Rejecting
    the whole file for that leaves the program unable to start at all, and with
    no way to recover from inside it, because the file is read before the window
    is built. Anything that does not validate therefore falls back to the field
    default and is reported in the log.

    Args:
        model: The config model for the section.
        config_dict: The parsed file.
        section: Name of the section.

    Returns:
        The section, with unusable entries replaced by their defaults.
    """
    values = dict(config_dict.get(section) or {})

    try:
        return model(**values)
    except ValidationError as first_error:
        rejected = {
            str(error["loc"][0])
            for error in first_error.errors()
            if error.get("loc")
        }

    for name in rejected:
        log.warning(
            "Ignoring '%s' in section [%s] of the config file: %r is not a value "
            "this version accepts. Falling back to the default.",
            name,
            section,
            values.get(name),
        )
        _ = values.pop(name, None)

    try:
        return model(**values)
    except ValidationError:
        log.warning(
            "Section [%s] of the config file could not be used at all; "
            "falling back to the defaults.",
            section,
        )
        return model()


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
