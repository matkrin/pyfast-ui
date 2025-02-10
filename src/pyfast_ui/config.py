from dataclasses import dataclass


@dataclass
class Config:
    channel: str = "udi"
    colormap: str = "bone"

    # Phase
    apply_auto_xphase: bool = True
    additional_x_phase: int = 0
    manual_y_phase: int = 0
    index_frame_to_correlate: int = 0
    sigma_gauss: int = 0

    # FFT filter
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

    # Creep
    creep_mode: str = "sin"
    weight_boundry: float = 0.0
    creep_num_cols: int = 3
    known_input: tuple[float, float, float] | None = None
    initial_guess: float = 0.3
    guess_ind: float = 0.2
    known_params: float | None = None

    # Drift
    drift_algorithm: str = "correlation"
    fft_drift: bool = True
    drifttype: str = "common"
    stepsize: int = 10
    known_drift: bool = False
    stackreg_reference: str = "previous"
    boxcar: int = 50
    median_filter: bool = True

    # Image correction
    correction_type: str = "align"
    align_type: str = "median"

    # Image filter
    filter_type: str = "gaussian2d"
    pixel_width: int = 3

    # Export
    export_movie: bool = True
    export_tiff: bool = True
    export_frames: bool = False
    scaling: tuple[float, float] = (2.0, 2.0)
    fps_factor: int = 5
    auto_label: bool = True
    frame_export_images: tuple[int, int] = (0, 1)
    frame_export_channel: str = "udi"
    frame_export_format: str = "gwy"
