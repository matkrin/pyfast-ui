from __future__ import annotations

import copy
from pathlib import Path
from typing import Hashable, Literal, Self, final

import h5py as h5  # pyright: ignore[reportMissingTypeStubs]
import matplotlib.pyplot as plt
import numpy as np
from pyfast_ui.pyfast_re.tqdm_logging import TqdmLogger
import scipy
import skimage
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, median_filter
from scipy.signal import convolve2d

from pyfast_ui.pyfast_re import frame_corrections
from pyfast_ui.pyfast_re.channels import Channels
from pyfast_ui.pyfast_re.creep import Creep, CreepMode
from pyfast_ui.pyfast_re.data_mode import DataMode, reshape_data
from pyfast_ui.pyfast_re.drift import (
    Drift,
    DriftMode,
    StackRegReferenceType,
)
from pyfast_ui.pyfast_re.export import FrameExport, FrameExportFormat, MovieExport
from pyfast_ui.pyfast_re.fft_filter import FftFilter, FftFilterParams
from pyfast_ui.pyfast_re.interpolation import (
    apply_interpolation,
    determine_interpolation,
)
from pyfast_ui.pyfast_re.phase import PhaseCorrection


@final
class FastMovie:
    """Class representing a FastSPM movie.

    Args:
        filename: Name of the .h5 file to be opened.
        x_phase: Shift of x-phase to be applied. If `None` the value of the
            .h5 file's metadata is used.
        y_phase: Shift of y-phase to be applied. If `None` the value of the
            .h5 file's metadata is used.

    Attributes:
        path: Path to the .h5 file.
        basename: Basename of the .h5 file.
        parent_dir: Folder where the .h5 file is located.
        data: Numeric data of the FastSPM movie.
        metadata: Metadata of the FastSPM movie.
        mode: Mode of the FastSPM movie.
        num_frames: Number of frames of the FastSPM movie.
    """

    def __init__(
        self, filename: str, x_phase: int | None = None, y_phase: int | None = None
    ) -> None:
        self.filename = filename
        self.path = Path(filename)
        self.basename = str(self.path.stem)
        self.parent_dir = str(self.path.resolve().parent)

        with h5.File(filename, mode="r") as f:
            self.data: NDArray[np.float32] = f["data"][()].astype(np.float32)  # pyright: ignore[reportIndexIssue, reportAttributeAccessIssue, reportUnknownMemberType]
            num_pixels = len(self.data)
            self.metadata = Metadata(f["data"].attrs, num_pixels)

        self.channels: Channels | None = None
        self.mode: DataMode = DataMode.TIMESERIES
        self._grid = None
        self.num_frames = self.metadata.num_frames
        # Ceep track of cutting so that labels at export are correct.
        self._cut_range = (0, self.metadata.num_images)

        self._drift_path_integrated = None
        self._drift_path_sequential = None

        # Initial phase correction from either parameters or file metadata
        if x_phase is None:
            x_phase = self.metadata.acquisition_x_phase
        if y_phase is None:
            y_phase = self.metadata.acquisition_y_phase

        y_phase_roll = y_phase * self.metadata.scanner_x_points * 2
        self.data = np.roll(self.data, x_phase + y_phase_roll)

    def fps(self) -> float:
        """Get the frames per second of the FastSPM movie.

        Returns:
            The number of frames per second of the FastSPM movie.
        """
        if self.channels is not None and self.channels.is_up_and_down():
            return self.metadata.scanner_y_frequency * 2

        return self.metadata.scanner_y_frequency

    def cut_range(self) -> tuple[int, int]:
        """Get the range of cut frames.

        Returns:
            Start and end of the cut movie.
        """
        return (self._cut_range[0], self._cut_range[1] - 1)

    def clone(self) -> Self:
        """Get a deepcopy of the `FastMovie`.

        Returns:
            FastMovie: Deepcopy of the FastMovie instance.
        """
        return copy.deepcopy(self)

    def to_movie_mode(
        self, channels: Literal["udi", "udf", "udb", "uf", "ub", "df", "db", "ui", "di"]
    ) -> None:
        """Transform from `DataMode.TIMESERIES` to `DataMode.MOVIE`.
        `FastMovie.data` gets converted to a 3D array, according to `channels`.

        Args:
            channels: Channels to select.

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.TIMESERIES:
            raise ValueError("FastMovie must be in timeseries mode.")

        self.channels = Channels(channels.lower())
        data = reshape_data(
            self.data,
            self.channels,
            self.metadata.num_images,
            self.metadata.scanner_x_points,
            self.metadata.scanner_y_points,
        )
        # Mutate data
        self.data = data
        self.num_frames = data.shape[0]
        self.mode = DataMode.MOVIE

    def rescale(self, scaling_factor: tuple[int, int]) -> None:
        """Rescale the frames of the `FastMovie` is y and x dimensions.

        Args:
            scaling_factor: Values for rescaling in y and x dimension (y, x).

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE:
            raise ValueError("FastMovie must be in movie mode.")

        scaled: list[NDArray[np.float32]] = []
        for i in range(self.data.shape[0]):
            scaled.append(skimage.transform.rescale(self.data[i], scaling_factor))  # pyright: ignore[reportAny, reportUnknownMemberType, reportUnknownArgumentType]

        self.data = np.array(scaled)

    def cut(self, cut_range: tuple[int, int]) -> None:
        """Cut out frames of the movie.

        Args:
            cut_range: Start (inclusive) and end frame (exclusive) for
                cutting (start, end).

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE or len(self.data.shape) != 3:
            raise ValueError("FastMovie must be in movie mode.")

        frame_start, frame_end = cut_range
        if frame_end > self.data.shape[0]:
            raise ValueError(f"Movie does not have {frame_end} frames.")

        if self.channels is not None and self.channels.is_up_and_down():
            frame_end = frame_end * 2 - frame_start

        self.data = self.data[frame_start:frame_end, :, :]
        self.num_frames = self.data.shape[0]

        # Adjust cut range
        self._cut_range = (frame_start + self._cut_range[0], cut_range[1])

    def crop(self, x_range: tuple[int, int], y_range: tuple[int, int]) -> None:
        """Crop movie frames.

        Args:
            x_range: Starting (inclusive) and ending pixel (exclusive) in
                x-dimension.
            y_range: Starting (inclusive) and ending pixel (exclusive) in
                y-dimension.

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE or len(self.data.shape) != 3:
            raise ValueError("FastMovie must be in movie mode.")

        x_start, x_end = x_range
        y_start, y_end = y_range

        if (
            x_start < 0
            or x_end > self.data.shape[2]
            or y_start < 0
            or y_end > self.data.shape[1]
        ):
            raise ValueError(
                f"Cannot cut, dimensions of the movie are (frames, y, x): {self.data.shape}"
            )

        self.data = self.data[:, y_start:y_end, x_start:x_end]

    def correct_phase(
        self,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int | None = None,
    ) -> None:
        """Correct the phase of the `FastMovie`.

        Args:
            auto_x_phase: Determine x-phase via correlation.
            frame_index_to_correlate: Index of the frame used for correlation.
            sigma_gauss:
            additional_x_phase: Value added to the x-phase.
            manual_y_phase: Override y-phase value from metadata.

        Returns:
            None: Updates the movie's data in-place.
        """
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
        filter_config: FftFilterParams,
        filter_broadness: float,
        num_x_overtones: int,
        num_pump_overtones: int,
        pump_freqs: list[float],
        high_pass_params: tuple[float, float],
    ) -> None:
        """Filtering of noise in the Fourier space.

        Args:
            filter_config: Configuration which filters to apply.
            filter_broadness: Broadness of FFT filters which cut out single frequencies in Hz.
                If None is given, takes the y scan frequency in the metadata.
            num_x_overtones: Number of x-overtones to filter.
                No effect if `filter_x_overtones` is `False`.
            num_pump_overtones: Number of pump frequency overtones to filter.
                No effect if `filter_pump` is `False`.
            pump_freqs: List of pump frequencies to filter.
                No effect if `filter_pump` is `False`
            high_pass_params: Frequency and sigma for high pass filter.
                No effect if `filter_high_pass` is `False`.

        Returns:
            None: Updates the movie's data in-place.
        """
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

    def plot_fft(self, plot_range: tuple[float, float] = (0, 40_000)) -> None:
        data = self.data.flatten()
        # data must be spectrum
        data_fft = scipy.fft.rfft(data)
        rate = self.metadata.acquisition_adc_samplingrate
        frequencies = scipy.fft.rfftfreq(len(data_fft) * 2 - 1, 1.0 / rate)

        xmin = int(len(frequencies) * plot_range[0] / frequencies[-1])  # pyright: ignore[reportAny]
        xmax = int(len(frequencies) * plot_range[1] / frequencies[-1])  # pyright: ignore[reportAny]
        fig, ax = plt.subplots()  # pyright: ignore[reportUnknownMemberType]
        _ = ax.plot(frequencies[xmin:xmax], np.real(data_fft[xmin:xmax]))  # pyright: ignore[reportUnknownMemberType]
        _ = ax.set_xlabel(r"$f\,\mathrm{\,in\,Hz}$")  # pyright: ignore[reportUnknownMemberType]
        fig.tight_layout()
        plt.show()  # pyright: ignore[reportUnknownMemberType]

    def correct_creep_non_bezier(
        self,
        creep_mode: Literal["sin", "root"],
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
    ) -> None:
        """Create a compressed grid to correct for STM creep by virtue of
        interpolating measured values of successive up and down measurements
        to minimize the difference.

        Args:
            creep_mode: Function to apply for creep fitting.
            initial_guess: Initial guess of creep function parameters (phase for sin, controlls curvature).
            guess_ind: How much of each frame is affected by creep (Only applicable for sin).
                Values >= 1 are interpreted as lines of the frame, values between
                0 and 1 will be interpreted as relative parts of the frames.
            known_params: Known creep function parameters either from previous fit or manual determination.
                If not `None`, no fit for the creep parameters will be perfomed.
        """
        # must be movie mode now
        mode = CreepMode(creep_mode.lower())
        creep = Creep(self, index_to_linear=guess_ind, creep_mode=mode)
        self._grid = creep.fit_creep((initial_guess,), known_params=known_params)

    def correct_creep_bezier(
        self,
        weight_boundary: float,
        creep_num_cols: int,
        guess_ind: float,
        known_input: tuple[float, float, float] | None,
    ) -> None:
        """Create a compressed grid to correct for STM creep by virtue of
        interpolating measured values of successive up and down measurements
        to minimize the difference.

        Args:
            weight_boundary: Additional weighting of pixels at upper and lower
                boundary of each frame in automated creep correction.
            creep_num_cols: Number of columns for fitting of creep correction.
            guess_ind:
            known_input: `None` or tuple of (shape[0], shape[1], shape[2]).
                Known creep parameters from either a previous run or manual fitting.
                If `None` parameters are fitted.
        """
        # must be movie mode now
        creep = Creep(self, index_to_linear=guess_ind)
        col_inds = np.linspace(
            self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, creep_num_cols
        ).astype(int)
        _opt, self._grid = creep.fit_creep_bez(
            col_inds=col_inds, w=weight_boundary, known_input=known_input
        )

    def interpolate(self) -> None:
        """Interpolates the pixels in a FAST movie using the analytic positions
        of the probe.

        Returns:
            None: Updates the movie's data in-place.
        """
        interpolation_result = determine_interpolation(
            self, offset=0.0, grid=self._grid
        )
        # Mutates data
        apply_interpolation(
            self,
            interpolation_result.interpolation_matrix_up,
            interpolation_result.interpolation_matrix_down,
        )
        # Cut off unwanted padding with zeros
        self.crop((4, self.data.shape[2] - 4), (4, self.data.shape[1] - 4))

        # plt.scatter(interpolation_result.x_coords_measured.flatten(), interpolation_result.y_coords_measured.flatten(), color="black", s=5)
        # plt.scatter(interpolation_result.x_coords_target.flatten(), interpolation_result.y_coords_target.flatten(), color="red", s=5)
        # plt.show()

    def correct_drift_correlation(
        self,
        # fft_drift: bool,
        mode: Literal["common", "full"],
        stepsize: int,
        boxcar: int,
        median_filter: bool,
    ) -> None:
        """Drift correction via cross correlation.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
            stepsize: Width of the moving window of frames to correlate.
            boxcar: Boxcar width with which drift path is smoothed.
            median_filter: Whether drift path should be smoothed by median filter.

        Returns:
            None: Updates the movie's data in-place.
        """
        driftmode = DriftMode(mode.lower())
        drift = Drift(
            self, stepsize=stepsize, boxcar=boxcar, median_filter=median_filter
        )
        result = drift.correct_correlation(driftmode)
        # Mutate data
        self.data = result.data
        self._drift_path_sequential = result.drift_path_sequential
        self._drift_path_integrated = result.drift_path_integrated

    def correct_drift_stackreg(
        self,
        mode: Literal["common", "full"],
        # stepsize: int,
        stackreg_reference: Literal["previous", "first", "mean"],
        boxcar: int,
        median_filter: bool,
    ):
        """Drift correction via stackreg image registration.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding
                around frames (`"full"`).
            stackreg_reference: Reference frame for stackreg algorithm.
            boxcar: Boxcar width with which drift path is smoothed.
            median_filter: Whether drift path should be smoothed by median filter.

        Returns:
            None: Updates the movie's data in-place.
        """
        driftmode = DriftMode(mode)
        reference = StackRegReferenceType(stackreg_reference)
        drift = Drift(self, stepsize=1, boxcar=boxcar, median_filter=median_filter)
        result = drift.correct_stackreg(driftmode, reference)
        # Mutate data
        self.data = result.data
        self._drift_path_sequential = result.drift_path_sequential
        self._drift_path_integrated = result.drift_path_integrated

    def correct_drift_known(
        self,
        mode: Literal["common", "full"],
    ):
        """Drift correction from a known drift path from a '.drift.txt' file.

        Args:
            mode: Cut out the largest common area (`"common"`) or apply padding

        Returns:
            None: Updates the movie's data in-place.
        """
        driftmode = DriftMode(mode)
        drift = Drift(self)
        # Mutate data
        result = drift.correct_known(driftmode)
        # Mutate data
        self.data = result.data
        self._drift_path_sequential = result.drift_path_sequential
        self._drift_path_integrated = result.drift_path_integrated

    def plot_drift_path(self) -> None:
        """Plot the drift path. Only possible after drift correction."""
        if self._drift_path_integrated is None or self._drift_path_sequential is None:
            raise ValueError("Drift correction must be applied first")

        fig, axs = plt.subplots(nrows=1, ncols=2)  # pyright: ignore[reportAny, reportUnknownMemberType]
        axs[0].plot(self._drift_path_integrated[0])  # pyright: ignore[reportAny]
        axs[0].plot(self._drift_path_integrated[1])  # pyright: ignore[reportAny]
        axs[0].set_title("Integrated drift path")  # pyright: ignore[reportAny]

        axs[1].plot(self._drift_path_sequential[0])  # pyright: ignore[reportAny]
        axs[1].plot(self._drift_path_sequential[1])  # pyright: ignore[reportAny]
        axs[1].set_title("Sequential drift path")  # pyright: ignore[reportAny]

        for ax in axs:  # pyright: ignore[reportAny]
            ax.set_box_aspect(1)  # pyright: ignore[reportAny]
            ax.legend(["y", "x"])  # pyright: ignore[reportAny]
            ax.set_xlabel("Frames")  # pyright: ignore[reportAny]
            ax.set_ylabel("Pixel")  # pyright: ignore[reportAny]

        fig.tight_layout()
        plt.show()  # pyright: ignore[reportUnknownMemberType]

    def remove_streaks(self) -> None:
        """Remove streaks by applying a convolutional filter in y-direction."""
        edge_removal = np.array(
            [
                [-1.0 / 12.0],
                [3.0 / 12.0],
                [8.0 / 12.0],
                [3.0 / 12.0],
                [-1 / 12.0],
            ]
        )

        for i in range(self.data.shape[0]):
            self.data[i] = frame_corrections.convolve_frame(self.data[i], edge_removal)  # pyright: ignore[reportAny]

    def align_rows(
        self, align_type: Literal["median", "median of diff", "mean", "poly2", "poly3"]
    ) -> None:
        """Frame background correction by row alignment.

        Args:
            align_type: Correction algorithm for row alignment.

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

        for i in TqdmLogger(range(self.data.shape[0]), desc="Aligning rows"):
            background = frame_corrections.align_rows(self.data[i], align_type)  # pyright: ignore[reportAny]
            # Mutate data
            self.data[i] -= background

    def level_plane(self) -> None:
        """Frame background correction by subtraction of a fitted background plane.

        Returns:
            None: Updates the movie's data in-place.
        """
        for i in TqdmLogger(range(self.data.shape[0]), desc="Leveling planes"):
            background = frame_corrections.level_plane(self.data[i])  # pyright: ignore[reportAny]
            # Mutate data
            self.data[i] -= background

    def fix_zero(self) -> None:
        """Correct data offset, so that the minimum data point is fixed to zero.

        Returns:
            None: Updates the movie's data in-place.
        """
        for i in range(self.data.shape[0]):
            # Mutate data
            self.data[i] -= self.data[i].min()  # pyright: ignore[reportAny]

    def filter_frames_gaussian(self, sigma: float) -> None:
        """Filter each frame using a gaussian convolutional filter.

        Args:
            sigma: Standard deviation for Gaussian kernel.

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

        for i in range(self.data.shape[0]):
            self.data[i] = gaussian_filter(
                self.data[i],  # pyright: ignore[reportAny]
                sigma,
            )

    def filter_frames(
        self,
        filter_type: Literal["median", "mean"],
        kernel_size: int,
    ) -> None:
        """Filter each frame using a convolutional filter.

        Args:
            filter_type: Type of convolutional filter to apply.
            kernel_size: Size of convolving kernel.

        Returns:
            None: Updates the movie's data in-place.
        """
        if self.mode != DataMode.MOVIE:
            raise ValueError("Data must be reshaped into movie mode")

        match filter_type:
            case "median":
                for i in range(self.data.shape[0]):
                    self.data[i] = median_filter(
                        self.data[i],  # pyright: ignore[reportAny]
                        size=(kernel_size, kernel_size),
                    )
            case "mean":
                kernel_shape = (kernel_size, kernel_size)
                kernel = np.ones(kernel_shape) / (kernel_size * kernel_size)
                for i in range(self.data.shape[0]):
                    self.data[i] = convolve2d(self.data[i], kernel, mode="same")  # pyright: ignore[reportAny]

            case _:  # pyright: ignore[reportUnnecessaryComparison]
                raise ValueError(  # pyright: ignore[reportUnreachable]
                    "Parameter 'filter_types' must be 'gauss', 'median' or 'mean'"
                )

    def export_mp4(
        self,
        fps_factor: int = 1,
        contrast: tuple[float, float] = (0.0, 1.0),
        color_map: str = "bone",
        label_frames: bool = True,
    ) -> None:
        """Export the movie as MP4 file.

        Args:
            fps_factor: Multiplication factor applied on frames per second from meta data.
            contrast: The lower and upper percentile bound of the color map.
            color_map: Matplotlib color map
            label_frames: Render labels with information on each frame.
        """
        movie_export = MovieExport(self)
        movie_export.export_mp4(fps_factor, contrast, color_map, label_frames)

    def export_tiff(self) -> None:
        """Export the movie as multipage TIFF file."""
        export = MovieExport(self)
        export.export_tiff()

    def export_frames_txt(self, frame_range: tuple[int, int]) -> None:
        """Export frames as .txt file (ASCII matrix).

        Args:
            frame_range: Start (inclusive) and end (exclusive) frame to be exported.
        """
        export = FrameExport(self, frame_range)
        export.export_txt()

    def export_frames_gwy(
        self, gwy_type: Literal["images", "volume"], frame_range: tuple[int, int]
    ) -> None:
        """Export frames as Gwyddion .gwy file.

        Args:
            gwy_type: Decides if frames are saved as single images or a
                volume brick in the .gwy file.
            frame_range: Start (inclusive) and end (exclusive) frame to be exported.
        """
        export = FrameExport(self, frame_range)
        export.export_gwy(gwy_type)

    def export_frames_image(
        self,
        image_format: FrameExportFormat,
        frame_range: tuple[int, int],
        contrast: tuple[float, float],
        color_map: str,
    ) -> None:
        """Export frame as image file.

        Args:
            image_format: Image format of the exported frames.
            frame_range: Start (inclusive) and end (exclusive) frame to be exported.
            contrast: The lower and upper percentile bound of the color map.
            color_map: Matplotlib color map.
        """
        export = FrameExport(self, frame_range)
        export.export_image(image_format, contrast, color_map)


@final
class Metadata:
    def __init__(self, meta_attrs: h5.AttributeManager, num_pixels: int) -> None:
        # If scalars are numpy types, convert to python types
        self._meta_attrs: dict[Hashable, int | float | str] = {  # pyright: ignore[reportAttributeAccessIssue]
            k: (v.item() if isinstance(v, np.generic) else v)
            for k, v in meta_attrs.items()  # pyright: ignore[reportUnknownVariableType]
        }
        self._correct_missspelled_keys()
        self.acquisition_x_phase = int(self._meta_attrs["Acquisition.X_Phase"])
        self.acquisition_y_phase = int(self._meta_attrs["Acquisition.Y_Phase"])
        self.scanner_x_points = int(self._meta_attrs["Scanner.X_Points"])
        self.scanner_y_points = int(self._meta_attrs["Scanner.Y_Points"])
        self.acquisition_adc_samplingrate = float(
            self._meta_attrs["Acquisition.ADC_SamplingRate"]
        )
        self.scanner_x_frequency = float(self._meta_attrs["Scanner.X_Frequency"])
        self.scanner_y_frequency = float(self._meta_attrs["Scanner.Y_Frequency"])
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
