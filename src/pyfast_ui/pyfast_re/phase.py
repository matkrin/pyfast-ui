from dataclasses import dataclass
from typing import TYPE_CHECKING, final

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.signal import correlate

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import DataMode, FastMovie


@dataclass
class PhaseCorrectionResult:
    data: NDArray[np.float32]
    applied_x_phase: int
    applied_y_phase: int


@final
class PhaseCorrection:
    def __init__(
        self,
        fast_movie: FastMovie,
        auto_x_phase: bool,
        frame_index_to_correlate: int,
        sigma_gauss: int = 0,
        additional_x_phase: int = 0,
        manual_y_phase: int = 0,
    ):
        self.fast_movie = fast_movie
        self.auto_x_phase = auto_x_phase
        self.frame_index_to_correlate = frame_index_to_correlate
        self.sigma_gauss = sigma_gauss
        self.additional_x_phase = additional_x_phase
        self.manual_y_phase = manual_y_phase

    def correct_phase(self) -> PhaseCorrectionResult:
        if len(self.fast_movie.data.shape) == 1 or self.fast_movie.mode != DataMode.MOVIE:
            num_images = self.fast_movie.metadata.num_images
            num_frames = self.fast_movie.metadata.num_frames
            y_points = self.fast_movie.metadata.scanner_y_points
            x_points = self.fast_movie.metadata.scanner_x_points
            ### Reshape to udi
            # Create a new array
            data = np.resize(
                self.fast_movie.data, (num_images * 2, y_points * 2, x_points)
            )
            # flip backwards lines horizontally
            data[:, 1 : y_points * 2 : 2, :] = data[:, 1 : y_points * 2 : 2, ::-1]
            # flip every up frame upside down
            data[0 : num_frames * 2 - 1 : 2, :, :] = data[
                0 : num_frames * 2 - 1 : 2, ::-1, :
            ]
        else:
            data = self.fast_movie.data.copy()

        if self.auto_x_phase:
            x_phase = get_x_phase_autocorrection(
                data, self.frame_index_to_correlate, self.sigma_gauss
            )
        else:
            x_phase = self.fast_movie.metadata.acquisition_x_phase

        x_phase += self.additional_x_phase
        y_phase = self.manual_y_phase
        x_points = data.shape[3]

        y_phase_roll = y_phase * x_points * 2
        data = np.roll(data.flatten(), x_phase + y_phase_roll)

        # self.reload_timeseries(y_phase=manual_y_phase, x_phase=x_phase)

        return PhaseCorrectionResult(data, x_phase, y_phase)


def get_x_phase_autocorrection(
    data: NDArray[np.float32], index_frame_to_correlate: int, sigma_gauss: int
) -> int:
    if len(data.shape) != 3:
        raise ValueError("Argument `data` must be 3 dimensional numpy array")

    num_of_correlated_lines = (len(data[0, :, 0]) - 4) / 2
    correlation_peak_values = np.zeros(int(num_of_correlated_lines))

    frame_to_correlate: NDArray[np.float32] = data[index_frame_to_correlate]

    frame_to_correlate -= frame_to_correlate.mean()
    frame_to_correlate /= frame_to_correlate.std()

    create_hamming = np.outer(
        np.ones(len(data[0, :, 0])), np.hamming(len(data[0, 0, :]))
    )
    frame_to_correlate = frame_to_correlate * create_hamming

    if sigma_gauss != 0:
        frame_to_correlate[::2] = gaussian_filter(frame_to_correlate[::2], sigma_gauss)
        frame_to_correlate[1::2] = gaussian_filter(
            frame_to_correlate[1::2], sigma_gauss
        )

    for i in range(2, len(data[0, :, 0]) - 2, 2):
        # create foreward different mean - like finite difference approx in numerical differentiation
        correlational_data_forewards = correlate(
            frame_to_correlate[i, :], frame_to_correlate[i + 1, :]
        )
        correlational_data_backwards = correlate(
            frame_to_correlate[i, :], frame_to_correlate[i - 1, :]
        )
        max_val = (
            np.argmax(correlational_data_forewards)
            + np.argmax(correlational_data_backwards)
        ) / 2
        correlation_peak_values[int(i / 2 - 1)] = max_val

    mean_correlation_peak_value = np.mean(correlation_peak_values)
    raw_xphase_correction = (
        mean_correlation_peak_value - (len(data[0, 0, :]) - 1)
    ) / 2  # -1 to get correct index
    xphase_autocorrection = int(np.round(raw_xphase_correction))

    # log.info(
    #     "Automatic xphase detection yielded a raw value of {} which was rounded to {}".format(
    #         round(raw_xphase_correction, 3), xphase_autocorrection
    #     )
    # )

    return xphase_autocorrection
