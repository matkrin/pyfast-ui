from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, final

import matplotlib.pyplot as plt
import numpy as np
import scipy
from numpy.typing import NDArray

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie, FftFilterConfig


class FftFilterType(Enum):
    GAUSS = auto()
    HIGHPASS = auto()
    LOWPASS = auto()


@dataclass
class FftFilterConfig:
    filter_x: bool
    filter_y: bool
    filter_x_overtones: bool
    filter_high_pass: bool
    filter_pump: bool
    filter_noise: bool


@final
class FftFilter:
    def __init__(
        self,
        fast_movie: FastMovie,
        filter_config: FftFilterConfig,
        filter_broadness: float | None,
        num_x_overtones: int,
        num_pump_overtones: int,
        pump_freqs: list[int],
        high_pass_params: tuple[float, float],
    ):
        self.fast_movie = fast_movie
        self.filter_config = filter_config
        self.filter_broadness = filter_broadness
        self.num_x_overtones = num_x_overtones
        self.num_pump_overtones = num_pump_overtones
        self.pump_freqs = pump_freqs
        self.high_pass_params = high_pass_params

    def filter_movie(self) -> NDArray[np.float32]:
        ## Data in frequency domain
        data_fft = np.fft.rfft(self.fast_movie.data.copy())

        ## Filter frequencies
        rate = self.fast_movie.metadata.acquisition_adc_samplingrate
        frequencies = np.fft.rfftfreq(len(data_fft) * 2 - 1, 1.0 / rate)

        freqs, pars, types = self._determine_filter_frequencies()

        for filter_freq, par, type_ in zip(freqs, pars, types):
            if type_ == FftFilterType.GAUSS:
                filter = 1.0 - np.exp(-0.5 * ((frequencies - filter_freq) / par) ** 2)

            elif type_ == FftFilterType.HIGHPASS:
                filter = 0.5 + 0.5 * scipy.special.erf(  # pyright: ignore[reportAny]
                    (frequencies - filter_freq) / np.sqrt(2 * par**2)
                )

            elif type_ == FftFilterType.LOWPASS:
                filter = 0.5 - 0.5 * scipy.special.erf(  # pyright: ignore[reportAny]
                    (frequencies - filter_freq) / np.sqrt(2 * par**2)
                )

            data_fft *= filter

        ## Filter noise
        if self.filter_config.filter_noise:
            noise_threshold = np.median(np.abs(data_fft))
            # filter_noise (must be spectrum)
            sigma = noise_threshold
            filter = 0.5 + 0.5 * scipy.special.erf(
                np.abs(data_fft) - noise_threshold / np.sqrt(2 * sigma**2)
            )

            data_fft *= filter

        return np.fft.irfft(data_fft).astype(np.float32)

    def _determine_filter_frequencies(
        self,
    ) -> tuple[list[float], list[float], list[FftFilterType]]:
        x_frequency = self.fast_movie.metadata.scanner_x_frequency
        y_frequency = self.fast_movie.metadata.scanner_y_frequency
        filter_broadness = self.filter_broadness or y_frequency

        freqs: list[float] = []
        pars: list[float] = []
        types: list[FftFilterType] = []
        # Low frequencies and x frequency
        if self.filter_config.filter_y:
            freqs.append(y_frequency * 2.0)
            pars.append(y_frequency)
            types.append(FftFilterType.HIGHPASS)

        if self.filter_config.filter_x:
            freqs.append(x_frequency)
            pars.append(filter_broadness)
            types.append(FftFilterType.GAUSS)

        # Pump frequencies and overtones
        if self.filter_config.filter_pump:
            for overtone in range(self.num_pump_overtones + 1):
                for pump_frequency in self.pump_freqs:
                    freqs.append((overtone + 1) * pump_frequency)
                    pars.append(filter_broadness)
                    types.append(FftFilterType.GAUSS)

        # x overtones
        if self.filter_config.filter_x_overtones:
            for overtone in range(self.num_x_overtones):
                freqs.append(x_frequency * (overtone + 2))
                pars.append(filter_broadness)
                types.append(FftFilterType.GAUSS)

        # High pass
        if self.filter_config.filter_high_pass:
            freqs.append(self.high_pass_params[0])
            pars.append(self.high_pass_params[1])
            types.append(FftFilterType.HIGHPASS)

        return freqs, pars, types


def show_fft(
    fast_movie: FastMovie, range_display: tuple[float, float] | None = None
) -> None:
    # data must be spectrum
    data_fft = np.fft.rfft(fast_movie.data)
    rate = fast_movie.metadata.acquisition_adc_samplingrate
    frequencies = np.fft.rfftfreq(len(data_fft) * 2 - 1, 1.0 / rate)

    if range_display is not None:
        xmin = int(len(frequencies) * range_display[0] / frequencies[-1])
        xmax = int(len(frequencies) * range_display[1] / frequencies[-1])
        _ = plt.plot(frequencies[xmin:xmax], np.real(data_fft[xmin:xmax]))
    else:
        _ = plt.plot(frequencies, np.real(data_fft), np.real(data_fft))

    _ = plt.xlabel(r"$f\,\mathrm{\,in\,Hz}$")
    plt.tight_layout()
