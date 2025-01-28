from typing import Callable, final, override
from PySide6.QtCore import QObject, QRunnable, Signal
import pyfastspm as pf
import numpy as np
from numpy.typing import NDArray
from pystackreg import StackReg


@final
class WorkerSignals(QObject):
    finished = Signal()


@final
class CreepWorker(QRunnable):
    def __init__(
        self,
        fast_movie: pf.FastMovie,
        creep_mode: str,
        weight_boundary: float,
        creep_num_cols: int,
        known_input: tuple[float, float, float] | None,
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
    ) -> None:
        super().__init__()

        self.ft = fast_movie
        self.creep_mode = creep_mode
        self.weight_boundary = weight_boundary
        self.creep_num_cols = creep_num_cols
        self.known_input = known_input
        self.initial_guess = initial_guess
        self.guess_ind = guess_ind
        self.known_params = known_params
        self.signals = WorkerSignals()

    @override
    def run(self) -> None:
        if self.creep_mode == "bezier":
            creep = pf.Creep(self.ft, index_to_linear=self.guess_ind)
            opt, grid = creep.fit_creep_bez(
                col_inds=np.linspace(
                    self.ft.data.shape[2] * 0.25,
                    self.ft.data.shape[2] * 0.75,
                    self.creep_num_cols,
                ).astype(int),
                w=self.weight_boundary,
                known_input=self.known_input,
            )

        elif self.creep_mode != "bezier" and self.creep_mode != "None":
            creep = pf.Creep(
                self.ft, index_to_linear=self.guess_ind, creep_mode=self.creep_mode
            )
            grid = creep.fit_creep(self.initial_guess, known_params=self.known_params)
        else:
            grid = None

        interpolation_matrix_up, interpolation_matrix_down = pf.interpolate(
            self.ft, grid=grid, give_grid=True
        )

        imrange = None

        _ = pf.interpolate(
            self.ft,
            grid=grid,
            image_range=imrange,
            interpolation_matrix_up=interpolation_matrix_up,
            interpolation_matrix_down=interpolation_matrix_down,
        )

        self.signals.finished.emit()


@final
class DriftWorker(QRunnable):
    def __init__(
        self,
        fast_movie: pf.FastMovie,
        fft_drift: bool,
        drifttype: str,
        stepsize: int,
        known_drift: bool,
        drift_algorithm: str,
        stackreg_reference: str,
    ) -> None:
        super().__init__()

        self.ft = fast_movie
        self.fft_drift = fft_drift
        self.drifttype = drifttype
        self.stepsize = stepsize
        self.known_drift = known_drift
        self.drift_algorithm = drift_algorithm
        self.stackreg_reference = stackreg_reference
        self.signals = WorkerSignals()

    @override
    def run(self) -> None:
        if self.fft_drift:
            driftrem = pf.Drift(self.ft, stepsize=self.stepsize, boxcar=True)
            corrected_data, drift_path = driftrem.correct(
                self.drifttype,
                algorithm=self.drift_algorithm,
                stackreg_reference=self.stackreg_reference,
                known_drift=self.known_drift,
            )


        self.ft.data = corrected_data

        self.signals.finished.emit()
