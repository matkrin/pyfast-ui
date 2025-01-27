from __future__ import annotations
from typing import Literal, final

import numpy as np
from numpy.typing import NDArray
import scipy
from scipy.interpolate import splev, splrep

from pyfast_ui.pyfast_re.fast_movie import Channels, FastMovie



@final
class CreepCorrection:


    def __init__(
        self,
        fast_movie: FastMovie,
        index_to_linear: float,
        creep_mode: Literal["sin", "root", "bezier"],
    ):
        self.channels = fast_movie.channels
        self.data = fast_movie.data
        self.number_x_pixels = fast_movie.metadata.scanner_x_points
        self.number_y_pixels = fast_movie.metadata.scanner_y_points
        if index_to_linear >= 1:
            index_to_linear = index_to_linear / self.number_y_pixels
        self.rel_ind_raw = index_to_linear
        self.y_grid_fold, self.y_grid_straight = self._y_grid()
        self.pixel_shift = self._get_shift(0.5)

        if creep_mode == "sin":
            self.bounds = [0.01, np.pi / 2 - 0.02]
            self.creep_function = self.sin_one_param
            self.limit_function = self.sin_limit_function

        elif creep_mode == "root":
            self.bounds = [np.int, 0]
            self.creep_function = self.root_creep
            self.limit_function = self.root_limit_function


    def _y_grid(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        dummy_y_vals = np.linspace(
            -self.number_y_pixels / 2,
            self.number_y_pixels / 2,
            self.number_y_pixels * self.number_x_pixels,
        )
        y_grid = dummy_y_vals.copy()
        y_grid = np.reshape(y_grid, (self.number_y_pixels, self.number_x_pixels))

        for i in range(self.number_y_pixels):
            y_grid[i, :] = y_grid[i, :: (-1) ** i]

        return y_grid, dummy_y_vals

    def _get_shift(self, image_range_percent: float):
        outer_cutoff = (1 - image_range_percent) / 2
        if self.channels is None:
            raise ValueError("`FastMovie.channels` is None")

        if self.channels.is_interlaced():
            out_cut_ind = int(self.number_y_pixels * outer_cutoff)
        else:
            out_cut_ind = int(self.number_y_pixels / 2 * outer_cutoff)

        l2 = np.zeros(self.number_x_pixels)
        for i in range(self.number_x_pixels):
            dat1 = self.data[3, out_cut_ind:-out_cut_ind, i]
            dat2 = self.data[4, out_cut_ind:-out_cut_ind, i]
            dat1: NDArray[np.float32] = (dat1 - dat1.mean()) / dat1.std()
            dat2: NDArray[np.float32] = (dat2 - dat2.mean()) / dat2.std()
            l1 = scipy.signal.correlate(dat1, dat2)
            l2[i] = np.argmax(l1)

        mean_shift = np.mean(l2)
        diff_in_pixels = len(dat1) - (mean_shift + 1)

        # logging

        return abs(diff_in_pixels)

    def fit_creep_sin(self, params=(0.6, ), frames: list[int] = [0, 2], known_params=None) -> None:

        if known_params == None:
            for frame_number_index in range(len(frames)):
                if frames[frame_number_index] % 2 != 0:
                    frames[frame_number_index] += 1
            fitresult: NDArray[np.float64] = np.zeros(len(params))
            count = 0
            for frame in frames:
                for row in np.linspace(
                    self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, 3
                ).astype(int):
                    try:
                        popt, pcov = scipy.optimize.curve_fit(
                            self._get_diff_from_grid,
                            (frame, row),
                            np.zeros_like(self.number_y_pixels),
                            params,
                            bounds=self.bounds,
                        )
                        fitresult += popt
                        count += 1
                    except Exception as e:
                        print(traceback.format_exc())
                        # print('Something went wrong.')
                        print('Caught Exception was "{}"'.format(e))
                        print("fit attempt failed, trying next...")
                        pass

            if count == 0:  ## Only happens if all curve_fit attempts fail.
                avg_result: NDArray[np.float64] = np.array(params)
                # self.processing_log.info(
                #     "Creep fitting failed on all iterations (all frames and rows). Using input parameter values {}:".format(
                #         avg_result
                #     )
                # )
            else:
                avg_result = np.asarray(fitresult) / float(count)
                # print("creep fit succeeded, result: {}".format(avg_result))
                # self.processing_log.info(
                #     "Creep fitting returned {} as optimal parameter values.".format(
                #         avg_result
                #     )
                # )
        else:
            avg_result = known_params
            # self.processing_log.info(
            #     "Creep parameters known. Using as {} as parameter values.".format(
            #         avg_result
            #     )
            # )
        ycreep = self.creep_function(*avg_result)
        y_creep = self.
        y_up, y_down, rest = self._shape_to_grid(ycreep)

        return (y_up, y_down)

    def fit_creep_root(self) -> None: ...
    def fit_creep_bezier(self) -> None: ...

    def _get_diff_from_grid(self, par: tuple[int, int], *params):
        """
        This function establisches the link between "curve_fit"
        in "fit_creep" and the actual construction of the
        creepcorrected grid. Then interpolates the values of the
        recorded from the uncorrected to the corrected grid. The
        function adjusts the fit for different movie modes.

        Parameters:
            par: tupple containing info which frame and which
                row within that frame to fit.
            params: Parameters of creep function.
                   Tupple of input values for creep function.

        Returns: Array of differences between up and down frame.
            This should approach zero as the creep corrected grid
            becomes better and better.
        """

        frame, row = par
        frame = int(frame)
        row = int(row)
        ycreep_up = self.creep_function(*params)
        comp_grid_up, comp_grid_down, startstop_lin = self._shape_to_grid(ycreep_up)
        if self.channels is None:
            raise ValueError("`FastMovie.channels` is None")

        # if "i" in self.channels:
        if self.channels.is_interlaced():
            tck_up = splrep(comp_grid_up[:, row], self.data[frame, :, row], s=0)
            tck_down = splrep(comp_grid_down[:, row], self.data[frame + 1, :, row], s=0)
            newvals_up = splev(self.y_grid_fold[row, :], tck_up, der=0)
            newvals_down = splev(self.y_grid_fold[row, :], tck_down, der=0)
        # elif "b" in self.channels:
        elif self.channels.is_backward():
            tck_up = splrep(comp_grid_up[1::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep( comp_grid_down[0::2, row], self.data[frame + 1, :, row], s=0)
            newvals_up = splev(self.y_grid_fold[row, 1::2], tck_up, der=0)
            newvals_down = splev(self.y_grid_fold[row, 1::2], tck_down, der=0)
        # elif "f" in self.channels:
        elif self.channels.is_forward():
            tck_up = splrep(comp_grid_up[0::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep(comp_grid_down[1::2, row], self.data[frame + 1, :, row], s=0)
            newvals_up = splev(self.y_grid_fold[row, 0::2], tck_up, der=0)
            newvals_down = splev(self.y_grid_fold[row, 0::2], tck_down, der=0)
        else:
            # self.processing_log.info("Channel Info not found in _get_diff_from_grid...")
            raise ValueError(
                'self.channels must be "i", "f" or "b" in _get_diff_from_grid. Could not detect any of those modes'
            )

        return_val = newvals_up - newvals_down
        return return_val
