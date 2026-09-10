from __future__ import annotations
import logging
import traceback
from copy import copy
from enum import Enum
from typing import TYPE_CHECKING, final

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import Rbf, splev, splrep
from scipy.ndimage import map_coordinates
from scipy.optimize import LinearConstraint, NonlinearConstraint, curve_fit, minimize
from scipy.signal import correlate

from pyfast_ui.pyfast_re import updown
from pyfast_ui.pyfast_re.data_mode import DataMode


if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


log = logging.getLogger(__name__)


def _displace_rows(frame, amplitude):
    """Move the rows of a frame by a bulge of `amplitude` px in the middle."""
    rows, columns = frame.shape
    positions = np.arange(rows)
    bulge = amplitude * np.sin(np.pi * positions / max(rows - 1, 1))
    grid_y, grid_x = np.meshgrid(positions, np.arange(columns), indexing="ij")

    return map_coordinates(
        frame, [grid_y + bulge[:, None], grid_x], order=3, mode="nearest"
    ).astype(np.float32)


class CreepMode(Enum):
    SIN = "sin"
    BEZIER = "bezier"
    ROOT = "root"


@final
class Creep:
    """
    Create a compressed grid to correct for STM creep by virtue of
    interpolating measured values of successive up and down measurements
    to minimize the difference.

    Args:
        FastMovie_instance: object of fast_movie class
        index_to_linear: float, gives the position of the transition
        from the sine function into a linear function.
        creep_mode: sets the creep function used in fit_creep(). Applicable arguments
        are 'sin' and 'root'. The Bezier creep fit has to be called separately.

    Both x and t may refer to the "time" axis. The time axis is always noramlized to 1
    in this class. As the time axis is of no relevance elsewhere in the code, this does not
    affect the rest of pyfastspm.
    """

    MIN_INFORMATIVE_FITS = 3
    """How many of the fifteen fit attempts must produce a value of their own
    before the average is taken as a measurement. Below this the parameter is
    not determined by the data and the initial guess is kept."""

    def __init__(
        self,
        fast_movie: FastMovie,
        index_to_linear: float = 0.5,
        creep_mode: CreepMode = CreepMode.SIN,
    ):
        if fast_movie.mode != DataMode.MOVIE:
            raise ValueError(f"`FastMovie` instance must be in mode {DataMode.MOVIE}")
        if fast_movie.channels is None:
            raise ValueError("`FastMovie.channels must be set`")

        self.channels = fast_movie.channels
        # self.processing_log = fast_movie.processing_log
        self.number_xpixels = fast_movie.metadata.scanner_x_points
        self.number_ypixels = fast_movie.metadata.scanner_y_points * 2
        self.rel_ind_raw = index_to_linear
        self.data = fast_movie.data
        self.ygridfold, self.ygridstraight = self.ygrid()
        self.pixel_shift = self._get_shift(0.5)
        creep_functions = {"sin": self.sin_one_param, "root": self.root_creep}
        self.creep_bounds = {"sin": ([0.01, np.pi / 2 - 0.02]), "root": ([-np.inf, 0])}
        creep_limit_function = {
            "sin": self.sin_limit_function,
            "root": self.root_limit_function,
        }
        self.creep_function = creep_functions[creep_mode.value]
        self.limit_function = creep_limit_function[creep_mode.value]
        self.limit_function()
        self.bounds = self.creep_bounds[creep_mode.value]

    def _get_shift(self, imrange: float):
        """How far creep displaces the middle of a frame, in pixels.

        Measured as the difference between the middle of the frame and its two
        ends. Up and down sweep share their turning points, so at the first and
        the last line they are at the same height whatever the creep; a
        displacement that is there as well is a constant offset between the
        sweeps, that is the y-phase, and it is not creep. The old measurement
        took the offset in the middle of frames 3 and 4 and returned the sum of
        the two, so an uncorrected y-phase was fed to the creep correction as if
        it were creep: on FS_251128_028, 29 rows of y-phase became a creep
        amplitude of 12 px and bent the ends of every frame by 6 rows.

        Args:
            imrange: kept for the signature; the profile covers the whole frame.

        Returns: number of pixels the middle of a frame is displaced by creep.
        """
        profile = updown.mismatch_profile(self.data)
        amplitude = abs(updown.creep_amplitude(profile))
        constant = updown.constant_offset(profile)

        log.info(
            "Creep amplitude %.2f px (middle against the ends). Constant up/down "
            "offset y %.2f px, x %.2f px, which is not creep.",
            amplitude,
            constant[0],
            constant[1],
        )

        return amplitude

    def _get_shift_middle_only(self, imrange: float):
        """The previous measurement, kept for comparison.

        Args:
            imrange: percentage of movie that should be used to evaluate shift. 0.5 recommended

        Returns: number of pixels frames are shifted in the middle of the frame.
        """
        outer_cutoff = (1 - imrange) / 2
        assert self.channels is not None  # type assertion
        if self.channels.is_interlaced():
            out_cut_ind = int(self.number_ypixels * outer_cutoff)
        else:
            out_cut_ind = int(self.number_ypixels / 2 * outer_cutoff)
        l2 = np.zeros(self.number_xpixels)
        for i in range(self.number_xpixels):
            dat1 = self.data[3, out_cut_ind:-out_cut_ind, i]
            dat2 = self.data[4, out_cut_ind:-out_cut_ind, i]
            dat1: NDArray[np.float32] = (dat1 - dat1.mean()) / dat1.std()
            dat2: NDArray[np.float32] = (dat2 - dat2.mean()) / dat2.std()
            l1: NDArray[np.float32] = correlate(dat1, dat2)
            l2[i] = np.argmax(l1)

        mean_shift = np.mean(l2)
        diff_in_pixels = len(dat1) - (mean_shift + 1)

        log.info(
            f"Correlation of image rows retunred a shift of {abs(diff_in_pixels)} pixels in the middle of the frame."
        )

        # Creep displaces the ends of a frame, not its middle, so a large shift
        # measured here is a constant offset between the up and the down frames,
        # that is an uncorrected y-phase. It is not creep, and letting it set the
        # bounds lets the fit reach creep amplitudes the movie does not show.
        if abs(diff_in_pixels) > 0.05 * self.number_ypixels:
            log.warning(
                "The up/down shift in the middle of the frame is %.1f px, %.0f %% "
                "of the frame height. The middle of a frame is where creep is "
                "smallest, so a shift of this size is a constant offset between "
                "the up and the down frames: correct the y-phase before the creep, "
                "or the creep fit is given a range several times too wide.",
                abs(diff_in_pixels),
                100.0 * abs(diff_in_pixels) / self.number_ypixels,
            )

        return abs(diff_in_pixels)

    def sin_one_param(self, f):
        """
        Creep described by a Sinusoid transitioning to a linear slope.
        Args:
            f: Frequency of the sinusoid. Frequency shoudl not be taken to literal in this
            context. f merely describes to curvature of the sinusoid. After considering all
            applicable restrictions (see Documentation), f is the only remaining free parameter.
        Returns:
            Creep corrected path in 1D. This will be reshaped to a 2D grid.
        """
        ind = int((1 - self.rel_ind_raw) * self.number_xpixels * self.number_ypixels)
        # max_val is used to set the largest value the sin function will reach.
        y_max = 1 - self.pixel_shift / (self.number_ypixels * self.rel_ind_raw)
        ϕ = np.arctan((y_max * f - np.sin(f)) / (np.cos(f) - 1))
        x = np.linspace(0, 1, self.number_xpixels * self.number_ypixels - ind)
        newvals = -np.tan(ϕ) / f + np.sin(f * x + ϕ) / (np.cos(ϕ) * f)
        newvals = (
            self.ygridstraight[ind]
            + (self.ygridstraight[-1] - self.ygridstraight[ind]) * newvals
        )
        ycreep = [*self.ygridstraight[:ind], *newvals]
        ycreep_up = ycreep + (self.ygridstraight[-1] - ycreep[-1]) / 2
        return ycreep_up

    def sin_limit_function(self):
        """
        This function updates parameter limits for curve_fit when optimizing sin_one_param.
        To stay within all applicable limits (see Documentation) the free parameters of the Sine creep
        function need to stay within certain limits, which depend on the offset between the up and down frames.
        This involves calculating the inverse of two functions (one of which is a Sinc).
        To make this easier, these two function have been approxiamted by polynomials of order 5 and 7 respectively.
        Obtaining the inverse of the funcion, then comes down to finding the roots of these polynomials after subtracting the
        y-value of interest.

        Returns: Nothing - updated the limits attribute directly.
        """
        y_max = 1 - self.pixel_shift / (self.number_ypixels * self.rel_ind_raw)
        sinc_params = [
            -1.10973034e-03,
            1.09139764e-02,
            -2.94504420e-03,
            -1.65051603e-01,
            -3.59723109e-04,
            1.00001927e00,
        ]
        f_sinc = np.poly1d(sinc_params)
        dev = [
            0.01030309,
            -0.07304712,
            0.21509938,
            -0.31886432,
            0.25850888,
            -0.06193083,
            0.01681623,
            0.49952107,
        ]
        f_dev = np.poly1d(dev)

        if y_max > 0.62 and y_max < 0.64:
            for i in (f_dev - y_max).roots:
                if i.real < 1.7 and i.real > 0 and i.imag == 0:
                    val_dev = i
            for i in (f_sinc - y_max).roots:
                if i.real < 1.7 and i.real > 0 and i.imag == 0:
                    val_sinc = i

            val = min(val_sinc.real, val_dev.real)

        elif y_max <= 0.62 and y_max > 0.5:
            for i in (f_dev - y_max).roots:
                if i.real < 1.7 and i.real > 0 and i.imag == 0:
                    val = i
        elif y_max >= 0.64:
            for i in (f_sinc - y_max).roots:
                if i.real < 1.7 and i.real > 0 and i.imag == 0:
                    val = i
        else:
            raise ValueError(
                """y_max seems to be smaller than 0.5. This will produce unphysical results. This process is therefore terminated. One possible remedy is to set a larger value for index_to_linear."""
            )

        self.creep_bounds["sin"][1] = val.real

    def root_creep(self, t0):
        """
        Describes the creep as a heavily constrained root function (see Documentation)
        After considering all applicable constraints, the function only has one free parameter remaining.
        t0 alters the curvature of the root function.
        """
        t = np.linspace(0, 1, self.number_ypixels * self.number_xpixels)
        corr = self.pixel_shift / self.number_ypixels
        t0_roots = (np.sqrt(0.5 - t0) - np.sqrt(-t0)) / (np.sqrt(1 - t0) - np.sqrt(-t0))
        bterm = (0.5 - (1 - corr) * t0_roots) / (0.5 - t0_roots)
        s = (
            (1 - corr - bterm) / (2 * np.sqrt(-t0) * (np.sqrt(1 - t0) - np.sqrt(-t0)))
            + bterm
            - 1
        ) / (2)
        ycreep_up = (
            (1 - 2 * s - corr + 2 * s - bterm)
            / (np.sqrt(1 - t0) - np.sqrt(-t0))
            * (np.sqrt(t - t0) - np.sqrt(-t0))
            - 2 * s * t
            + t * bterm
        )
        offset_value = ycreep_up[-1] / 2
        ycreep_up -= offset_value
        return ycreep_up * self.number_ypixels

    def root_limit_function(self):
        """
        This function updates parameter limits for curve_fit when optimizing root_creep.
        To stay within all applicable limits (see Documentation) the free parameters of the root creep
        function need to stay within certain limits, which depend on the offset between the up and down frames.
        This involves calculating the inverse of a function, which for simplicities sake has been approximated by a 3rd order polynomial.
        In this case, this polynomial can be called directly to obtain the new limit.

        Returns: Nothing - updated the limits attribute directly.
        """
        corr = self.pixel_shift / self.number_ypixels
        poly_params = [-1.20681028e01, -8.77850843e-01, -1.55941856e-02, 1.06317446e-04]
        f_poly = np.poly1d(poly_params)
        val_poly = f_poly(corr)
        self.creep_bounds["root"][1] = val_poly

    def _shape_to_grid(self, ycreep_up):
        """Convert shape of input array to mesh."""
        ycreep_down = ycreep_up.copy()
        ycreep_down = -ycreep_down[::-1]
        ycreep_down_mesh = np.reshape(
            ycreep_down, (self.number_ypixels, self.number_xpixels)
        )
        ycreep_down_mesh[1::2, :] = ycreep_down_mesh[1::2, ::-1]

        ycreep_up_mesh = np.reshape(
            ycreep_up, (self.number_ypixels, self.number_xpixels)
        )
        ycreep_up_mesh[1::2, :] = ycreep_up_mesh[1::2, ::-1]

        return ycreep_up_mesh, ycreep_down_mesh, (ycreep_up[0], ycreep_up[-1])

    def ygrid(self):
        """
        Returns: default uncorrected y grid.
                 We are aware that this is somewhat redundant since
                 this job is beeing done in interpolate already.
                 We will sort this out in the future.
        """
        dummy_yvals = np.linspace(
            -self.number_ypixels / 2,
            self.number_ypixels / 2,
            self.number_ypixels * self.number_xpixels,
        )
        ygrid = dummy_yvals.copy()
        ygrid = np.reshape(ygrid, (self.number_ypixels, self.number_xpixels))

        for i in range(self.number_ypixels):
            ygrid[i, :] = ygrid[i, :: (-1) ** i]

        return ygrid, dummy_yvals

    def _get_diff_from_grid(self, par, *params):
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
        if self.channels.is_interlaced():
            tck_up = splrep(comp_grid_up[:, row], self.data[frame, :, row], s=0)
            tck_down = splrep(comp_grid_down[:, row], self.data[frame + 1, :, row], s=0)
            newvals_up = splev(self.ygridfold[row, :], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, :], tck_down, der=0)

        elif self.channels.is_backward():
            tck_up = splrep(comp_grid_up[1::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep(
                comp_grid_down[0::2, row], self.data[frame + 1, :, row], s=0
            )
            newvals_up = splev(self.ygridfold[row, 1::2], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, 1::2], tck_down, der=0)

        elif self.channels.is_forward():
            tck_up = splrep(comp_grid_up[0::2, row], self.data[frame, :, row], s=0)
            tck_down = splrep(
                comp_grid_down[1::2, row], self.data[frame + 1, :, row], s=0
            )
            newvals_up = splev(self.ygridfold[row, 0::2], tck_up, der=0)
            newvals_down = splev(self.ygridfold[row, 0::2], tck_down, der=0)

        else:
            self.processing_log.info("Channel Info not found in _get_diff_from_grid...")
            raise ValueError(
                'self.channels must be "i", "f" or "b" in _get_diff_from_grid. Could not detect any of those modes'
            )

        return_val = newvals_up - newvals_down
        return return_val

    def _clamp_to_bounds(self, params):
        """Bring the initial guess inside the fitted bounds.

        The guess is taken as a magnitude, so the same positive number works
        for every mode. The sin parameter is positive and the root parameter
        negative, and their ranges say which is wanted, so a value on the wrong
        side is simply reflected. Nobody has to remember the sign of the mode
        they picked, and the input box in the interface needs no negative range.

        The upper bound is not a constant: it follows from the pixel shift
        measured in the movie, so a movie with little creep gets a tight bound.
        The default guess of 0.3 sits above it for such a movie, and curve_fit
        then raises "Initial guess is outside of provided bounds" for every
        frame and row, so the fit never runs and the uncorrected input value is
        used instead, itself out of range.

        Args:
            params: The initial guess as given by the caller.

        Returns:
            The guess, moved just inside the bounds where necessary.
        """
        lower, upper = self.bounds
        span = upper - lower
        if np.isfinite(span):
            margin = 0.02 * span
        else:
            # The root mode's range is (-inf, u] with u a hair below zero, and
            # the function degenerates as the parameter approaches u: the grid
            # comes out non-monotonic, every spline fit fails with "Error on
            # input data" and the interpolation that follows keeps a single
            # column of the movie. Clamping onto the bound is therefore the
            # worst possible choice; a value well inside is needed.
            margin = 0.5
        clamped = []
        for value in params:
            wanted = float(value)
            if not lower <= wanted <= upper and lower <= -wanted <= upper:
                log.info(
                    "Creep initial guess %+.4g reflected to %+.4g, which is the "
                    "side of zero this mode works on.",
                    wanted,
                    -wanted,
                )
                wanted = -wanted

            inside = float(np.clip(wanted, lower + margin, upper - margin))
            if inside != wanted:
                outside = not lower <= wanted <= upper
                log.warning(
                    "Creep initial guess %.4g %s the bounds [%.4g, %.4g] that "
                    "this movie allows; using %.4g instead.",
                    wanted,
                    "lies outside" if outside else "sits too close to",
                    lower,
                    upper,
                    inside,
                )
            clamped.append(inside)

        return tuple(clamped)

    def _sample_frames(self, count=5):
        """Even frame numbers spread over the movie.

        The fit used to run on frames 0 and 2 alone, so a creep that changes
        over a movie of several hundred frames was represented by its first
        second, and an atypical pair at the start decided the correction for
        everything that followed.
        """
        last = max(int(self.data.shape[0]) - 2, 0)
        frames = np.linspace(0, last, count).astype(int)
        return sorted({int(frame) - int(frame) % 2 for frame in frames})

    def _is_informative(self, value, guess):
        """Whether a fitted value says anything about the movie.

        curve_fit returns the initial guess when the first step does not
        improve the residual, and a bound when the residual falls monotonically
        towards it. Both mean that the data do not determine the parameter, and
        averaging them in produces a number that looks like a measurement but is
        not one.
        """
        lower, upper = self.bounds
        span = upper - lower
        tolerance = 1e-3 * span if np.isfinite(span) else 1e-3
        if abs(value - guess) <= max(tolerance, 1e-9):
            return False
        return not (
            abs(value - lower) <= tolerance or abs(value - upper) <= tolerance
        )

    def fit_creep(self, params=(0.6,), frames=None, known_params=None):
        """
        Minimizes "_get_diff_from_grid" to obtain creep corrected
        grid of predicted STM tip movement.

        Parameters:
            params: tupple containing initial guess for input parameters
                of the creep function.
            frames: n-Tuple/list deciding which frames are used for fit. `None`
                spreads the samples over the whole movie.
            known_params: tuple or None. Params are only calculated if this is None.
                Otherwise reestimatin is skipped and known_params is used as optimizer.

        Returns:
            2-Tuple of creep corrected up and down grids.
        """
        if known_params == None:
            log.info("Starting creep correction")
            params = self._clamp_to_bounds(params)
            if frames is None:
                frames = self._sample_frames()
            else:
                frames = [
                    frame + 1 if frame % 2 != 0 else frame for frame in list(frames)
                ]
            guess = float(params[0])
            informative = []
            uninformative = 0
            for frame in frames:
                for row in np.linspace(
                    self.data.shape[2] * 0.25, self.data.shape[2] * 0.75, 3
                ).astype(int):
                    try:
                        popt, pcov = curve_fit(
                            self._get_diff_from_grid,
                            (frame, row),
                            np.zeros_like(self.number_ypixels),
                            params,
                            bounds=self.bounds,
                        )
                    except Exception as e:
                        # One line per failed fit; the full traceback of every
                        # frame and row buried the actual message in a batch run.
                        log.debug(traceback.format_exc())
                        log.warning(
                            "Creep fit failed for frame %d, row %d: %s", frame, row, e
                        )
                        continue

                    if self._is_informative(float(popt[0]), guess):
                        informative.append(np.asarray(popt, dtype=float))
                    else:
                        uninformative += 1

            attempts = len(frames) * 3
            if len(informative) < self.MIN_INFORMATIVE_FITS:
                avg_result = np.array(params)
                log.warning(
                    "Creep fit: only %d of %d attempts determined a parameter, "
                    "the rest stayed at the initial guess or ran into a bound. "
                    "One or two values out of fifteen are noise, not a "
                    "measurement, so the guess %s is kept and the creep "
                    "correction for this movie is probably better switched off. "
                    "A large measured up/down shift is the usual cause: check the "
                    "y-phase first, since a constant offset between the up and the "
                    "down frames enters that shift and inflates the bounds.",
                    len(informative),
                    attempts,
                    avg_result,
                )
            else:
                # The median, not the mean: a single frame that ran into a bound
                # would drag a mean of a handful of values a long way.
                avg_result = np.median(np.stack(informative), axis=0)
                log.info(
                    "Creep fitting returned %s, from %d of %d attempts; %d gave no "
                    "information (initial guess or bound) and were left out. "
                    "Spread of the usable values: %.4g.",
                    avg_result,
                    len(informative),
                    attempts,
                    uninformative,
                    float(np.ptp(np.stack(informative), axis=0)[0]),
                )
        else:
            avg_result = known_params
            log.info(
                f"Creep parameters known. Using as {avg_result} as parameter values."
            )
        ycreep = self.creep_function(*avg_result)
        y_up, y_down, rest = self._shape_to_grid(ycreep)
        self._check_grid(y_up, y_down, avg_result)

        return (y_up, y_down)

    def _check_grid(self, y_up, y_down, params):
        """Refuse a grid that is not a usable sampling of the frame.

        The corrected grid has to rise along a column: it is the list of
        positions the tip visited, and the interpolation reads it as one. A
        parameter at the edge of its range can produce a grid that turns back
        on itself, and then every spline fails, the interpolation marks nearly
        every pixel invalid and the movie comes out one column wide. That is
        worse than no correction, and silent, so it is caught here.
        """
        for name, grid in (("up", y_up), ("down", y_down)):
            grid = np.asarray(grid)
            column = grid[:, grid.shape[1] // 2]
            steps = np.diff(column)
            if steps.size and np.any(steps <= 0):
                raise ValueError(
                    f"The creep parameters {np.round(params, 5)} produce a "
                    f"{name} grid that does not increase along the frame "
                    f"({int(np.sum(steps <= 0))} of {steps.size} steps go "
                    "backwards). The correction is refused; for the root mode "
                    "the initial guess has to be negative and away from zero, "
                    "-0.5 is a reasonable value."
                )

    def fit_creep_measured(self, params=(0.3,), known_params=None):
        """Creep grid from the robust fit.

        Same contract as `fit_creep`, so the two can be swapped: it returns the
        up and the down grid, and honours `known_params`.
        """
        if known_params is not None:
            amplitude = self.pixel_shift
            shape = float(np.atleast_1d(known_params)[0])
            log.info(
                "Creep parameters known. Using shape %.4g with the measured "
                "amplitude %.2f px.",
                shape,
                amplitude,
            )
        else:
            amplitude, shape = self.fit_creep_robust(float(params[0]))

        self.pixel_shift = float(amplitude)
        self.limit_function()
        y_up, y_down, _, offset = self._balanced_grids(self.creep_function(shape))
        if abs(offset) > 0.5:
            log.info(
                "Creep correction: the model displaced the two sweeps by %.1f px "
                "against each other; that is a rigid shift, not creep, and has "
                "been taken back out.",
                offset,
            )

        return (y_up, y_down)

    # --- robust fit -------------------------------------------------------

    FIT_FRAMES = 6
    """Frame pairs the fit is measured on, spread over the whole movie."""

    FIT_COLUMNS = 40
    """Columns per frame. More than this buys nothing; the estimate is already
    dominated by how well the surface can be registered, not by counting."""

    FIT_BLOCKS = 8
    """Row blocks the residual displacement is measured in, at most. A block
    has to be several times taller than whatever repeats along the frame: on an
    atomically resolved surface a block one lattice period tall matches equally
    well at every period and the measurement is then free to return anything."""

    MIN_BLOCK_ROWS = 24
    """Smallest block, in rows, whatever the surface."""

    PERIODS_PER_BLOCK = 3.0
    """How many repeats of the dominant vertical period a block must span."""

    def _balanced_grids(self, path):
        """Up and down grid for a creep path, with no rigid offset between them.

        The model builds the down grid as the mirror of the up grid and recentres
        both, which is meant to leave the movie where it was. It does not: the
        larger the correction, the more the two sweeps end up displaced against
        each other, and on a movie with strong creep that reaches a dozen rows,
        which is far more than the distortion the correction removed. The offset
        is not creep, it is a rigid shift, so it is taken out here and left to
        the y-phase and the alignment, whose job it is.
        """
        up_grid, down_grid, rest = self._shape_to_grid(np.asarray(path))
        offset = float(np.mean(up_grid) - np.mean(down_grid))
        if offset:
            up_grid = up_grid - offset / 2.0
            down_grid = down_grid + offset / 2.0

        return up_grid, down_grid, rest, offset

    def _resampled_pair(self, frame, up_grid, down_grid, rows, keep, target):
        """One up frame and the following down frame, both put on the same grid."""
        columns = self._fit_columns()
        first = np.zeros((keep.size, columns.size), dtype=np.float32)
        second = np.zeros_like(first)
        for index, column in enumerate(columns):
            source_up = up_grid[rows, column]
            source_down = down_grid[rows, column]
            positions = target[keep, column]
            first[:, index] = splev(
                positions, splrep(source_up, self.data[frame, rows, column], s=0)
            )
            second[:, index] = splev(
                positions,
                splrep(source_down, self.data[frame + 1, rows, column], s=0),
            )
        return first, second

    def _fit_columns(self):
        return np.unique(
            (
                np.linspace(0.05, 0.95, self.FIT_COLUMNS) * (self.data.shape[2] - 1)
            ).astype(int)
        )

    def _vertical_period(self):
        """Dominant repeat along the frame, in rows, or 0 if there is none.

        Read off the mean power spectrum of a few frames. A lattice shows up as
        a sharp line; a surface without one leaves the spectrum smooth and the
        peak is not worth acting on.
        """
        frames = self.data[: min(len(self.data), 8)]
        rows = frames.shape[1]
        window = np.hanning(rows)[:, None]
        spectrum = np.zeros(rows // 2 + 1)
        for frame in frames:
            centred = frame - frame.mean(axis=0, keepdims=True)
            spectrum += np.abs(np.fft.rfft(centred * window, axis=0)).mean(axis=1)

        # ignore the lowest frequencies, which carry the shape of the surface
        low = max(3, rows // 60)
        band = spectrum[low:]
        if band.size < 4:
            return 0.0
        peak = int(np.argmax(band))
        if band[peak] < 3.0 * float(np.median(band)):
            return 0.0

        return float(rows) / float(peak + low)

    def _fit_block_count(self, rows):
        if getattr(self, "_period_cache", None) is None:
            self._period_cache = self._vertical_period()
        period = self._period_cache
        smallest = max(self.MIN_BLOCK_ROWS, self.PERIODS_PER_BLOCK * period)
        count = int(min(self.FIT_BLOCKS, max(3, rows // max(smallest, 1))))
        if period and not getattr(self, "_period_logged", False):
            self._period_logged = True
            log.info(
                "Creep fit: the frame repeats every %.1f rows, so the residual is "
                "measured in %d blocks of %d rows.",
                period,
                count,
                int(rows / count),
            )
        return count

    def _lattice_wavevector(self, frame):
        """Dominant 2D Fourier component of a resampled frame.

        On a surface that repeats, the displacement between two frames is read
        off the phase of that component. A block correlation cannot do it: the
        surface looks the same at every repeat, so every block returns much the
        same number and the row dependence, which is the whole quantity of
        interest, is lost. Measured on FS_251128_001 against a known
        distortion, the correlation reported a flat 1.0 px where the truth ran
        from 0.25 to 0.95 px, while the phase followed it to 0.05 px.
        """
        rows, columns = frame.shape
        window = np.outer(np.hanning(rows), np.hanning(columns))
        spectrum = np.abs(np.fft.fft2((frame - frame.mean()) * window)) ** 2
        ky = np.fft.fftfreq(rows)[:, None] * np.ones((1, columns))
        kx = np.ones((rows, 1)) * np.fft.fftfreq(columns)[None, :]
        usable = (np.abs(ky) > 2.0 / rows) & (np.abs(ky) < 0.45) & (np.abs(kx) < 0.45)
        index = np.unravel_index(np.argmax(np.where(usable, spectrum, 0.0)), spectrum.shape)

        return float(ky[index[0], 0]), float(kx[0, index[1]])

    def _phase_displacement(self, first, second, carrier, period):
        """Displacement between two blocks from the phase of their lattice."""
        amplitude_first = (first * carrier).sum()
        amplitude_second = (second * carrier).sum()
        difference = np.angle(amplitude_first * np.conjugate(amplitude_second))

        return float(difference / (2.0 * np.pi) * period)

    def _fit_frames(self):
        num_frames = len(self.data)
        return [
            int(index) - int(index) % 2
            for index in np.linspace(0, num_frames - 2, self.FIT_FRAMES)
        ]

    def residual_profile(self, amplitude, shape, estimator=None, probe=None):
        """Displacement left between up and down frames, per row block.

        The two sweeps are resampled onto the same grid with the creep
        parameters given, and what remains is measured block by block. A
        correction that is right leaves a flat profile; one that is too weak
        leaves the bulge in the middle that creep produces, and one that is too
        strong turns the bulge inside out.

        Only one scan direction is used. An interlaced frame interleaves the
        forward and the backward lines, and what is left of the doubling between
        them is a zigzag along every column that the block registration locks
        onto instead of the surface. Taking every second line removes it.

        Returns:
            The profile, or `None` if the parameters produce a grid that is not
            usable.
        """
        self.pixel_shift = float(amplitude)
        self.limit_function()
        try:
            path = np.asarray(self.creep_function(shape))
        except Exception:
            return None
        up_grid, down_grid, _, _ = self._balanced_grids(path)

        rows = slice(0, None, 2) if self.channels.is_interlaced() else slice(None)
        target = self.ygridfold
        low = max(up_grid[rows][0, :].max(), down_grid[rows][0, :].max())
        high = min(up_grid[rows][-1, :].min(), down_grid[rows][-1, :].min())
        sub_target = target[rows, :]
        keep = np.where(
            (sub_target.min(axis=1) >= low) & (sub_target.max(axis=1) <= high)
        )[0]
        blocks = self._fit_block_count(keep.size)
        if keep.size < 3 * blocks:
            return None

        columns = self._fit_columns()
        if np.any(np.diff(up_grid[rows][:, columns], axis=0) <= 0) or np.any(
            np.diff(down_grid[rows][:, columns], axis=0) <= 0
        ):
            # A grid that turns back on itself cannot be interpolated on.
            return None

        if estimator is None:
            estimator = getattr(self, "_estimator", "correlation")

        edges = np.linspace(1, keep.size - 1, blocks + 1).astype(int)
        measured = [[] for _ in range(blocks)]
        carrier = None
        for frame in self._fit_frames():
            first, second = self._resampled_pair(
                frame, up_grid, down_grid, rows, keep, sub_target
            )
            if probe is not None:
                # a known distortion, for choosing between the estimators
                second = _displace_rows(second, probe)
            prepared_first = updown._prepare(first)
            prepared_second = updown._prepare(second)

            if estimator == "phase" and carrier is None:
                wave_y, wave_x = self._lattice_wavevector(prepared_first)
                if not wave_y:
                    estimator = "correlation"
                else:
                    grid_y = np.arange(first.shape[0])[:, None]
                    grid_x = np.arange(first.shape[1])[None, :]
                    carrier = np.exp(
                        -2j * np.pi * (wave_y * grid_y + wave_x * grid_x)
                    )
                    self._carrier_period = 1.0 / abs(wave_y)

            for block in range(blocks):
                low_row, high_row = edges[block], edges[block + 1]
                if estimator == "phase" and carrier is not None:
                    measured[block].append(
                        self._phase_displacement(
                            prepared_first[low_row:high_row],
                            prepared_second[low_row:high_row],
                            carrier[low_row:high_row],
                            self._carrier_period,
                        )
                    )
                else:
                    measured[block].append(
                        float(
                            updown._shift(
                                prepared_first[low_row:high_row],
                                prepared_second[low_row:high_row],
                            )[0]
                        )
                    )

        return np.array([np.median(values) for values in measured])

    def _fit_score(self, amplitude, shape):
        """How much of the row dependence is left, in pixels. Lower is better."""
        profile = self.residual_profile(amplitude, shape)
        if profile is None or not np.all(np.isfinite(profile)):
            return np.inf
        # The rigid part is the y-phase's business and is taken out later, so
        # only the spread counts here.
        return float(np.ptp(profile))

    def _choose_estimator(self):
        """Decide how to measure the residual, by trying both on a known case.

        A displacement between two frames can be read from a block correlation
        or, where the surface repeats, from the phase of the repeat. Neither
        wins everywhere: the correlation goes blind when a block is not much
        taller than the repeat, and the phase is meaningless when there is no
        repeat to speak of. Rather than guess from the spectrum, a known
        distortion is imposed on the second frame of each pair and whichever
        estimator recovers it more closely is used for the fit.
        """
        rows_kept = self.residual_profile(0.0, self._neutral_shape(),
                                          estimator="correlation")
        if rows_kept is None:
            return "correlation"

        blocks = rows_kept.size
        centre = np.linspace(0.0, np.pi, blocks + 2)[1:-1]
        truth = np.sin(centre)          # a one pixel bulge in the middle
        errors = {}
        for estimator in ("correlation", "phase"):
            base = self.residual_profile(0.0, self._neutral_shape(),
                                         estimator=estimator)
            with_probe = self.residual_profile(
                0.0, self._neutral_shape(), estimator=estimator, probe=1.0
            )
            if base is None or with_probe is None:
                errors[estimator] = np.inf
                continue
            recovered = np.abs(with_probe - base)
            errors[estimator] = float(np.abs(recovered - truth).mean())

        choice = min(errors, key=lambda key: errors[key])
        log.info(
            "Creep fit: a known one pixel bulge is recovered to %.2f px by the "
            "correlation and %.2f px by the lattice phase; using the %s.",
            errors["correlation"],
            errors["phase"],
            choice,
        )
        return choice

    def _neutral_shape(self):
        return -0.3 if getattr(self.creep_function, "__name__", "") == "root_creep" else 0.3

    def fit_creep_robust(self, shape_guess):
        """Find the creep parameters by measuring what the correction leaves.

        The fit that came with the code minimised the pixel difference between
        the two sweeps with `curve_fit`. That does not work here: the difference
        is dominated by how the surface images differently in the two scan
        directions, the geometry is a small term on top, and the numerical
        gradient of the whole thing is zero to fifteen digits, so the optimiser
        returns the value it started from. Measured on five movies, not one of
        the fifteen attempts per movie ever moved off its starting point.

        What is measured instead is the displacement the correction leaves
        between the sweeps, block by block along the frame, which is what the
        eye sees as the ends of the frame breathing. Both the amplitude and the
        shape are searched, on a coarse grid first and then locally, because the
        landscape is smooth but the gradient is not worth trusting.

        Returns:
            Amplitude and shape.
        """
        self._estimator = self._choose_estimator()
        measured = abs(updown.creep_amplitude(updown.mismatch_profile(self.data)))
        ceiling = max(4.0 * measured, 3.0)
        amplitudes = np.linspace(0.0, ceiling, 7)
        # By name, not by identity: attribute access builds a fresh bound
        # method every time, so `is` never matches.
        if getattr(self.creep_function, "__name__", "") == "root_creep":
            shapes = np.array([-0.05, -0.3, -1.0, -3.0, -8.0])
        else:
            shapes = np.array([0.05, 0.3, 0.7, 1.1, 1.4])

        best = (np.inf, 0.0, float(shapes[0]))
        for amplitude in amplitudes:
            for shape in shapes:
                score = self._fit_score(amplitude, shape)
                if score < best[0]:
                    best = (score, float(amplitude), float(shape))

        # Local refinement of the amplitude, which is the parameter that matters:
        # the shape only sets how the compression is distributed over the ends.
        step = float(amplitudes[1] - amplitudes[0]) / 2.0
        score, amplitude, shape = best
        for _ in range(3):
            for candidate in (amplitude - step, amplitude + step):
                if candidate < 0.0:
                    continue
                trial = self._fit_score(candidate, shape)
                if trial < score:
                    score, amplitude = trial, candidate
            step /= 2.0

        without = self._fit_score(0.0, shape)
        if not score < without:
            log.warning(
                "Creep fit: no amplitude improved on leaving the movie alone "
                "(best %.3f px against %.3f px), so no creep correction is "
                "applied. The up and down frames of this movie disagree in a way "
                "the model does not describe.",
                score,
                without,
            )
            return 0.0, float(shape)

        log.info(
            "Creep fit: amplitude %.2f px, shape %.3f. Residual displacement "
            "between the sweeps %.3f px, against %.3f px uncorrected and %.3f px "
            "for the amplitude the up/down profile suggests (%.2f px).",
            amplitude,
            shape,
            score,
            without,
            self._fit_score(measured, shape),
            measured,
        )
        return float(amplitude), float(shape)

    def _Bezier(self, y, shape, pixels, Bezier_points):
        """Corrects a given y meshgrid for probe creep, using a Bezier curve

        Args:
            y: y meshgrid
            ny: number of pixels in y direction
            shape: 3-tuple describing the shape of the creep in y direction; see wiki for further information
            pixels: excess pixels at top/bottom in up/down frames
            Bezier_points: number of grid points for the numeric creep function

        Returns:
            y_up, y_down: corrected y meshgrids in both scan directions

        """

        ny = y.shape[0]

        # different creep in up and down frames
        y_up = copy(y)
        y_down = copy(y)

        ind_up = int(
            ny * (1.0 - shape[0])
        )  # point at which the y movement becomes linear in upward frames
        ind_down = int(ny * shape[0])

        # creep is approximated by a Bezier curve
        P0_up = np.array([min(y[ind_up, :]), min(y[ind_up, :])])
        P12_up = np.array([y[-1, 0] - pixels, y[-1, 0] - pixels])
        P3_up = np.array([y[-1, 0], y[-1, 0] - pixels])

        P1_up = (1.0 - shape[1]) * P0_up + shape[1] * P12_up
        P2_up = (1.0 - shape[2]) * P12_up + shape[2] * P3_up

        # since Bezier curves have the form (x,y)(t) rather than y(x), one has to numerically find y(x) on a grid
        Bezier_points = int(Bezier_points)
        t = np.linspace(0.0, 1.0, Bezier_points)

        Bezier_up = np.array(
            [
                (1 - t) ** 3 * P0_up[0]
                + 3 * (1 - t) ** 2 * t * P1_up[0]
                + 3 * (1 - t) * t**2 * P2_up[0]
                + t**3 * P3_up[0],
                (1 - t) ** 3 * P0_up[1]
                + 3 * (1 - t) ** 2 * t * P1_up[1]
                + 3 * (1 - t) * t**2 * P2_up[1]
                + t**3 * P3_up[1],
            ]
        )

        j = 0  # initialize j only once to avoid the algorithm searching the entire Bezier curve for each y element
        for i, yi in enumerate(y_up[ind_up:, :]):
            ind = ind_up + i  # actual index of the line in the original array
            if "i" in self.channels:
                fb = (-1) ** (ind % 2)  # 1 for forward, -1 for backward
            if "f" in self.channels:
                fb = -1
            if "b" in self.channels:
                fb = 1
            for ii, yii in enumerate(
                yi[::fb]
            ):  # restore time ordering which is lost by reshaping
                while j < int(Bezier_points - 1):
                    if (
                        Bezier_up[0, j] >= yii
                    ):  # find the index at which the Bezier curve passes yii
                        break
                    j += 1
                w0 = yii - Bezier_up[0, j - 1]
                w1 = Bezier_up[0, j] - yii
                y_up[ind, ::fb][ii] = (
                    w0 * Bezier_up[1, j] + w1 * Bezier_up[1, j - 1]
                ) / (w0 + w1)  # linear interpolation between two Bezier points

        P3_down = np.array([max(y[ind_down, :]), max(y[ind_down, :])])
        P12_down = np.array([y[0, 0] + pixels, y[0, 0] + pixels])
        P0_down = np.array([y[0, 0], y[0, 0] + pixels])

        P2_down = (1.0 - shape[1]) * P3_down + shape[1] * P12_down
        P1_down = (1.0 - shape[2]) * P12_down + shape[2] * P0_down

        Bezier_down = np.array(
            [
                (1 - t) ** 3 * P0_down[0]
                + 3 * (1 - t) ** 2 * t * P1_down[0]
                + 3 * (1 - t) * t**2 * P2_down[0]
                + t**3 * P3_down[0],
                (1 - t) ** 3 * P0_down[1]
                + 3 * (1 - t) ** 2 * t * P1_down[1]
                + 3 * (1 - t) * t**2 * P2_down[1]
                + t**3 * P3_down[1],
            ]
        )

        j = 0
        for i, yi in enumerate(y_down[:ind_down, :]):
            if self.channels.is_interlaced():
                fb = (-1) ** (ind % 2)  # 1 for forward, -1 for backward

            if self.channels.is_forward():
                fb = 1

            if self.channels.is_backward():
                fb = -1

            for ii, yii in enumerate(yi[::fb]):
                while j < int(Bezier_points - 1):
                    if Bezier_down[0, j] >= yii:
                        break
                    j += 1
                w0 = yii - Bezier_down[0, j - 1]
                w1 = Bezier_down[0, j] - yii
                y_down[i, ::fb][ii] = (
                    w0 * Bezier_down[1, j] + w1 * Bezier_down[1, j - 1]
                ) / (w0 + w1)

        # shift everything, such that endpoints match
        y_down -= pixels / 2.0
        y_up += pixels / 2.0

        return y_up, y_down

    def _interpolate_col(self, col, y):
        """Spline interpolation of a single column on a given y grid

        Args:
            col: the column
            y: y grid of the data in col

        Returns:
            col_int: interpolated data on an equidistant grid of the same size as the original grid

        """

        t1 = np.linspace(min(y), max(y), len(y))

        tck = splrep(y, col, s=0)
        col_int = splev(t1, tck, der=0)

        return col_int

    def _interpolate_col_param(self, input1, shape1, shape2, shape3):
        """Given the shape parameters and pixel number for the creep correction,
        compares the interpolated data of a column in an up and a down frame

        Args:
            input1: 5-tuple containing:
                y: not creep corrected y grid of the column
                up: data of the column in the up frame
                down: data of the column in the down frame
                Bezier_points: number of grid points for the numeric creep function
                w: additional weighting of the lines at the upper and lower boundary;
                    weight function is w*y**2/max(y)**2 + 1
                pixels: pixel number for creep correction
            shape1, shape2, shape3: shape parameters for creep correction

        Returns:
            on equidistant grid: difference between interpolated up and down column times the weight function

        """
        y, up, down, Bezier_points, w, pixels = input1

        y_up, y_down = self._Bezier(
            np.reshape(y, (len(y), 1)),
            (shape1, shape2, shape3),
            pixels[0],
            Bezier_points[0],
        )

        up_int = self._interpolate_col(up, y_up)
        down_int = self._interpolate_col(down, y_down)

        weight = w[0] * y**2 / max(y) ** 2 + 1

        return np.reshape((up_int - down_int), -1) * weight

    def fit_creep_bez(
        self,
        col_inds,
        frames=[0, 2],
        w=0.0,
        shape=(0.5, 2.0 / 3.0, 1.0 / 3.0),
        Bezier_points=1000,
        known_input=None,
    ):
        """Fits the shape parameters and pixel number for creep correction to minimize
        the difference between interpolated columns in up and a down frames. The fit
        is adjusted for different movie modes.

        Args:
            fast_movie: FastMovie object
            col_inds: n-tuple of indices of the columns to fit
            frames: n-tuple of indices of the frames to fit
            w: additional weighting of the lines at the upper and lower boundary;
                weight function is w*y**2/max(y)**2 + 1
                defaults to 0
            shape: 3-tuple containing the initial guess for the shape parameters;
                defaults to (.5, 2./3., 1./3.)
            Bezier_points: number of grid points for the numeric creep function

        Returns:
            opt: 4 element array containing the fitted shape parameters and pixel number (in this order)

        """

        if known_input == None:
            log.info("Start bezier creep correction")
            log.info(
                f"Fitting creep correction at columns {col_inds} in images {frames}. Initial guess: shape = ({shape[0]:05.4f}, {shape[1]:05.4f}, {shape[2]:05.4f})."
            )
            log.info(
                f"Additional weight to top and bottom image boundary: {w}. Number of evaluation points for numeric creep function: {Bezier_points}."
            )

            opt_list = []

            # The following is necessary to comply with new version of scipy.
            if self.channels.is_interlaced():
                y_len = len(self.ygridfold[:, col_inds[0]])
            else:
                y_len = len(self.ygridfold[::2, col_inds[0]])
            Bezier_points = [Bezier_points for i in range(y_len)]
            w = [w for i in range(y_len)]
            pixels = [self.pixel_shift for i in range(y_len)]

            for col_ind in col_inds:
                if self.channels.is_interlaced():
                    y = self.ygridfold[:, col_ind]
                elif self.channels.is_forward():
                    y = self.ygridfold[0::2, col_ind]
                elif self.channels.is_backward():
                    y = self.ygridfold[1::2, col_ind]

                for frame in frames:
                    up = self.data[frame, :, col_ind]
                    down = self.data[frame + 1, :, col_ind]
                    opt_i, _ = curve_fit(
                        self._interpolate_col_param,
                        (y, up, down, Bezier_points, w, pixels),
                        np.zeros_like(up),
                        (shape[0], shape[1], shape[2]),
                        bounds=([0.1, 0.0, 0.0], [1.0, 1.0, 1.0]),
                    )

                    opt_list.append(opt_i)

            opt = np.mean(np.array(opt_list), 0)

            log.info(
                f"Optimized creep parameters for later use: shape = ({opt[0]:05.4f}, {opt[1]:05.4f}, {opt[2]:05.4f})"
            )
        else:
            opt = known_input
            pixels = [self.pixel_shift]
            Bezier_points = [Bezier_points]

            log.info(f"Known parameters used = ({opt[0]}, {opt[1]}, {opt[2]})")

        if self.channels.is_interlaced():
            grid_up, grid_down = self._Bezier(
                self.ygridfold, opt, pixels[0], Bezier_points[0]
            )

        elif self.channels.is_forward():
            ygrid = self.ygridfold[0::2]
            up, down = self._Bezier(ygrid, opt, pixels[0], Bezier_points[0])
            grid_up = np.zeros((np.shape(up)[0] * 2, np.shape(up)[1]))
            grid_down = grid_up.copy()
            grid_up[0::2] = up[:]
            grid_down[1::2] = down[:]

        elif self.channels.is_backward():
            ygrid = self.ygridfold[1::2]
            up, down = self._Bezier(ygrid, opt, pixels[0], Bezier_points[0])
            grid_up = np.zeros((np.shape(up)[0] * 2, np.shape(up)[1]))
            grid_down = grid_up.copy()
            grid_up[1::2] = up[:]
            grid_down[0::2] = down[:]

        return opt, (grid_up, grid_down)
