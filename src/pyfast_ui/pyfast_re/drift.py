"""
First try at movie stabilisation with
OpenCV. Parts of the code are inspired by:
learnopencv.com/video-stabilisation-using-point-feature-matching-in-opencv
"""

from __future__ import annotations
from enum import Enum
from typing import TYPE_CHECKING, final
import matplotlib.pyplot as plt
import numpy as np
from pystackreg import StackReg
from scipy.ndimage import convolve
from scipy.signal import correlate, medfilt
from skimage.transform import resize

from pyfast_ui.pyfast_re.data_mode import DataMode

if TYPE_CHECKING:
    from pyfast_ui.pyfast_re.fast_movie import FastMovie


class DriftMode(Enum):
    FULL = "full"
    COMMON = "common"


class KnownDriftType(Enum):
    INTEGRATED = "integrated"
    SEQUENTIAL = "sequential"


class StackRegReferenceType(Enum):
    PREVIOUS = "previous"
    FIRST = "first"
    MEAN = "mean"


@final
class Drift:
    """
    Initialise Drift class with Fast movie instance to then
    drift correct the movie data.

    Args:
        FastmovieInstance: FastMovie object
        stepsize: integer, the difference between frames that are correlated
        corrspeed: int, the difference between two correlation windows
        show_path: Parameter, if True rare and filter drift path are plotted.
        boxcar: Parameter of the boxcar filter that applied to the
            drift path. Set to 0 if no boxcar filter should be applied
        median_filter: Paramter to decide if the drift path should be smoothed
            via a median_filter of kernel size 3
    """

    def __init__(
        self,
        fast_movie: FastMovie,
        stepsize: int = 40,
        corrspeed: int = 1,
        show_path: bool = False,
        boxcar: int = 50,
        median_filter: bool = True,
    ):
        if fast_movie.mode != DataMode.MOVIE:
            raise ValueError(f"`FastMovie` instance must be in mode {DataMode.MOVIE}")
        if fast_movie.channels is None:
            raise ValueError("`FastMovie.channels must be set`")

        self.data = fast_movie.data
        self.file = fast_movie.filename.replace(".h5", ".drift.txt")
        # self.processing_log = fast_movie.processing_log
        self.channels = fast_movie.channels
        self.stepsize = stepsize
        self.corrspeed = corrspeed
        self.n_frames = np.shape(self.data)[0]
        self.img_width = np.shape(self.data)[2]
        self.img_height = np.shape(self.data)[1]
        self.boxcar = boxcar
        self.median_filter = median_filter

        if self.img_width > self.img_height:
            self.im_size = 2 ** (int(np.log2(self.img_width)) + 1)
        else:
            self.im_size = 2 ** (int(np.log2(self.img_height)) + 1)

        self.convdims = (self.im_size * 2 - 1, self.im_size * 2 - 1)
        self.transformations = np.zeros((2, self.n_frames))
        self.integrated_trans = None
        self.show_path = show_path

        # if self.stepsize is None:
        #     self.stepsize = int(self.n_frames / 3)

    def correct_correlation(
        self,
        mode: DriftMode = DriftMode.FULL,
    ):
        self._get_drift_correlation()
        self._filter_drift()
        self._write_drift()

        match mode:
            case DriftMode.FULL:
                return self._adjust_movie_buffered(), self.integrated_trans
            case DriftMode.COMMON:
                return self._adjust_movie_common(), self.integrated_trans

    def correct_stackreg(
        self,
        mode: DriftMode = DriftMode.FULL,
        stackreg_reference: StackRegReferenceType = StackRegReferenceType.PREVIOUS,
    ):
        self._get_drift_stackreg(stackreg_reference)
        self._filter_drift()
        self._write_drift()

        match mode:
            case DriftMode.FULL:
                return self._adjust_movie_buffered(), self.integrated_trans
            case DriftMode.COMMON:
                return self._adjust_movie_common(), self.integrated_trans

    def correct_known(
        self,
        mode: DriftMode = DriftMode.FULL,
        known_drift_type: KnownDriftType = KnownDriftType.INTEGRATED,
    ):
        driftfile = self.file.replace(".h5", ".drift.txt")

        match known_drift_type:
            case KnownDriftType.INTEGRATED:
                self.integrated_trans = np.loadtxt(driftfile).T[0:2, :]
                # self.processing_log.info("Known drift used: {}".format(known_drift_type))
            case KnownDriftType.SEQUENTIAL:
                self.transformations = np.loadtxt(driftfile).T[2:4, :]
                self.integrated_trans = np.cumsum(self.transformations, axis=1)
                self._write_drift()
                # self.processing_log.info("Known drift used: {}".format(known_drift))

        match mode:
            case DriftMode.FULL:
                return self._adjust_movie_buffered(), self.integrated_trans
            case DriftMode.COMMON:
                return self._adjust_movie_common(), self.integrated_trans

    def _get_drift_correlation(self):
        """Calculation of the drift by fft correlation."""
        movie = np.zeros((self.n_frames, self.im_size, self.im_size))
        hamm = np.sqrt(
            np.outer(np.hamming(self.img_height), np.hamming(self.img_width))
        )
        for i in range(self.n_frames):
            imag = self.data[i, :, :].copy()
            imag /= imag.std()
            imag -= imag.mean()
            imag = hamm * imag
            movie[i, :, :] = resize(
                imag, (self.im_size, self.im_size), anti_aliasing=True, order=0
            )
        for i in range(self.n_frames):
            try:
                fftd = correlate(
                    movie[self.corrspeed * i, :, :],
                    movie[self.corrspeed * i + self.stepsize, :, :],
                    method="fft",
                )
                maxind = np.argmax(fftd)
                indices = np.unravel_index(maxind, self.convdims)
                effektive_shift = np.asarray(
                    [
                        [(-(self.im_size - 1) + indices[0]) / self.stepsize],
                        [(indices[1] - (self.im_size - 1)) / self.stepsize],
                    ]
                )
                self.transformations[:, i] = effektive_shift.T
            except Exception:
                pass
        # print("last found correlation indices are {}".format(indices))

    def _get_drift_stackreg(self, reference):
        stackreg = StackReg(StackReg.TRANSLATION)
        transformation_matrices = stackreg.register_stack(
            self.data, reference=reference.value
        )
        x_path_integrated = []
        y_path_integrated = []
        for mat in transformation_matrices:
            x_path_integrated.append(-mat[0, 2])
            y_path_integrated.append(-mat[1, 2])

        # self.integrated_trans = np.array([y_path_integrated, x_path_integrated])
        x_path = np.array(x_path_integrated)
        x_path = np.diff(x_path, prepend=0)
        y_path = np.array(y_path_integrated)
        y_path = np.diff(y_path, prepend=0)

        self.transformations = np.stack((y_path, x_path))

    def _filter_drift(self):
        """
        smooth and filter drift path
        """
        boxwidth = self.boxcar
        boxcar = np.ones((1, boxwidth)) / boxwidth
        boxcar = boxcar[0, :]

        if self.median_filter:
            self.transformations[0, :] = medfilt(self.transformations[0, :], 3)
            self.transformations[1, :] = medfilt(self.transformations[1, :], 3)

        self.integrated_trans = np.cumsum(self.transformations, axis=1)
        # linear extrapolation
        pos = np.linspace(0, self.n_frames - 1, self.n_frames)
        k1, d1 = np.polyfit(
            pos[: -self.stepsize], self.integrated_trans[0, : -self.stepsize], 1
        )
        k2, d2 = np.polyfit(
            pos[: -self.stepsize], self.integrated_trans[1, : -self.stepsize], 1
        )
        self.integrated_trans[0, -self.stepsize :] = d1 + k1 * pos[-self.stepsize :]
        self.integrated_trans[1, -self.stepsize :] = d2 + k2 * pos[-self.stepsize :]

        if self.boxcar != 0:
            # self.processing_log.info( "Boxcar filter used with boxsize: {}".format(boxwidth))
            transformations_conv = np.zeros((2, self.n_frames))
            transformations_conv[0, :] = convolve(self.integrated_trans[0], boxcar)
            transformations_conv[1, :] = convolve(self.integrated_trans[1], boxcar)
            self.integrated_trans = transformations_conv

    def _write_drift(self):
        """
        Writes a drift.txt file
        """
        with open(self.file, "w") as fileobject:
            fileobject.write(
                "# {0:>10}   {1:>12}  {2:>12}  {3:>12} \n".format(
                    "y integrated", "x integrated", "y sequential", "x sequential"
                )
            )
            for i in range(np.shape(self.transformations)[1]):
                fileobject.write(
                    "{0:>14.5f}   {1:>12.5f}  {2:>12.5f}  {3:>12.5f} \n".format(
                        self.integrated_trans[0, i],
                        self.integrated_trans[1, i],
                        self.transformations[0, i],
                        self.transformations[1, i],
                    )
                )

    def _adjust_movie_buffered(self):
        """embed movie frames into buffered background to
        move freely according to drift path. The image ration
        is changed back for interlace movies (2:1) to fit the
        overall system architecture"""
        maxy, maxx = np.max(self.integrated_trans, 1)
        miny, minx = np.min(self.integrated_trans, 1)
        buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
        # print("Buffer values are {} in x and {} in y.".format(buffx, buffy))
        ## This is to see effect of scaling

        if self.channels.is_interlaced():
            self.rescale_width = int(self.im_size / 2)
            maxx = maxx / 2
            minx = minx / 2
        else:
            self.rescale_width = self.im_size

        buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

        corr_movie = np.zeros(
            (self.n_frames, self.im_size + int(buffy), self.rescale_width + int(buffx)),
            dtype=np.float32,
        )
        for i in range(self.n_frames):
            shift1, shift2 = self.integrated_trans[:, i]
            shift1 = int(np.round(shift1))

            if self.channels.is_interlaced():
                shift2 = int(np.round(shift2) / 2)
            else:
                shift2 = int(np.round(shift2))
            # possibly there is a +1 in the i for the frame to be taken.
            corr_movie[
                i,
                int(abs(miny)) + 1 + shift1 : int(abs(miny))
                + 1
                + self.im_size
                + shift1,
                int(abs(minx)) + 1 + shift2 : int(abs(minx))
                + 1
                + self.rescale_width
                + shift2,
            ] = resize(
                self.data[i, :, :],
                (self.im_size, self.rescale_width),
                anti_aliasing=True,
                order=3,
            )

        print("drift correction finished")
        return corr_movie

    def _adjust_movie_common(self):
        """cut out section from movie frames, which stays constant during
        the entire movie."""
        maxy, maxx = np.max(self.integrated_trans, 1)
        miny, minx = np.min(self.integrated_trans, 1)
        buffy = int(np.round(np.abs(maxy) + np.abs(miny))) + 1
        # print(buffx, buffy)
        ## This is to see effect of scaling

        if self.channels.is_interlaced():
            self.rescale_width = int(self.im_size / 2)
            maxx = maxx / 2
            minx = minx / 2
        else:
            self.rescale_width = self.im_size

        buffx = int(np.round(np.abs(maxx) + np.abs(minx))) + 1

        corr_movie = np.zeros(
            (self.n_frames, self.im_size - int(buffy), self.rescale_width - int(buffx)),
            dtype=np.float32,
        )
        for i in range(self.n_frames):
            shift1, shift2 = self.integrated_trans[:, -i + 1]
            shift1 = int(np.round(shift1))

            if self.channels.is_interlaced():
                shift2 = int(np.round(shift2) / 2)
            else:
                shift2 = int(np.round(shift2))
            # possibly there is a +1 in the i for the frame to be taken.

            corr_movie[i, :, :] = resize(
                self.data[i, :, :],
                (self.im_size, self.rescale_width),
                anti_aliasing=True,
                order=4,
            )[
                int(abs(miny)) + 1 + shift1 : int(abs(miny))
                + 1
                + self.im_size
                - int(buffy)
                + shift1,
                int(abs(minx)) + 1 + shift2 : int(abs(minx))
                + 1
                + self.rescale_width
                - int(buffx)
                + shift2,
            ]

        print("drift correction finished")
        return corr_movie


def meanfilter(data, kernel=3):
    """
    possible meanfilter.

    Args:
        data: 1D array
        kernel: Size of values that are filtered

    Returns:
        filtered: adjusted array
    """
    filtered = np.zeros(len(data))
    if kernel % 2 == 0:
        kernel += 1
    for i in range(len(data)):
        down = i - int(kernel / 2)
        up = i + int(kernel / 2) + 1
        if down < 0:
            down = 0
        if up > len(data):
            up = int(len(data))
        filtered[i] = np.mean(data[down:up])
    return filtered
