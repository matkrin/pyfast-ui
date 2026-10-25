"""How an up frame and the following down frame disagree, row by row.

Two different errors show up as a displacement between the up and the down
sweep of one image, and they used to be measured as one number:

- the y-phase, a constant offset of the whole frame, from cutting the series at
  the wrong sample
- creep, the delayed response of the y piezo after a turning point

They can be told apart by their shape. The two sweeps share their turning
points, so at the first and the last line of a frame the up and the down sweep
are at the same height and only the y-phase remains. Between the turning points
the piezo lags, so the lines in the middle carry the y-phase plus the creep.
Measuring the offset in the middle alone, as the creep correction used to,
returns the sum and cannot separate them: a movie with a large y-phase then
gets a creep correction that bends the ends of every frame to compensate an
offset that is not creep.

The horizontal offset is measured alongside. It has no creep term, so whatever
appears there is a rigid displacement between the two sweeps.
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift
from skimage.registration import phase_cross_correlation

log = logging.getLogger(__name__)


BLOCKS = 8
"""Row blocks per frame. Eight leaves each block wide enough to register on a
242 row frame and still resolves the difference between the ends and the
middle."""

PAIRS = 24
"""Frame pairs averaged over. The offset is a property of the scan, so it is
the same in every pair, and this only fights the noise."""

SMOOTH = 0.6
"""Gaussian smoothing before the correlation, in pixels. Enough to take the
edge off the pixel noise, small enough to leave the features that carry the
displacement.

A band pass was tried here, as in the drift estimation, and had to be dropped:
combined with the phase normalisation of the correlation it attenuates small
displacements badly. Measured on a frame displaced by a known amount, a band
passed 0.5 px came back as 0.12 to 0.24 px, while the lightly smoothed frame
gave 0.46 px."""


def _prepare(frame: NDArray[np.float32]) -> NDArray[np.float64]:
    frame = frame.astype(np.float64)
    frame = frame - frame.mean()
    return gaussian_filter(frame, SMOOTH)


def _shift(a: NDArray[np.float64], b: NDArray[np.float64]) -> NDArray[np.float64]:
    """Sub-pixel displacement of b relative to a.

    The plain correlation is used, not the masked one: on a residual of a
    pixel or two the bias of the plain variant towards zero does not matter,
    and the masked variant resolves whole pixels only.
    """
    result = phase_cross_correlation(a, b, upsample_factor=50)  # pyright: ignore[reportUnknownVariableType]
    value = result[0] if isinstance(result, tuple) else result
    return np.asarray(value, dtype=float)


COARSE_LIMIT = 0.25
"""Largest coarse displacement considered, as a fraction of the frame size.

A y-phase of a few tens of rows has to be within reach; anything beyond a
quarter of the frame is not a y-phase but a failed measurement, and letting the
search go there only invites one.
"""

COARSE_TIE = 0.02
"""How much worse than the best a candidate may score and still be preferred if
it displaces the frame less, as a fraction of the best score.

On a periodic surface every lattice period is a candidate: shifting one frame by
a whole period puts the atoms back on top of each other and scores almost as
well as the true offset. The alternatives are then near ties, and the smallest
of them is the physical one, because a larger displacement would have to show up
as a y-phase of that size as well.
"""


def _best_offset(
    a: NDArray[np.float64], b: NDArray[np.float64], axis: int, limit: float
) -> float:
    """Whole-pixel displacement of b relative to a along one axis.

    A plain normalised correlation over the overlap, evaluated at every
    candidate offset within `limit` and taking the best. The phase correlation
    used here before whitens the spectrum, which on a lattice movie raises the
    aliases of the true peak to the height of the peak itself: after an FFT
    filter has taken out the slow, non-periodic part of the image, on
    FS_250903_037 it locked onto an offset two lattice periods away, +15 px
    instead of -8 px. The creep step reads that number as the constant up/down
    offset, sizes its correction by it and then cuts the rows it has bent, so a
    filter setting that does not move the image by a hundredth of a pixel cost
    fourteen rows at each end of every frame.
    """
    length = a.shape[axis]
    span = min(max(int(round(limit * length)), 1), length - 2)
    a = np.moveaxis(a, axis, 0)
    b = np.moveaxis(b, axis, 0)

    offsets = np.arange(-span, span + 1)
    scores = np.zeros(offsets.size)
    for index, offset in enumerate(offsets):
        if offset >= 0:
            first, second = a[offset:], b[: length - offset]
        else:
            first, second = a[: length + offset], b[-offset:]
        first = first - first.mean()
        second = second - second.mean()
        norm = float(np.sqrt((first * first).sum() * (second * second).sum()))
        if norm > 0.0:
            scores[index] = float((first * second).sum()) / norm

    best = scores.max()
    if best <= 0.0:
        return 0.0

    close = offsets[scores >= best * (1.0 - COARSE_TIE)]
    return float(close[np.argmin(np.abs(close))])


def _coarse_shift(
    a: NDArray[np.float64], b: NDArray[np.float64], limit: float = COARSE_LIMIT
) -> NDArray[np.float64]:
    """Whole-pixel displacement of the frame as a whole.

    A y-phase of a few tens of rows is larger than one row block, so the blocks
    have to be brought together before their residual can be measured. The two
    axes are searched one after the other; they are all but independent here,
    since the y-phase moves the frame vertically and leaves the columns alone.
    """
    return np.array(
        [
            _best_offset(a, b, axis=0, limit=limit),
            _best_offset(a, b, axis=1, limit=limit),
        ]
    )


def mismatch_profile(
    data: NDArray[np.float32],
    blocks: int = BLOCKS,
    pairs: int = PAIRS,
    coarse_limit: float = COARSE_LIMIT,
) -> NDArray[np.float64]:
    """Displacement between up and down frames, per row block.

    Neighbouring frames are one up and one down, in alternating order, so the
    displacement between them changes sign from pair to pair. Half the
    difference between the two orders is the part that belongs to the up/down
    asymmetry; everything else, drift above all, is common to both and drops
    out.

    Args:
        data: The movie, in movie mode.
        blocks: Row blocks per frame.
        pairs: How many frame pairs to measure.
        coarse_limit: Largest displacement to consider, as a fraction of the
            frame height. Worth narrowing when the movie is known to be close to
            aligned already, as after a creep correction: a wide search on a
            lattice movie can prefer an offset a whole period out.

    Returns:
        Array of shape (blocks, 2) with the (y, x) displacement of each row
        block, in pixels, from the top of the frame to the bottom.
    """
    num_frames = len(data)
    if num_frames < 3:
        return np.zeros((blocks, 2))

    edges = np.linspace(0, data.shape[1], blocks + 1).astype(int)
    measured: dict[int, list[list[NDArray[np.float64]]]] = {
        0: [[] for _ in range(blocks)],
        1: [[] for _ in range(blocks)],
    }

    # Both parities have to be sampled: one gives the up to down displacement,
    # the other down to up, and the difference of the two is the measurement.
    # Spreading indices evenly does not guarantee that, they can all come out
    # even, in which case nothing is measured at all.
    half = max(pairs // 2, 1)
    starts = np.linspace(0, num_frames - 3, half).astype(int)
    indices = np.unique(
        np.concatenate([starts - (starts % 2), starts - (starts % 2) + 1])
    )
    indices = indices[(indices >= 0) & (indices < num_frames - 1)]

    # Coarse first, so that the blocks overlap at all, then the residual per
    # block. A y-phase of thirty rows exceeds the height of one block, and a
    # block correlated over that distance returns nothing usable.
    #
    # The coarse offset is a property of the scan and therefore the same in
    # every pair of one parity, so it is taken over the movie rather than pair
    # by pair. On a lattice movie a single pair can prefer an offset a whole
    # period out by a percent or two of correlation, and a pair started that far
    # from its neighbours takes its whole profile with it; measured on
    # FS_251128_026 after the creep correction, pair by pair it read +7.7 px
    # where the movie is in fact +0.3 px out.
    prepared = {index: _prepare(data[index]) for index in indices}
    prepared.update(
        {index + 1: _prepare(data[index + 1]) for index in indices}
    )
    coarse_by_parity: dict[int, NDArray[np.float64]] = {}
    for parity in (0, 1):
        seen = [
            _coarse_shift(prepared[index], prepared[index + 1], coarse_limit)
            for index in indices
            if index % 2 == parity
        ]
        coarse_by_parity[parity] = (
            np.median(np.stack(seen), axis=0) if seen else np.zeros(2)
        )

    for index in indices:
        first, second = prepared[index], prepared[index + 1]
        coarse = coarse_by_parity[index % 2]
        aligned = ndshift(second, coarse, order=1, mode="nearest")
        for block in range(blocks):
            low, high = edges[block], edges[block + 1]
            if high - low < 8:
                continue
            residual = _shift(first[low:high], aligned[low:high])
            measured[index % 2][block].append(coarse + residual)

    profile = np.zeros((blocks, 2))
    for block in range(blocks):
        even, odd = measured[0][block], measured[1][block]
        if not even or not odd:
            continue
        profile[block] = 0.5 * (
            np.median(np.stack(even), axis=0) - np.median(np.stack(odd), axis=0)
        )

    return profile


def rigid_offset(profile: NDArray[np.float64]) -> NDArray[np.float64]:
    """The offset to take out of an already creep-corrected movie, in pixels.

    Not the same estimator as `constant_offset`, and for a reason. Before the
    creep correction the turning points are the reference: the two sweeps meet
    there, so the ends carry the y-phase alone and the ends are what the
    y-phase should read.

    Afterwards they are not. The creep model measures its own correction
    against the ends and recentres the grid, so a movie whose bulk it has
    straightened comes out with a residual offset spread over the whole frame.
    Reading the ends again then anchors on the one place the model has already
    used and pushes the residual back into the middle of the frame, which is
    where it is most visible. The median over all blocks leaves the residual
    centred instead: half of what is left above the middle, half below.
    """
    if len(profile) < 3:
        return np.zeros(2)

    return np.median(profile, axis=0)


def constant_offset(profile: NDArray[np.float64]) -> NDArray[np.float64]:
    """The part of the mismatch that is the same in every row, in pixels.

    Vertically this is read off the first and the last block, where the two
    sweeps meet at their turning points and creep has decayed; the blocks in
    between carry creep on top and would inflate it.

    Horizontally there is no creep term, the y sweep does not move the tip
    sideways, so every block measures the same quantity and the median over all
    of them is used. Two blocks would be needlessly noisy.
    """
    if len(profile) < 3:
        return np.zeros(2)

    return np.array(
        [0.5 * (profile[0, 0] + profile[-1, 0]), float(np.median(profile[:, 1]))]
    )


def profile_noise(profile: NDArray[np.float64]) -> float:
    """How much blocks that should agree disagree, in pixels.

    The two sweeps enter a frame from opposite ends, so the creep they leave
    behind is symmetric about the middle: the block a given distance from the
    top must agree with the block the same distance from the bottom. How far
    they disagree is a measure of everything that is not creep, noise above all,
    and on a movie with few distinct features, a scan zoomed far in for
    instance, that reaches a couple of pixels. A profile that runs uphill rather
    than bulging in the middle is caught by the same test.
    """
    if len(profile) < 4:
        return 0.0

    return float(np.abs(profile[:, 0] - profile[::-1, 0]).mean() / 2.0)


def creep_amplitude(profile: NDArray[np.float64]) -> float:
    """How far the middle of a frame is displaced beyond the constant offset.

    This is the quantity the creep correction should be sized by. It is
    unaffected by the y-phase, which adds the same amount to every block.

    Zero when the profile is too noisy to say. The creep correction takes this
    number as the amplitude to remove, so noise here is not harmless: it warps
    every frame by half of whatever is passed in.
    """
    if len(profile) < 4:
        return 0.0

    middle = profile[len(profile) // 2 - 1 : len(profile) // 2 + 1, 0].mean()
    amplitude = float(middle - constant_offset(profile)[0])

    noise = profile_noise(profile)
    if abs(amplitude) <= noise:
        log.info(
            "Creep amplitude %.2f px is within the scatter of the measurement "
            "(%.2f px), so it is not determined; treating it as zero.",
            amplitude,
            noise,
        )
        return 0.0

    return amplitude


def describe(profile: NDArray[np.float64]) -> str:
    """One line for the log."""
    constant = constant_offset(profile)
    return (
        f"up/down mismatch: constant part y {constant[0]:+.2f} px, "
        f"x {constant[1]:+.2f} px; creep part y {creep_amplitude(profile):+.2f} px; "
        "profile "
        + " ".join(f"{value:+.2f}" for value in profile[:, 0])
    )
