from pyfast_ui.pyfast_re.drift import Drift, DriftMode, DriftNonscaling
from pyfast_ui.pyfast_re.fast_movie import FastMovie, FftFilterParams
from pyfast_ui.pyfast_re.channels import Channels
import matplotlib.pyplot as plt
import timeit

h5_file = "/home/matthias/Documents/fast_movies/FS_240715_035.h5"
# h5_file = "/home/matthias/github/pyfastspm/examples/F20190424_1.h5"
# h5_file = "/home/matthias/github/pyfastspm/examples/20141003_24.h5"

# h5_file = "/Users/matthias/github/pyfastspm/examples/F20190424_1.h5"

def correct_drift_scaling(fast_movie):
    driftmode = DriftMode("common")
    drift = Drift(
        fast_movie, stepsize=20, boxcar=5, median_filter=True
    )
    _, _ = drift.correct_correlation(driftmode)    

def correct_drift_noscaling(fast_movie):
    driftmode = DriftMode("common")
    drift = DriftNonscaling(
        fast_movie, stepsize=20, boxcar=5, median_filter=True
    )
    _, _ = drift.correct_correlation(driftmode)    


for channel in [c.value for c in Channels]:
    fast_movie = FastMovie(h5_file, y_phase=0)
    metadata = fast_movie.metadata

    fast_movie.correct_phase(
        auto_x_phase=True,
        frame_index_to_correlate=0,
        additional_x_phase=0,
        manual_y_phase=0,
    )

    fft_filter_config = FftFilterParams(
        filter_x=True,
        filter_y=True,
        filter_x_overtones=True,
        filter_high_pass=True,
        filter_pump=True,
        filter_noise=False,
    )
    fast_movie.fft_filter(
        fft_filter_config,
        num_x_overtones=10,
        filter_broadness=0,
        num_pump_overtones=3,
        pump_freqs=[1500, 1000],
        high_pass_params=(1000.0, 600.0),
    )

    # fast_movie.to_movie_mode("udi")
    # fast_movie.to_movie_mode("uf")
    # fast_movie.to_movie_mode("df")
    # fast_movie.to_movie_mode("db")
    # fast_movie.to_movie_mode("ub")
    fast_movie.to_movie_mode(channel)

    fast_movie.correct_creep_non_bezier(
        creep_mode="sin",
        initial_guess=0.3,
        guess_ind=0.2,
        known_params=None,
    )
    fast_movie.interpolate()

    # correct_drift_noscaling(fast_movie)
    # break

    # time_scaling = timeit.timeit(
    #     lambda: correct_drift_scaling(fast_movie),
    #     number=5,
    # )
    # time_nonscaling = timeit.timeit(
    #     lambda: correct_drift_noscaling(fast_movie),
    #     number=5,
    # )

    # print(f"{time_scaling=}")
    # print(f"{time_nonscaling=}")

    # fast_movie.correct_drift_correlation(
    #     mode="common",
    #     stepsize=20,
    #     boxcar=50,
    #     median_filter=True,
    # )
    # fast_movie.plot_drift_path()

    # fast_movie.correct_drift_stackreg(
    #     drifttype="full",
    #     stackreg_reference="previous",
    #     boxcar=50,
    #     median_filter = True,
    # )

    
    driftmode = DriftMode("common")
    drift = DriftNonscaling(
        fast_movie, stepsize=20, boxcar=5, median_filter=True
    )
    fast_movie.data, _ = drift.correct_correlation(driftmode)    

    # fast_movie.align_rows()

    if "i" in channel:
        fast_movie.rescale((1, 2))
    else:
        fast_movie.rescale((2, 2))

    # fast_movie.crop((50, 120), (50, 120))
    # fast_movie.cut((20, 50))
    # fast_movie.algin_rows("median")

    fast_movie.export_mp4(fps_factor=2)
    # fast_movie.export_tiff()
    #
    # fast_movie.export_frames_image("png", (0, 3), color_map="bone")
    # fast_movie.export_frames_txt((0, 3))
    # fast_movie.export_frames_gwy("images", (0, 5))
    # fast_movie.export_frames_gwy("volume", (0, 5))

    break
