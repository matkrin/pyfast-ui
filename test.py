from pyfast_ui.pyfast_re.fast_movie import FastMovie, FftFilterConfig
from pyfast_ui.pyfast_re.channels import Channels

# h5_file = "/home/matthias/Documents/fast_movies/FS_240715_035.h5"
h5_file = "/home/matthias/github/pyfastspm/examples/F20190424_1.h5"

for channel in [c.value for c in Channels]:
    fast_movie = FastMovie(h5_file)
    metadata = fast_movie.metadata

    fast_movie.correct_phase(auto_x_phase=True, frame_index_to_correlate=0, additional_x_phase=0)

    fft_filter_config = FftFilterConfig(
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
    fast_movie.to_movie_mode(channel)
    fast_movie.correct_creep_non_bezier(
        creep_mode="sin",
        initial_guess=0.3,
        guess_ind=0.2,
        known_params=None,
    )
    fast_movie.interpolate()

    fast_movie.correct_drift_correlation(
        mode="full",
        stepsize=20,
        boxcar=50,
        median_filter = True,
    )

    # fast_movie.correct_drift_stackreg(
    #     drifttype="full",
    #     stackreg_reference="previous",
    #     boxcar=50,
    #     median_filter = True,
    # )
    
    fast_movie.align_rows()

    if "i" in channel:
        fast_movie.rescale((1, 2))

    # fast_movie.export_mp4(fps_factor=2)
    # fast_movie.export_frames_image("png", (0, 3), color_map="bone")
    # fast_movie.export_tiff()
    fast_movie.export_frames_gwy("volume", (0, 5))
    break
