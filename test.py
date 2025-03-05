from pyfast_ui.pyfast_re.drift import Drift, DriftMode, DriftNoScaling
from pyfast_ui.pyfast_re.fast_movie import FastMovie, FftFilterParams
from pyfast_ui.pyfast_re.channels import Channels
import matplotlib.pyplot as plt
import time

import scipy
import numpy as np
import skimage

h5_file = "/home/matthias/Documents/fast_movies/FS_240715_035.h5"
# h5_file = "/home/matthias/github/pyfastspm/examples/F20190424_1.h5"
# h5_file = "/home/matthias/github/pyfastspm/examples/20141003_24.h5"

# h5_file = "/Users/matthias/github/pyfastspm/examples/F20190424_1.h5"


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
    
    # frame1 = fast_movie.data[0]
    # frame2 = fast_movie.data[1]
    # print(f"{frame1.shape=}")
    # print("-" * 80)

    # start = time.perf_counter()
    # correlation = scipy.signal.correlate(
    #     frame1,
    #     frame2,
    #     mode="same",  # keeps dimensions same
    #     method="fft"
    # )
    # end = time.perf_counter()
    # indices_correlation = np.unravel_index(correlation.argmax(), correlation.shape)
    # print(f"{correlation.shape=}")
    # print(f"{indices_correlation}")
    # print("Corr Took: ", end - start)
    # print("-" * 80)

    # start2d = time.perf_counter()
    # correlation2d = scipy.signal.correlate2d(
    #     frame1,
    #     frame2,
    #     mode="same",  # keeps dimensions same
    #     boundary="fill"
    # )
    # end2d = time.perf_counter()
    # indices_correlation2d = np.unravel_index(np.argmax(correlation2d), correlation2d.shape)
    # print(f"{correlation2d.shape=}")
    # print(f"{indices_correlation2d=}")
    # print("Corr2d Took: ", end2d - start2d)
    # print("-" * 80)
    # print("diff", 1/((end - start)/ (end2d - start2d)))



    # print("-" * 80)
    # start_phase = time.perf_counter()
    # print(f"{skimage.registration.phase_cross_correlation(frame1, frame2)}")
    # end_phase = time.perf_counter()
    # print("Took: ", end_phase- start_phase)
    # print("-" * 80)

    # print("=" * 80)
    # print("=" * 80)
    
    # frame_w = frame1.shape[1]
    # frame_h = frame1.shape[0]
    # if frame_w > frame_h:
    #     frame_size = 2 ** (int(np.log2(frame_w)) + 1)
    # else:
    #     frame_size = 2 ** (int(np.log2(frame_h)) + 1)

    # start_resizing = time.perf_counter()
    # frame1_sized = skimage.transform.resize(frame1, (frame_size, frame_size))
    # frame2_sized = skimage.transform.resize(frame1, (frame_size, frame_size))
    # end_resizing = time.perf_counter()
    # print("Resizing took: ", end_resizing - start_resizing)

    # start_sized = time.perf_counter()
    # correlation = scipy.signal.correlate(
    #     frame1_sized,
    #     frame2_sized,
    #     mode="same",  # keeps dimensions same
    #     method="fft"
    # )
    # end_sized = time.perf_counter()
    # indices_correlation = np.unravel_index(correlation.argmax(), correlation.shape)
    # print(f"{correlation.shape=}")
    # print(f"{indices_correlation}")
    # print("Corr Sized Took: ", end_sized - start_sized)
    # print("-" * 80)

    # start2d = time.perf_counter()
    # correlation2d = scipy.signal.correlate2d(
    #     frame1_sized,
    #     frame2_sized,
    #     mode="same",  # keeps dimensions same
    #     boundary="fill"
    # )
    # end2d = time.perf_counter()
    # indices_correlation2d = np.unravel_index(np.argmax(correlation2d), correlation2d.shape)
    # print(f"{correlation2d.shape=}")
    # print(f"{indices_correlation2d=}")
    # print("Corr Sized 2d Took: ", end2d - start2d)
    # print("-" * 80)
    # print("diff", 1/((end - start)/ (end2d - start2d)))


    # print("*" * 80)
    # print("diff", 1/((end - start)/ (end_sized - start_sized)))
    # print("*" * 80)
    
     
    # fig, axs = plt.subplots(1, 4)
    # axs[0].imshow(frame1)
    # axs[1].imshow(frame2)
    # axs[2].imshow(correlation)
    # # axs[3].imshow(correlation2d)
    # plt.show()

    stepsize = 10
    corrspeed = 1
    boxcar = 50
    median_filter = True

    start = time.perf_counter()
    drift = Drift(fast_movie, stepsize=stepsize, corrspeed=corrspeed, boxcar=boxcar, median_filter=median_filter)
    data, path = drift.correct_correlation(DriftMode.FULL, )
    end = time.perf_counter()
    # fast_movie.data = data
    
    start_no_scaling = time.perf_counter()
    drift_no_scaling = DriftNoScaling(fast_movie, stepsize=stepsize, corrspeed=corrspeed, boxcar=boxcar, median_filter=median_filter)
    data_no_scaling, path_no_scaling = drift_no_scaling.correct_correlation(DriftMode.FULL, )
    end_no_scaling = time.perf_counter()
    # fast_movie.data = data_no_scaling

    start_phase_cc = time.perf_counter()
    drift_phase_cc = DriftNoScaling(fast_movie, stepsize=stepsize, corrspeed=corrspeed, boxcar=boxcar, median_filter=median_filter)
    data_phase_cc, path_phase_cc = drift_phase_cc.correct_phase_cross_correlation(DriftMode.FULL, )
    end_phase_cc = time.perf_counter()
    fast_movie.data = data_phase_cc
    ###
    
    print("=" * 80)
    print("Time scaling: ", end - start)
    print("Time NO scaling: ", end_no_scaling - start_no_scaling)
    print("Time phase cross corr: ", end_phase_cc - start_phase_cc)
    print("Paths same with and without scaling: ", np.all(path == path_no_scaling))
    # print("Data same with and without scaling: ", np.all(data== data_no_scaling))
    print("=" * 80)

    # _fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(10, 10))  # pyright: ignore[reportAny, reportUnknownMemberType]
    # axs = axs.flatten()
    # axs[0].plot(path[0])  # pyright: ignore[reportAny]
    # axs[0].plot(path[1])  # pyright: ignore[reportAny]

    # axs[1].plot(path_no_scaling[0])  # pyright: ignore[reportAny]
    # axs[1].plot(path_no_scaling[1])  # pyright: ignore[reportAny]

    # axs[2].plot(path_phase_cc[0])  # pyright: ignore[reportAny]
    # axs[2].plot(path_phase_cc[1])  # pyright: ignore[reportAny]

    # axs[3].imshow(data[0])
    # axs[4].imshow(data_no_scaling[0])
    # axs[5].imshow(data_phase_cc[0])
    

    # for ax in axs:  # pyright: ignore[reportAny]
    #     ax.set_box_aspect(1)  # pyright: ignore[reportAny]


    # plt.tight_layout()
    # plt.show()  # pyright: ignore[reportUnknownMemberType]

    # fast_movie.align_rows()

    if "i" in channel:
        fast_movie.rescale((1, 2))
    else:
        fast_movie.rescale((2, 2))

    # fast_movie.crop((50, 120), (50, 120))
    # fast_movie.cut((20, 50))
    # fast_movie.algin_rows("median")

    # fast_movie.export_mp4(fps_factor=2)
    fast_movie.export_tiff()
    #
    # fast_movie.export_frames_image("png", (0, 3), color_map="bone")
    # fast_movie.export_frames_txt((0, 3))
    # fast_movie.export_frames_gwy("images", (0, 5))
    # fast_movie.export_frames_gwy("volume", (0, 5))

    # break
