import sys
from typing import final, override

import numpy as np
import pyfastspm as pf
from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pyfast_ui.creep_group import CreepGroup
from pyfast_ui.drift_group import DriftGroup
from pyfast_ui.export_group import ExportGroup
from pyfast_ui.fft_filters_group import FFTFiltersGroup
from pyfast_ui.image_correction import ImageCorrectionGroup
from pyfast_ui.image_filters import ImageFilterGroup
from pyfast_ui.import_group import ImportGroup
from pyfast_ui.movie_window import MovieWindow
from pyfast_ui.phase_group import PhaseGroup

FAST_FILE = "/home/matthias/github/pyfastspm/examples/F20190424_1.h5"


@final
class MainGui(QMainWindow):
    """Main GUI"""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PyfastSPM")
        # self.setGeometry(100, 100, 700, 400)
        self.move(50, 50)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QHBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        # self.open_btn = QPushButton("Open test file")
        # _ = self.open_btn.clicked.connect(self.on_open_btn_click)
        #
        # self.central_layout.addWidget(self.open_btn)

        self.plot_windows: dict[str, MovieWindow] = dict()
        self.operate_on: str | None = None

        self.operate_label = QLabel("Operate on: ")

        self.import_group = ImportGroup(image_range=None, apply_auto_xphase=True)
        self.phase_group = PhaseGroup(
            apply_auto_xphase=True,
            additional_x_phase=0,
            manual_y_phase=0,
            index_frame_to_correlate=0,
            sigma_gauss=0,
        )
        self.fft_filters_group = FFTFiltersGroup(
            filter_x=True,
            filter_y=True,
            filter_x_overtones=False,
            filter_high_pass=True,
            filter_pump=True,
            filter_noise=False,
            display_spectrum=True,
            filter_broadness=None,
            num_x_overtones=10,
            high_pass_params=(1000.0, 600.0),
            num_pump_overtones=3,
            pump_freqs=(
                1500.0,
                1000.0,
            ),
            fft_display_range=(0, 40_000),
        )
        self.creep_group = CreepGroup(
            creep_mode="sin",
            weight_boundry=0.0,
            creep_num_cols=3,
            known_input=None,
            initial_guess=0.3,
            guess_ind=0.2,
            known_params=None,
        )
        # TODO
        streak_removal_group = QGroupBox("Streak Removal")
        # TODO
        crop_group = QGroupBox("Crop")
        self.drift_group = DriftGroup(
            fft_drift=True, drifttype="common", stepsize=100, known_drift=False
        )
        self.image_correction_group = ImageCorrectionGroup(
            correction_type="align", align_type="median"
        )
        self.image_filter_group = ImageFilterGroup(
            filter_type="gaussian2d", pixel_width=3
        )
        self.export_group = ExportGroup(
            export_movie=True,
            export_frames=False,
            frame_export_images=(0, 1),
            frame_export_channel="udi",
            contrast=(0.1, 0.99),
            scaling=(2.0, 2.0),
            fps_factor=5,
            color_map="inferno",
            frame_export_format="tiff",
            auto_label=True,
        )

        horizontal_layout = QHBoxLayout()
        vertical_layout_left = QVBoxLayout()
        vertical_layout_right = QVBoxLayout()
        vertical_layout_left.addWidget(self.operate_label)
        vertical_layout_left.addWidget(self.import_group)
        vertical_layout_left.addWidget(self.phase_group)
        vertical_layout_left.addWidget(self.fft_filters_group)
        vertical_layout_left.addWidget(self.creep_group)
        vertical_layout_right.addWidget(self.drift_group)
        vertical_layout_right.addWidget(self.image_correction_group)
        vertical_layout_right.addWidget(self.image_filter_group)
        vertical_layout_right.addWidget(self.export_group)
        horizontal_layout.addLayout(vertical_layout_left)
        horizontal_layout.addLayout(vertical_layout_right)
        self.central_layout.addLayout(horizontal_layout)

        # Connect signals
        _ = self.import_group.apply_btn.clicked.connect(self.on_import_btn_click)
        _ = self.phase_group.apply_btn.clicked.connect(self.on_phase_apply)
        _ = self.fft_filters_group.apply_btn.clicked.connect(self.on_fft_filter_apply)
        _ = self.creep_group.apply_btn.clicked.connect(self.on_creep_apply)
        _ = self.image_correction_group.apply_btn.clicked.connect(
            self.on_image_correction_apply
        )
        _ = self.image_filter_group.apply_btn.clicked.connect(
            self.on_image_filter_apply
        )
        _ = self.drift_group.apply_btn.clicked.connect(self.on_drift_apply)
        _ = self.export_group.apply_btn.clicked.connect(self.on_export_apply)

    def update_focused_window(self, focused_window: str) -> None:
        print("Focus on: ", focused_window)
        self.operate_on = focused_window
        self.operate_label.setText(f"Operate on: {focused_window}")

    def on_open_btn_click(self) -> None:
        ft = pf.FastMovie(FAST_FILE, y_phase=0)
        movie_window = MovieWindow(ft)
        _ = movie_window.window_focused.connect(self.update_focused_window)
        self.plot_windows.update({movie_window.filename: movie_window})
        movie_window.show()

    def on_import_btn_click(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Save HTML report as...",
            # dir="",
            filter="HDF5 Files (*.h5)",
        )

        if file_path:
            ft = pf.FastMovie(str(file_path), y_phase=0)
            movie_window = MovieWindow(ft)
            _ = movie_window.window_focused.connect(self.update_focused_window)
            self.plot_windows.update({movie_window.filename: movie_window})
            movie_window.show()
        else:
            print("No file chosen.")

    def on_phase_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        ft.data = fast_movie_window.ft_raw_data
        ft.mode = "timeseries"

        apply_auto_xphase = self.phase_group.apply_auto_xphase
        index_frame_to_correlate = self.phase_group.index_frame_to_correlate
        sigma_gauss = self.phase_group.sigma_gauss
        additional_x_phase = self.phase_group.additional_x_phase
        manual_y_phase = self.phase_group.manual_y_phase

        _x_phase = ft.correct_phase(
            apply_auto_xphase=apply_auto_xphase,
            index_frame_to_correlate=index_frame_to_correlate,
            sigma_gauss=sigma_gauss,
            additional_x_phase=additional_x_phase,
            manual_y_phase=manual_y_phase,
        )

        ft.reshape_to_movie("uf")
        fast_movie_window.img_plot.set_clim(ft.data.min(), ft.data.max())

    def on_fft_filter_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return
        ft = fast_movie_window.ft
        ft.data = fast_movie_window.ft_raw_data
        ft.mode = "timeseries"

        filter_broadness = self.fft_filters_group.filter_broadness
        fft_display_range = self.fft_filters_group.fft_display_range
        pump_freqs = self.fft_filters_group.pump_freqs
        num_pump_overtones = self.fft_filters_group.num_pump_overtones
        num_x_overtones = self.fft_filters_group.num_x_overtones
        high_pass_params = self.fft_filters_group.high_pass_params

        filterparams = self.fft_filters_group.filterparams
        if any(self.fft_filters_group.filterparams):
            pf.filter_movie(
                ft=ft,
                filterparam=filterparams,
                filter_broadness=filter_broadness,
                fft_display_range=fft_display_range,
                pump_freqs=pump_freqs,
                num_pump_overtones=num_pump_overtones,
                num_x_overtones=num_x_overtones,
                high_pass_params=high_pass_params,
            )

        ft.reshape_to_movie("uf")

    def on_creep_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        creep_mode = self.creep_group.creep_mode
        if self.import_group.is_image_range:
            image_range = self.import_group.image_range
        else:
            image_range = None
        print(f"{image_range=}")

        # Basic settings -> Streak removal for interlacing
        remove_streaks = False
        # Basic setting -> Export
        export_movie = True
        export_frames = False
        # Advanced settings
        guess_ind = self.creep_group.guess_ind
        weight_boundary = self.creep_group.weight_boundry
        creep_num_cols = self.creep_group.creep_num_cols
        initial_guess = (self.creep_group.initial_guess,)
        known_input = None
        known_params = None

        if creep_mode == "bezier":
            creep = pf.Creep(ft, index_to_linear=guess_ind)
            opt, grid = creep.fit_creep_bez(
                col_inds=np.linspace(
                    ft.data.shape[2] * 0.25, ft.data.shape[2] * 0.75, creep_num_cols
                ).astype(int),
                w=weight_boundary,
                known_input=known_input,
            )

        elif creep_mode != "bezier" and creep_mode != "None":
            creep = pf.Creep(ft, index_to_linear=guess_ind, creep_mode=creep_mode)
            grid = creep.fit_creep(initial_guess, known_params=known_params)
        else:
            grid = None

        interpolation_matrix_up, interpolation_matrix_down = pf.interpolate(
            ft, grid=grid, give_grid=True
        )

        if export_movie or export_frames:
            imrange = image_range

        pf.interpolate(
            ft,
            grid=grid,
            image_range=imrange,
            interpolation_matrix_up=interpolation_matrix_up,
            interpolation_matrix_down=interpolation_matrix_down,
        )

        if remove_streaks:
            edge_removal = [
                [-1.0 / 12.0],
                [3.0 / 12.0],
                [8.0 / 12.0],
                [3.0 / 12.0],
                [-1.0 / 12.0],
            ]
            pf.conv_mat(ft, edge_removal, image_range=imrange)

        fast_movie_window.img_plot.set_clim(ft.data.min(), ft.data.max())

    def on_drift_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        print(f"{ft.data.shape=}")
        fft_drift = self.drift_group.fft_drift
        stepsize = self.drift_group.stepsize
        drifttype = self.drift_group.drift_type
        known_drift = self.drift_group.known_drift

        # Basic settings -> Channels to export
        export_channels = "udi"

        if self.import_group.is_image_range:
            image_range = self.import_group.image_range
        else:
            image_range = None
        print(f"{image_range=}")

        if fft_drift:
            driftrem = pf.Drift(ft, stepsize=stepsize, boxcar=True)
            ft.data, drift_path = driftrem.correct(drifttype, known_drift=known_drift)

        fast_movie_window.recreate_plot()

        print(f"{ft.data.shape=}")
        # if image_range == None:
        #     first = 0
        #     last = -1
        # elif "u" and "d" in export_channels:
        #     first = image_range[0] * 2
        #     last = image_range[1] * 2
        # else:
        #     first = image_range[0]
        #     last = image_range[1]

    def on_export_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        export_movie = self.export_group.export_movie
        export_frames = self.export_group.export_frames
        color_map = self.export_group.color_map
        contrast = self.export_group.contrast
        fps_factor = self.export_group.fps_factor
        scaling = self.export_group.scaling
        auto_label = self.export_group.auto_label

        frame_export_images = self.export_group.frame_export_images
        frame_export_channel = self.export_group.frame_export_channel
        frame_export_format = self.export_group.frame_export_format

        if self.import_group.is_image_range:
            image_range = self.import_group.image_range
        else:
            image_range = None

        if export_movie:
            ft.export_movie(
                color_map=color_map,
                contrast=contrast,
                fps_factor=fps_factor,
                image_range=image_range,
                scaling=scaling,
                auto_label=auto_label,
            )

        if export_frames:
            ft.export_frame(
                images=frame_export_images,
                channel=frame_export_channel,
                color_map=color_map,
                file_format=frame_export_format,
                contrast=contrast,
                scaling=scaling,
            )

    def on_image_correction_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        align_type = self.image_correction_group.align_type
        correction_type = self.image_correction_group.correction_type

        if correction_type == "align":
            pf.align_rows(ft, align_type)
        elif correction_type == "plane":
            pf.level_plane(ft)
        elif correction_type == "fixzero":
            pf.fix_zero(ft)

        fast_movie_window.recreate_plot()

    def on_image_filter_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft
        filter_type = self.image_filter_group.filter_type
        pixel_width = self.image_filter_group.pixel_width

        if filter_type == "mean2d":
            pf.mean_2d(ft, pixel_width)
        elif filter_type == "median2d":
            pf.median_2d(ft, pixel_width)
        elif filter_type == "gaussian2d":
            pf.gaussian_2d(ft, pixel_width)

        fast_movie_window.recreate_plot()

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        for plot_window in self.plot_windows.values():
            _ = plot_window.close()
        QCoreApplication.quit()


def main():
    app = QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
