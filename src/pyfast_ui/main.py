import sys
from pathlib import Path
from typing import final, override

import matplotlib.pyplot as plt
import numpy as np
import pyfastspm as pf
from PySide6.QtCore import QCoreApplication, QThreadPool
from PySide6.QtGui import QCloseEvent, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pyfast_ui.channel_select_group import ChannelSelectGroup
from pyfast_ui.config import Config, GeneralConfig, init_config
from pyfast_ui.creep_group import CreepGroup
from pyfast_ui.custom_widgets import LabeledCombobox
from pyfast_ui.drift_group import DriftGroup
from pyfast_ui.export_group import ExportGroup
from pyfast_ui.fft_filters_group import FFTFiltersGroup
from pyfast_ui.histogram_window import HistogramWindow
from pyfast_ui.image_correction import ImageCorrectionGroup
from pyfast_ui.image_filters import ImageFilterGroup
from pyfast_ui.modify_group import ModifyGroup
from pyfast_ui.movie_window import MovieInfo, MovieWindow
from pyfast_ui.phase_group import PhaseGroup
from pyfast_ui.workers import CreepWorker, DriftWorker, FftFilterWorker

FAST_FILE = "/home/matthias/github/pyfastspm/examples/F20190424_1.h5"


@final
class MainGui(QMainWindow):
    """Main GUI"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("PyfastSPM")
        self.setGeometry(100, 100, 850, 1000)
        self.move(50, 50)

        self.central_widget = QWidget()
        scroll_area = QScrollArea()
        self.central_layout = QHBoxLayout()
        self.central_widget.setLayout(self.central_layout)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.central_widget)
        self.setCentralWidget(scroll_area)

        self.thread_pool = QThreadPool()

        ### TEST BUTTON
        self.open_btn = QPushButton("Open test file")
        _ = self.open_btn.clicked.connect(self.on_open_btn_click)

        self.central_layout.addWidget(self.open_btn)
        ###

        self.plot_windows: dict[int, MovieWindow] = dict()
        self.histogram_windows: dict[int, HistogramWindow] = dict()
        self.operate_on: int | None = None
        config = init_config()

        self.operate_label = QLabel("Operate on: ")

        config_layout = QHBoxLayout()
        self.load_config_btn = QPushButton("Load config")
        self.save_config_btn = QPushButton("Save config")
        config_layout.addWidget(self.load_config_btn)
        config_layout.addWidget(self.save_config_btn)

        self.import_btn = QPushButton("Import movie")
        # Batch mode
        self.batch_btn = QPushButton("Batch")
        _ = self.batch_btn.clicked.connect(self.batch_mode)
        self.channel_select_group = ChannelSelectGroup()
        # Colormap
        self._colormap = LabeledCombobox("Colormap", plt.colormaps())
        self._colormap.combobox.setCurrentText(config.general.colormap)
        # Histogram
        self.histogram_btn = QPushButton("Histogram")
        _ = self.histogram_btn.clicked.connect(self.on_histogram_btn)

        self.modify_group = ModifyGroup((0, 0))
        self.phase_group = PhaseGroup.from_config(config.phase)
        self.fft_filters_group = FFTFiltersGroup.from_config(config.fft_filter)
        self.creep_group = CreepGroup.from_config(config.creep)

        # TODO
        # streak_removal_group = QGroupBox("Streak Removal")
        # TODO
        # crop_group = QGroupBox("Crop")

        self.drift_group = DriftGroup.from_config(config.drift)
        self.image_correction_group = ImageCorrectionGroup.from_config(
            config.image_correction
        )
        self.image_filter_group = ImageFilterGroup.from_config(config.image_filter)
        self.export_group = ExportGroup.from_config(config.export)

        horizontal_layout = QHBoxLayout()
        vertical_layout_left = QVBoxLayout()
        vertical_layout_right = QVBoxLayout()
        vertical_layout_left.addWidget(self.operate_label)
        vertical_layout_left.addWidget(self.import_btn)
        vertical_layout_left.addLayout(config_layout)
        vertical_layout_left.addWidget(self.batch_btn)
        vertical_layout_left.addWidget(self.channel_select_group)
        vertical_layout_left.addWidget(self._colormap)
        vertical_layout_left.addWidget(self.histogram_btn)
        # Groups
        vertical_layout_left.addWidget(self.modify_group)
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
        _ = self.load_config_btn.clicked.connect(self.load_config)
        _ = self.save_config_btn.clicked.connect(self.save_config)
        _ = self.import_btn.clicked.connect(self.on_import_btn_click)
        _ = self.channel_select_group.new_btn.clicked.connect(
            self.on_channel_select_new
        )
        _ = self._colormap.value_changed.connect(self.update_colormap)
        _ = self.modify_group.toggle_selection_btn.clicked.connect(
            self.on_toggle_selection_btn
        )
        _ = self.modify_group.crop_btn.clicked.connect(self.on_crop_btn)
        _ = self.phase_group.apply_btn.clicked.connect(self.on_phase_apply)
        _ = self.phase_group.new_btn.clicked.connect(self.on_phase_new)
        _ = self.fft_filters_group.apply_btn.clicked.connect(self.on_fft_filter_apply)
        _ = self.fft_filters_group.new_btn.clicked.connect(self.on_fft_filter_new)
        _ = self.creep_group.apply_btn.clicked.connect(self.on_creep_apply)
        _ = self.creep_group.new_btn.clicked.connect(self.on_creep_new)
        _ = self.image_correction_group.apply_btn.clicked.connect(
            self.on_image_correction_apply
        )
        _ = self.image_correction_group.new_btn.clicked.connect(
            self.on_image_correction_new
        )
        _ = self.image_filter_group.apply_btn.clicked.connect(
            self.on_image_filter_apply
        )
        _ = self.image_filter_group.new_btn.clicked.connect(self.on_image_filter_new)
        _ = self.drift_group.apply_btn.clicked.connect(self.on_drift_apply)
        _ = self.drift_group.new_btn.clicked.connect(self.on_drift_new)
        _ = self.export_group.apply_btn.clicked.connect(self.on_export_apply)

    def update_focused_window(self, movie_info: MovieInfo) -> None:
        print(f"Focus on: {movie_info}")
        self.operate_on = movie_info.id_
        self.operate_label.setText(
            f"Operate on: {movie_info.filename}({movie_info.id_})"
        )

    def create_new_movie_window(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        channel = fast_movie_window.channel
        new_ft = fast_movie_window.clone_fast_movie()
        new_movie_window = MovieWindow(new_ft, channel, self._colormap.value())
        new_movie_window.show()
        _ = new_movie_window.window_focused.connect(self.update_focused_window)
        self.plot_windows.update({new_movie_window.info.id_: new_movie_window})
        self.update_focused_window(new_movie_window.info)

    def on_movie_window_closed(self, movie_info: MovieInfo) -> None:
        del self.plot_windows[movie_info.id_]

    def on_open_btn_click(self) -> None:
        ft = pf.FastMovie(FAST_FILE, y_phase=0)
        movie_window = MovieWindow(ft, "udi", self._colormap.value())
        _ = movie_window.window_focused.connect(self.update_focused_window)
        _ = movie_window.window_closed.connect(self.on_movie_window_closed)
        self.plot_windows.update({movie_window.info.id_: movie_window})
        self.update_focused_window(movie_window.info)
        movie_window.show()

    def on_import_btn_click(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            caption="Choose HDF5 file",
            # dir="",
            filter="HDF5 Files (*.h5)",
        )

        if file_path:
            ft = pf.FastMovie(str(file_path), y_phase=0)
            colormap = self._colormap.value()
            movie_window = MovieWindow(ft, "udi", colormap)
            _ = movie_window.window_focused.connect(self.update_focused_window)
            self.plot_windows.update({movie_window.info.id_: movie_window})
            self.update_focused_window(movie_window.info)
            movie_window.show()
        else:
            print("No file chosen.")

    def save_config(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            caption="Save TOML config as...",
            # directory="",
            filter="TOML Files (*.toml)",
        )
        filepath = Path(filepath).with_suffix(".toml")

        general_config = GeneralConfig(
            channel=self.channel_select_group.channel, colormap=self._colormap.value()
        )
        config = Config(
            general=general_config,
            phase=self.phase_group.to_config(),
            fft_filter=self.fft_filters_group.to_config(),
            creep=self.creep_group.to_config(),
            drift=self.drift_group.to_config(),
            image_correction=self.image_correction_group.to_config(),
            image_filter=self.image_filter_group.to_config(),
            export=self.export_group.to_config(),
        )
        config.save_toml(filepath)

    def load_config(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            caption="Choose toml config file",
            # dir="",
            filter="TOML Files (*.toml)",
        )

        config = Config.load_toml(Path(filepath))
        self._colormap.set_value(config.general.colormap)
        self.update_colormap(config.general.colormap)
        self.channel_select_group.channel = config.general.channel

        self.phase_group.update_from_config(config.phase)
        self.fft_filters_group.update_from_config(config.fft_filter)
        self.creep_group.update_from_config(config.creep)
        self.drift_group.update_from_config(config.drift)
        self.image_correction_group.update_from_config(config.image_correction)
        self.image_filter_group.update_from_config(config.image_filter)
        self.export_group.update_from_config(config.export)

    def batch_mode(self) -> None:
        dirname = QFileDialog.getExistingDirectory(
            self,
            caption="Choose folder to process",
            # directory="",
            options=QFileDialog.Option.ShowDirsOnly,
        )
        dirpath = Path(dirname)
        for filepath in dirpath.iterdir():
            if filepath.is_file() and filepath.suffix == ".h5":
                ft = pf.FastMovie(str(filepath), y_phase=0)
                colormap = self._colormap.value()
                movie_window = MovieWindow(ft, "udi", colormap)
                _ = movie_window.window_focused.connect(self.update_focused_window)
                self.plot_windows.update({movie_window.info.id_: movie_window})
                self.update_focused_window(movie_window.info)
                # movie_window.show()
                self.on_phase_apply()
                self.on_fft_filter_apply()
                _ = self.thread_pool.waitForDone()
                self.on_creep_apply()
                _ = self.thread_pool.waitForDone()
                self.on_drift_apply()
                _ = self.thread_pool.waitForDone()
                self.on_image_correction_apply()
                _ = self.thread_pool.waitForDone()
                self.on_image_filter_apply()
                _ = self.thread_pool.waitForDone()
                self.on_export_apply()
                _ = self.thread_pool.waitForDone()

    def update_colormap(self, value: str) -> None:
        for movie_window in self.plot_windows.values():
            movie_window.set_colormap(value)

    def on_channel_select_new(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        channel = self.channel_select_group.channel

        new_ft = fast_movie_window.clone_fast_movie()
        new_ft.reload_timeseries()
        new_movie_window = MovieWindow(new_ft, channel, self._colormap.value())
        new_movie_window.show()
        _ = new_movie_window.window_focused.connect(self.update_focused_window)
        self.plot_windows.update({new_movie_window.info.id_: new_movie_window})
        self.update_focused_window(new_movie_window.info)

    def on_toggle_selection_btn(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        is_active = fast_movie_window.is_selection_active()
        fast_movie_window.selection_set_active(not is_active)
        print(f"{is_active=}")
        self.modify_group.toggle_selection_btn.setDefault(not is_active)

    def on_crop_btn(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return


        if fast_movie_window.ft.mode == "timeseries":
            print("not in image mode")
            return

        rectangle = fast_movie_window.get_selection()
        self.create_new_movie_window()
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if rectangle is not None and fast_movie_window is not None:
            ft = fast_movie_window.ft
            ul, ur, lr, ll = rectangle
            x_start, y_start = ul
            y_end = ll[1]
            x_end = lr[0]
            if "i" in ft.channels:
                x_start = round(x_start / 2)
                x_end = round(x_end / 2)
            print(f"cropping with {y_start}, {y_end=}, {x_start=}, {x_end=}")

            fast_movie_window.stop_playing()
            ft.data = ft.data[:, y_start:y_end, x_start:x_end]
            fast_movie_window.recreate_plot()
            fast_movie_window.start_playing()

    def on_phase_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        ft = fast_movie_window.ft

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

        fast_movie_window.recreate_plot()

    def on_phase_new(self) -> None:
        self.create_new_movie_window()
        self.on_phase_apply()

    def on_fft_filter_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return
        ft = fast_movie_window.ft

        fast_movie_window.start_processing("FFT-filtering...")

        filter_broadness = self.fft_filters_group.filter_broadness
        fft_display_range = self.fft_filters_group.fft_display_range
        pump_freqs = self.fft_filters_group.pump_freqs
        num_pump_overtones = self.fft_filters_group.num_pump_overtones
        num_x_overtones = self.fft_filters_group.num_x_overtones
        high_pass_params = self.fft_filters_group.high_pass_params

        filterparams = self.fft_filters_group.filterparams

        fft_filter_worker = FftFilterWorker(
            fast_movie=ft,
            filterparams=filterparams,
            filter_broadness=filter_broadness,
            fft_display_range=fft_display_range,
            pump_freqs=pump_freqs,
            num_pump_overtones=num_pump_overtones,
            num_x_overtones=num_x_overtones,
            high_pass_params=high_pass_params,
        )

        _ = fft_filter_worker.signals.finished.connect(fast_movie_window.end_processing)
        self.thread_pool.start(fft_filter_worker)

    def on_fft_filter_new(self) -> None:
        self.create_new_movie_window()
        self.on_fft_filter_apply()

    def on_creep_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        fast_movie_window.start_processing("Correcting creep...")

        ft = fast_movie_window.ft
        if ft.mode == "timeseries":
            ft.reshape_to_movie(fast_movie_window.channel)

        creep_mode = self.creep_group.creep_mode
        # if self.import_group.is_image_range:
        #     image_range = self.import_group.image_range
        # else:
        #     image_range = None
        image_range = None
        print(f"{image_range=}")

        # Basic settings -> Streak removal for interlacing
        # remove_streaks = False
        # Basic setting -> Export
        # export_movie = True
        # export_frames = False
        # Advanced settings
        guess_ind = self.creep_group.guess_ind
        weight_boundary = self.creep_group.weight_boundry
        creep_num_cols = self.creep_group.creep_num_cols
        initial_guess = (self.creep_group.initial_guess,)
        known_input = None
        known_params = None

        creep_worker = CreepWorker(
            fast_movie=ft,
            creep_mode=creep_mode,
            weight_boundary=weight_boundary,
            creep_num_cols=creep_num_cols,
            known_input=known_input,
            initial_guess=initial_guess,
            guess_ind=guess_ind,
            known_params=known_params,
        )
        _ = creep_worker.signals.finished.connect(fast_movie_window.end_processing)
        self.thread_pool.start(creep_worker)

    def on_creep_new(self) -> None:
        self.create_new_movie_window()
        self.on_creep_apply()

    def on_drift_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        fast_movie_window.start_processing("Correcting drift...")

        ft = fast_movie_window.ft
        if ft.mode == "timeseries":
            ft.reshape_to_movie(fast_movie_window.channel)

        drift_algorithm = self.drift_group.drift_algorithm
        fft_drift = self.drift_group.fft_drift
        stepsize = self.drift_group.stepsize
        drifttype = self.drift_group.drift_type
        known_drift = self.drift_group.known_drift
        stackreg_reference = self.drift_group.stackreg_reference
        boxcar = self.drift_group.boxcar
        median_filter = self.drift_group.median_filter

        # if self.import_group.is_image_range:
        #     image_range = self.import_group.image_range
        # else:
        #     image_range = None
        image_range = None
        print(f"Drift correction with {drift_algorithm=}")

        drift_worker = DriftWorker(
            fast_movie=ft,
            fft_drift=fft_drift,
            drifttype=drifttype,
            stepsize=stepsize,
            known_drift=known_drift,
            drift_algorithm=drift_algorithm,
            stackreg_reference=stackreg_reference,
            boxcar=boxcar,
            median_filter=median_filter,
        )

        _ = drift_worker.signals.finished.connect(fast_movie_window.end_processing)
        self.thread_pool.start(drift_worker)

        # if image_range == None:
        #     first = 0
        #     last = -1
        # elif "u" and "d" in export_channels:
        #     first = image_range[0] * 2
        #     last = image_range[1] * 2
        # else:
        #     first = image_range[0]
        #     last = image_range[1]

    def on_drift_new(self) -> None:
        self.create_new_movie_window()
        self.on_drift_apply()

    def on_export_apply(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        histogram_window = self.histogram_windows.get(self.operate_on)
        if histogram_window is None:
            contrast = (0.0, 1.0)
        else:
            contrast = histogram_window.contrast_percent()

        print(contrast)
        ft = fast_movie_window.ft
        color_map = self._colormap.value()

        export_movie = self.export_group.export_movie
        export_tiff = self.export_group.export_tiff
        export_frames = self.export_group.export_frames
        fps_factor = self.export_group.fps_factor
        scaling = self.export_group.scaling
        auto_label = self.export_group.auto_label

        frame_export_images = self.export_group.frame_export_images
        frame_export_channel = self.export_group.frame_export_channel
        frame_export_format = self.export_group.frame_export_format

        # if self.import_group.is_image_range:
        #     image_range = self.import_group.image_range
        # else:
        #     image_range = None
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

        if export_tiff:
            ft.export_tiff(image_range)

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

    def on_image_correction_new(self) -> None:
        self.create_new_movie_window()
        self.on_image_correction_apply()

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

    def on_image_filter_new(self) -> None:
        self.create_new_movie_window()
        self.on_image_filter_apply()

    def on_histogram_btn(self) -> None:
        if self.operate_on is None:
            return
        fast_movie_window = self.plot_windows.get(self.operate_on)
        if fast_movie_window is None:
            return

        print("New Histogram")
        histogram_window = HistogramWindow(
            fast_movie_window.ft, fast_movie_window.info, fast_movie_window.set_clim
        )
        self.histogram_windows.update({fast_movie_window.info.id_: histogram_window})
        histogram_window.show()

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        for plot_window in list(self.plot_windows.values()):
            _ = plot_window.close()
        QCoreApplication.quit()


def main():
    app = QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
