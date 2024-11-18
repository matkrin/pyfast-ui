import copy
import os
import sys
from PySide6.QtCore import QSize, Slot
from PySide6.QtGui import QKeySequence, QShortcut, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

import pyfastspm as pf
from pyfastspm.artefact_removal.conv_mat import FastMovie

from pyfast_ui.creep_group import CreepGroup
from pyfast_ui.drift_group import DriftGroup
from pyfast_ui.fft_filters_group import FFTFiltersGroup
from pyfast_ui.import_group import ImportGroup

FAST_FILE = "/Users/matthias/github/pyfastspm/examples/F20190424_1.h5"


class MainGui(QMainWindow):
    """Main GUI"""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PyfastSPM")
        # self.setGeometry(100, 100, 700, 400)
        self.move(50, 50)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        self.open_btn = QPushButton("Open test file")
        _ = self.open_btn.clicked.connect(self.on_open_btn_click)

        self.central_layout.addWidget(self.open_btn)

        self.plot_windows: list[QWidget] = []

        self.import_group = ImportGroup(None, True)
        self.fft_filters_group = FFTFiltersGroup(
            filter_x=True,
            filter_y=True,
            filter_x_overtones=False,
            filter_high_pass=True,
            filter_pump=True,
            filter_noise=False,
            display_spectrum=True,
        )
        export_group = QGroupBox("Export")
        creep_group = CreepGroup("sin")
        streak_removal_group = QGroupBox("Streak Removal")
        crop_group = QGroupBox("Crop")
        drift_group = DriftGroup(
            fft_drift=True, drifttype="common", stepsize=100, known_drift=False
        )
        image_correction_group = QGroupBox("Image Correction")
        image_filters_group = QGroupBox("2D Filters")
        export_group = QGroupBox("Export")

        self.central_layout.addWidget(self.import_group)
        self.central_layout.addWidget(self.fft_filters_group)
        self.central_layout.addWidget(creep_group)
        self.central_layout.addWidget(drift_group)

        _ = self.import_group.apply_btn.clicked.connect(self.on_import_btn_click)
        _ = self.fft_filters_group.apply_btn.clicked.connect(self.on_apply_fft_filter)

    def on_open_btn_click(self) -> None:
        ft = pf.FastMovie(FAST_FILE, y_phase=0)
        movie_window = MovieWindow(ft)
        self.plot_windows.append(movie_window)
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
            self.plot_windows.append(movie_window)
            movie_window.show()
        else:
            print("No file chosen.")

    def on_apply_fft_filter(self) -> None:
        print("FFT filter applied")
        ft = self.plot_windows[0].ft
        ft.data = self.plot_windows[0].ft_raw_data
        ft.mode = "timeseries"
        filterparams = self.fft_filters_group.filterparams
        if any(self.fft_filters_group.filterparams):
            pf.filter_movie(
                ft=ft,
                filterparam=filterparams,
                filter_broadness=None,
                fft_display_range=(0, 40000),
                pump_freqs=tuple(
                    (
                        1500.0,
                        1000.0,
                    )
                ),
                num_pump_overtones=3,
                num_x_overtones=10,
                high_pass_params=(1000.0, 600.0),
            )

        ft.reshape_to_movie()
        self.plot_windows[0].img_plot.set_clim(ft.data.min(), ft.data.max())


class MovieWindow(QWidget):
    def __init__(self, fast_movie: FastMovie) -> None:
        super().__init__()
        self.filename = os.path.basename(fast_movie.h5file.filename)
        self.setWindowTitle(self.filename)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.ft_raw_data = copy.deepcopy(fast_movie.data)
        self.ft = fast_movie
        self.ft.reshape_to_movie()
        print(self.ft.data)
        self.num_frames = self.ft.data.shape[0]
        print(self.num_frames)

        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)

        self.current_frame_num = 0
        self.ax = self.canvas.figure.subplots()
        self.img_plot = self.ax.imshow(
            self.ft.data[self.current_frame_num], interpolation="none"
        )
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.img_plot.figure.tight_layout(pad=0)

        prev_btn = QPushButton()
        prev_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekBackward))
        prev_btn.setIconSize(QSize(30, 30))
        prev_btn.setFixedSize(35, 35)
        prev_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        next_btn = QPushButton()
        next_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSeekForward))
        next_btn.setIconSize(QSize(30, 30))
        next_btn.setFixedSize(35, 35)
        prev_btn.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)

        first_btn = QPushButton()
        first_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        first_btn.setIconSize(QSize(30, 30))
        first_btn.setFixedSize(35, 35)

        last_btn = QPushButton()
        last_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        last_btn.setIconSize(QSize(30, 30))
        last_btn.setFixedSize(35, 35)

        self.curr_frame_input = QSpinBox(self)
        self.curr_frame_input.setRange(0, 5000)
        self.curr_frame_input.setValue(0)
        self.current_frame_lbl = QLabel(f"/{self.num_frames}")

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)  # Vertical line
        separator.setFixedWidth(2)

        play_btn = QPushButton()
        play_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        play_btn.setIconSize(QSize(30, 30))
        play_btn.setFixedSize(35, 35)

        pause_btn = QPushButton()
        pause_btn.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        pause_btn.setIconSize(QSize(30, 30))
        pause_btn.setFixedSize(35, 35)

        self.fps_input = QSpinBox(self)
        self.fps_input.setRange(1, 50)
        self.fps_input.setValue(24)
        fps_lbl = QLabel("fps")

        # Layout
        control_layout = QHBoxLayout()
        layout.addLayout(control_layout)

        control_layout.addWidget(play_btn)
        control_layout.addWidget(pause_btn)

        control_layout.addWidget(self.fps_input)
        control_layout.addWidget(fps_lbl)
        control_layout.addWidget(separator)

        control_layout.addWidget(first_btn)
        control_layout.addWidget(prev_btn)
        control_layout.addWidget(next_btn)
        control_layout.addWidget(last_btn)
        control_layout.addWidget(self.curr_frame_input)
        control_layout.addWidget(self.current_frame_lbl)

        # Strech the canvas when window resizes
        layout.setStretch(1, 2)
        control_layout.setStretch(10, 2)

        # Connect signals
        _ = prev_btn.clicked.connect(self.on_prev_btn_clicked)
        _ = next_btn.clicked.connect(self.on_next_btn_clicked)
        _ = first_btn.clicked.connect(self.on_first_btn_clicked)
        _ = last_btn.clicked.connect(self.on_last_btn_clicked)
        _ = play_btn.clicked.connect(self.on_play_btn_clicked)
        _ = pause_btn.clicked.connect(self.on_pause_btn_clicked)

        _ = self.curr_frame_input.valueChanged.connect(self.on_current_frame_changed)
        _ = self.fps_input.valueChanged.connect(self.on_play_btn_clicked)

        # Keyboard shortcuts
        shortcut_next = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Period), self)
        shortcut_next.activated.connect(self.on_next_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Comma), self)
        shortcut_prev.activated.connect(self.on_prev_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.ALT | Qt.Key_Space), self)
        shortcut_prev.activated.connect(self.on_play_btn_clicked)

    def update_frame(self) -> None:
        self.img_plot.set_data(self.ft.data[self.current_frame_num])
        self.img_plot.figure.canvas.draw()
        self.curr_frame_input.setValue(self.current_frame_num)
        self.current_frame_lbl.setText(f"/{self.num_frames - 1}")

    @Slot()
    def on_prev_btn_clicked(self) -> None:
        if self.current_frame_num > 0:
            self.current_frame_num -= 1
        else:
            self.current_frame_num = self.num_frames - 1

        self.update_frame()

    @Slot()
    def on_next_btn_clicked(self) -> None:
        if self.current_frame_num < self.num_frames - 1:
            self.current_frame_num += 1
        else:
            self.current_frame_num = 0

        self.update_frame()

    @Slot()
    def on_first_btn_clicked(self) -> None:
        self.current_frame_num = 0
        self.update_frame()

    @Slot()
    def on_last_btn_clicked(self) -> None:
        self.current_frame_num = self.num_frames - 1
        self.update_frame()

    @Slot()
    def on_current_frame_changed(self, value) -> None:
        if value > self.num_frames - 1:
            self.current_frame_num = self.num_frames - 1
        elif value < 0:
            self.current_frame_num = 0
        else:
            self.current_frame_num = value

        self.update_frame()

    @Slot()
    def on_play_btn_clicked(self) -> None:
        fps = self.fps_input.value()
        update_time = 1 / fps * 1000

        self.timer = self.canvas.new_timer(update_time)
        self.timer.add_callback(self.on_next_btn_clicked)
        self.timer.start()

    @Slot()
    def on_pause_btn_clicked(self) -> None:
        self.timer.stop()


def main():
    app = QApplication(sys.argv)
    window = MainGui()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
