from __future__ import annotations

import copy
from dataclasses import dataclass
import os
from typing import final, override

import numpy as np
import skimage as ski
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas

# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pyfastspm import FastMovie
from PySide6.QtCore import QSize, Signal, SignalInstance
from PySide6.QtGui import QCloseEvent, QFocusEvent, QKeySequence, QShortcut, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

COLORMAP = "bone"


@dataclass
class MovieInfo:
    id_: int
    filename: str


@final
class MovieWindow(QWidget):
    window_focused = Signal(MovieInfo)
    window_closed = Signal(MovieInfo)

    def __init__(self, fast_movie: FastMovie, channel: str) -> None:
        super().__init__()
        self.ft = fast_movie
        print(f"New window with channel: {channel}")
        self.channel = channel
        self.ft.channel = channel
        # self.filename: str = os.path.basename(fast_movie.filename)
        # self.movie_id = id(self)
        self.info = MovieInfo(
            id_=id(self), filename=os.path.basename(fast_movie.filename)
        )

        self.setWindowTitle(f"{self.info.filename}({self.info.id_})-{self.channel}")
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.num_frames: int = self.ft.num_frames
        self.current_frame_num: int = 0

        self.canvas = FigureCanvas(Figure(figsize=(4, 4)))
        self.movie_controls = MovieControls(
            num_frames=self.num_frames,
            fps=12,
            focus_signal=self.window_focused,
            movie_info=self.info,
        )

        self.plot_data = self.ft.data
        self.ax = None
        self.img_plot = None
        self.create_plot()

        # Layout
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        layout.addWidget(self.movie_controls)
        # Strech the canvas when window resizes
        layout.setStretch(1, 2)

        self.process_indicator = ProcessIndicator("")
        self.process_indicator.hide()
        layout.addWidget(self.process_indicator)

        # Connect signals
        _ = self.movie_controls.prev_btn.clicked.connect(self.on_prev_btn_clicked)
        _ = self.movie_controls.next_btn.clicked.connect(self.on_next_btn_clicked)
        _ = self.movie_controls.first_btn.clicked.connect(self.on_first_btn_clicked)
        _ = self.movie_controls.last_btn.clicked.connect(self.on_last_btn_clicked)
        _ = self.movie_controls.play_btn.clicked.connect(self.on_play_btn_clicked)

        _ = self.movie_controls.curr_frame_input.valueChanged.connect(
            self.on_current_frame_changed
        )
        _ = self.movie_controls.fps_input.valueChanged.connect(self.on_play_btn_clicked)

        # Keyboard shortcuts
        shortcut_next = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Period), self)
        shortcut_next.activated.connect(self.on_next_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Comma), self)
        shortcut_prev.activated.connect(self.on_prev_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Space), self)
        shortcut_prev.activated.connect(self.on_play_btn_clicked)

        self.start_playing()
        self.movie_controls.play_btn.setChecked(True)

    def clone_fast_movie(self) -> FastMovie:
        return self.ft.clone()

    def set_movie_id(self, new_movie_id: int) -> None:
        self.info.id_ = new_movie_id
        self.setWindowTitle(f"{self.info.filename}({self.info.id_})-{self.channel}")

    def update_plot_data(self) -> None:
        if self.ft.mode == "timeseries":
            data: NDArray[np.float32] = self.ft.reshape_data(
                copy.deepcopy(self.ft.data),
                self.channel,
                self.ft.metadata["Scanner.X_Points"],
                self.ft.metadata["Scanner.Y_Points"],
                self.ft.num_images,
                self.ft.num_frames,
            )
            self.num_frames = data.shape[0]
            self.ft.num_frames = self.num_frames
            self.movie_controls.set_num_frames(self.num_frames)
        else:
            data = copy.deepcopy(self.ft.data)

        self.plot_data = data

    def create_plot(self) -> None:
        self.update_plot_data()

        if "i" in self.channel:
            print("=" * 80, "RESIZE")
            num_frames = self.plot_data.shape[0]
            y_shape = self.plot_data.shape[1]
            x_shape = self.plot_data.shape[2] * 2
            data_scaled = np.zeros((num_frames, y_shape, x_shape))
            for i in range(num_frames):
                data_scaled[i] = ski.transform.resize(
                    self.plot_data[i], (y_shape, x_shape)
                )
            self.plot_data = data_scaled

        self.ax = self.canvas.figure.subplots()
        self.img_plot = self.ax.imshow(
            self.plot_data[self.current_frame_num],
            interpolation="none",
            cmap=COLORMAP,
        )

        self.img_plot.set_clim(self.ft.data.min(), self.ft.data.max())
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.img_plot.figure.tight_layout(pad=0)

    def recreate_plot(self) -> None:
        if self.img_plot:
            self.img_plot.remove()
        self.ax.cla()
        self.canvas.figure.clf()
        self.create_plot()

    def start_processing(self, message: str) -> None:
        self.process_indicator.status_label.setText(message)
        self.process_indicator.show()

    def end_processing(self) -> None:
        self.process_indicator.hide()
        self.recreate_plot()

    def update_frame(self) -> None:
        self.img_plot.set_data(self.plot_data[self.current_frame_num])
        self.img_plot.figure.canvas.draw()
        self.movie_controls.curr_frame_input.setValue(self.current_frame_num)
        self.movie_controls.current_frame_lbl.setText(f"/{self.num_frames - 1}")

    def start_playing(self) -> None:
        fps = self.movie_controls.fps_input.value()
        update_time = 1 / fps * 1000

        self.timer = self.canvas.new_timer(update_time)
        self.timer.add_callback(self.on_next_btn_clicked)
        self.timer.start()

    def stop_playing(self) -> None:
        self.timer.stop()

    def on_prev_btn_clicked(self) -> None:
        if self.current_frame_num > 0:
            self.current_frame_num -= 1
        else:
            self.current_frame_num = self.num_frames - 1

        self.update_frame()

    def on_next_btn_clicked(self) -> None:
        if self.current_frame_num < self.num_frames - 1:
            self.current_frame_num += 1
        else:
            self.current_frame_num = 0

        self.update_frame()

    def on_first_btn_clicked(self) -> None:
        self.current_frame_num = 0
        self.update_frame()

    def on_last_btn_clicked(self) -> None:
        self.current_frame_num = self.num_frames - 1
        self.update_frame()

    def on_current_frame_changed(self, value: int) -> None:
        if value > self.num_frames - 1:
            self.current_frame_num = self.num_frames - 1
        elif value < 0:
            self.current_frame_num = 0
        else:
            self.current_frame_num = value

        self.update_frame()

    def on_play_btn_clicked(self) -> None:
        if self.movie_controls.play_btn.isChecked():
            self.start_playing()
        else:
            self.stop_playing()

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        print(f"Focused: {self.info.id_}")
        self.window_focused.emit(self.info)
        super().focusInEvent(event)

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        print(f"Closed {self.info}")
        self.timer.stop()
        self.window_closed.emit(self.info)
        super().closeEvent(event)


@final
class MovieControlButton(QPushButton):
    def __init__(self, icon: QStyle.StandardPixmap, checkable: bool = False) -> None:
        super().__init__()
        self.setIcon(self.style().standardIcon(icon))
        self.setIconSize(QSize(30, 30))
        self.setFixedSize(35, 35)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setCheckable(checkable)


@final
class MovieControlSpinBox(QSpinBox):
    def __init__(self, focus_signal: SignalInstance, movie_info: MovieInfo) -> None:
        super().__init__()
        self.focus_signal = focus_signal
        self.movie_info = movie_info
        self.setFocusPolicy(Qt.StrongFocus)

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        self.focus_signal.emit(self.movie_info)
        super().focusInEvent(event)


@final
class MovieControls(QWidget):
    def __init__(
        self,
        num_frames: int,
        fps: int,
        focus_signal: SignalInstance,
        movie_info: MovieInfo,
    ) -> None:
        super().__init__()
        self.prev_btn = MovieControlButton(QStyle.SP_MediaSeekBackward)
        self.next_btn = MovieControlButton(QStyle.SP_MediaSeekForward)
        self.first_btn = MovieControlButton(QStyle.SP_MediaSkipBackward)
        self.last_btn = MovieControlButton(QStyle.SP_MediaSkipForward)

        self.curr_frame_input = MovieControlSpinBox(focus_signal, movie_info)
        self.curr_frame_input.setRange(0, 5000)
        self.curr_frame_input.setValue(0)
        self.current_frame_lbl = QLabel(f"/{num_frames}")

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)  # Vertical line
        separator.setFixedWidth(2)

        self.play_btn = MovieControlButton(QStyle.SP_MediaPlay, checkable=True)

        self.fps_input = MovieControlSpinBox(focus_signal, movie_info)
        self.fps_input.setRange(1, 50)
        self.fps_input.setValue(fps)
        fps_lbl = QLabel("fps")

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.play_btn)

        layout.addWidget(self.fps_input)
        layout.addWidget(fps_lbl)
        layout.addWidget(separator)

        layout.addWidget(self.first_btn)
        layout.addWidget(self.prev_btn)
        layout.addWidget(self.next_btn)
        layout.addWidget(self.last_btn)
        layout.addWidget(self.curr_frame_input)
        layout.addWidget(self.current_frame_lbl)
        # Strech the canvas when window resizes
        layout.setStretch(10, 2)

    def set_num_frames(self, new_num_frames: int) -> None:
        self.current_frame_lbl.setText(f"/{new_num_frames}")


@final
class ProcessIndicator(QWidget):
    def __init__(self, label: str):
        super().__init__()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setValue(0)

        self.status_label = QLabel(label)

        layout = QHBoxLayout(self)
        layout.addWidget(self.status_label)
        layout.addWidget(self.progress_bar)
        self.setLayout(layout)
