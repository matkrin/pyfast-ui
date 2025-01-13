import copy
import os
from typing import final, override

from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas

# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from pyfastspm import FastMovie
from PySide6.QtCore import QSize, Signal, SignalInstance, Slot
from PySide6.QtGui import QFocusEvent, QIcon, QKeySequence, QShortcut, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


@final
class MovieWindow(QWidget):
    window_focused = Signal(str)

    def __init__(self, fast_movie: FastMovie) -> None:
        super().__init__()
        self.filename: str = os.path.basename(fast_movie.filename)
        self.setWindowTitle(self.filename)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.ft_raw_data = copy.deepcopy(fast_movie.data)
        self.ft = fast_movie
        self.ft.reshape_to_movie()
        self.num_frames: int = self.ft.data.shape[0]

        self.current_frame_num = 0

        self.canvas = FigureCanvas(Figure(figsize=(5, 5)))

        self.ax = None
        self.img_plot = None
        self.create_plot()

        self.movie_controls = MovieControls(
            num_frames=self.num_frames,
            fps=24,
            focus_signal=self.window_focused,
            window_id=self.filename,
        )

        # Layout
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        layout.addWidget(self.movie_controls)
        # Strech the canvas when window resizes
        layout.setStretch(1, 2)

        # Connect signals
        _ = self.movie_controls.prev_btn.clicked.connect(self.on_prev_btn_clicked)
        _ = self.movie_controls.next_btn.clicked.connect(self.on_next_btn_clicked)
        _ = self.movie_controls.first_btn.clicked.connect(self.on_first_btn_clicked)
        _ = self.movie_controls.last_btn.clicked.connect(self.on_last_btn_clicked)
        _ = self.movie_controls.play_btn.clicked.connect(self.on_play_btn_clicked)
        _ = self.movie_controls.pause_btn.clicked.connect(self.on_pause_btn_clicked)

        _ = self.movie_controls.curr_frame_input.valueChanged.connect(
            self.on_current_frame_changed
        )
        _ = self.movie_controls.fps_input.valueChanged.connect(self.on_play_btn_clicked)

        # Keyboard shortcuts
        shortcut_next = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Period), self)
        shortcut_next.activated.connect(self.on_next_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.CTRL | Qt.Key_Comma), self)
        shortcut_prev.activated.connect(self.on_prev_btn_clicked)
        shortcut_prev = QShortcut(QKeySequence(Qt.ALT | Qt.Key_Space), self)
        shortcut_prev.activated.connect(self.on_play_btn_clicked)

    def create_plot(self) -> None:
        self.ax = self.canvas.figure.subplots()
        self.img_plot = self.ax.imshow(
            self.ft.data[self.current_frame_num], interpolation="none"
        )
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.img_plot.figure.tight_layout(pad=0)

    def recreate_plot(self) -> None:
        self.ax.cla()
        self.canvas.figure.clf()
        self.create_plot()

    def update_frame(self) -> None:
        self.img_plot.set_data(self.ft.data[self.current_frame_num])
        self.img_plot.figure.canvas.draw()
        self.movie_controls.curr_frame_input.setValue(self.current_frame_num)
        self.movie_controls.current_frame_lbl.setText(f"/{self.num_frames - 1}")

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
        fps = self.movie_controls.fps_input.value()
        update_time = 1 / fps * 1000

        self.timer = self.canvas.new_timer(update_time)
        self.timer.add_callback(self.on_next_btn_clicked)
        self.timer.start()

    def on_pause_btn_clicked(self) -> None:
        self.timer.stop()

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        print(f"Focused: {self.filename}")
        self.window_focused.emit(self.filename)
        super().focusInEvent(event)


@final
class MovieControlButton(QPushButton):
    def __init__(self, icon: QStyle.StandardPixmap) -> None:
        super().__init__()
        self.setIcon(self.style().standardIcon(icon))
        self.setIconSize(QSize(30, 30))
        self.setFixedSize(35, 35)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)


@final
class MovieControlSpinBox(QSpinBox):
    def __init__(self, focus_signal: SignalInstance, window_id: str) -> None:
        super().__init__()
        self.focus_signal = focus_signal
        self.window_id = window_id
        self.setFocusPolicy(Qt.StrongFocus)

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        self.focus_signal.emit(self.window_id)
        super().focusInEvent(event)


@final
class MovieControls(QWidget):
    def __init__(
        self, num_frames: int, fps: int, focus_signal: SignalInstance, window_id: str
    ) -> None:
        super().__init__()
        self.prev_btn = MovieControlButton(QStyle.SP_MediaSeekBackward)
        self.next_btn = MovieControlButton(QStyle.SP_MediaSeekForward)
        self.first_btn = MovieControlButton(QStyle.SP_MediaSkipBackward)
        self.last_btn = MovieControlButton(QStyle.SP_MediaSkipForward)

        self.curr_frame_input = MovieControlSpinBox(focus_signal, window_id)
        self.curr_frame_input.setRange(0, 5000)
        self.curr_frame_input.setValue(0)
        self.current_frame_lbl = QLabel(f"/{num_frames}")

        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)  # Vertical line
        separator.setFixedWidth(2)

        self.play_btn = MovieControlButton(QStyle.SP_MediaPlay)
        self.pause_btn = MovieControlButton(QStyle.SP_MediaPause)

        self.fps_input = MovieControlSpinBox(focus_signal, window_id)
        self.fps_input.setRange(1, 50)
        self.fps_input.setValue(fps)
        fps_lbl = QLabel("fps")

        layout = QHBoxLayout()
        self.setLayout(layout)
        layout.addWidget(self.play_btn)
        layout.addWidget(self.pause_btn)

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
