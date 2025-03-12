from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import final, override

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas

# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure
from matplotlib.widgets import RectangleSelector
from numpy.typing import NDArray
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
from skimage.transform import resize

from pyfast_ui.pyfast_re.channels import Channels
from pyfast_ui.pyfast_re.data_mode import DataMode, reshape_data
from pyfast_ui.pyfast_re.fast_movie import FastMovie


@dataclass
class MovieInfo:
    """Info about a `FastMovie` in a [`MovieWindow`][pyfast_ui.movie_window.MovieWindow]."""

    id_: int
    filename: str
    is_selection_active: bool


@final
class MovieWindow(QWidget):
    """Window containing a `FastMovie`.

    Args:
        fast_movie: The `FastMovie` to display.
        channel: The channel to be selected from `FastMovie`.
        colormap: The colormap in which the movie should be displayed.

    Attributes:
        fast_movie: The `FastMovie` to display.
        channel: The channel to be selected from `FastMovie`.
        colormap: The colormap in which the movie should be displayed.
        info: [`MovieInfo`][pyfast_ui.movie_window.MovieInfo] about `fast_movie`.
        num_frames: Total number of frames of `fast_movie`.
        current_frame_num: The current frame number that is displayed.
        canvas: The matplotlib canvas on which `img_plot` is displayed.
        ax: The matplotlib axis of `img_plot`.
        img_plot: The image plot of the current movie frame.
        rectangle_selection: The matplotlib [`RectangleSelector`](https://matplotlib.org/stable/api/widgets_api.html#matplotlib.widgets.RectangleSelector).
        process_indicator: The process indicator that is active if movie
            processing takes place.
    """

    window_focused = Signal(MovieInfo)
    window_closed = Signal(MovieInfo)

    def __init__(
        self, fast_movie: FastMovie, channel: str, colormap: str = "bone"
    ) -> None:
        super().__init__()
        self.ft = fast_movie
        print(f"New window with channel: {channel}")
        self.picked_channels = Channels(channel)
        self.ft.channels = Channels(channel)
        self.colormap = colormap
        self.info = MovieInfo(
            id_=id(self),
            filename=os.path.basename(fast_movie.filename),
            is_selection_active=False,
        )

        self.setWindowTitle(f"{self.info.filename}({self.info.id_})-{self.picked_channels}")
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.num_frames: int = self.ft.num_frames
        self.current_frame_num: int = 0
        print(f"{self.num_frames=}")

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
        self.rectangle_selection = None
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
        """Clones the `FastMovie` contained in this `MovieWindow`.

        Returns:
            A deepcopy of the `FastMovie`.
        """
        return self.ft.clone()

    def set_movie_id(self, new_movie_id: int) -> None:
        """Sets the `id_` of the `FastMovie`s `MovieInfo`.

        Args:
            new_movie_id: The new movie_id.
        """
        self.info.id_ = new_movie_id
        self.setWindowTitle(f"{self.info.filename}({self.info.id_})-{self.picked_channels}")

    def update_plot_data(self) -> None:
        """Updates [`self.plot_data`][pyfast_ui.movie_window.MovieWindow.plot_data].
        This function should be called after the `FastMovie`'s data changed.
        """
        if self.ft.mode == DataMode.TIMESERIES:
            print("timesseries")
            data: NDArray[np.float32] = reshape_data(
                copy.deepcopy(self.ft.data),
                self.picked_channels,
                self.ft.metadata.num_images,
                self.ft.metadata.scanner_x_points,
                self.ft.metadata.scanner_y_points,
            )
            self.num_frames = data.shape[0]
            # self.ft.num_frames = self.num_frames
            self.movie_controls.set_num_frames(self.num_frames)
        else:
            data = copy.deepcopy(self.ft.data)

        self.plot_data = data
        print(f"{self.plot_data.shape=}")

    def create_plot(self) -> None:
        """Creates an image plot of the `FastMovie`'s data."""
        self.update_plot_data()

        if self.picked_channels.is_interlaced():
            print("=" * 80, "RESIZE")
            num_frames = self.plot_data.shape[0]
            y_shape = self.plot_data.shape[1]
            x_shape = self.plot_data.shape[2] * 2
            print(num_frames, y_shape, x_shape)
            data_scaled = np.zeros((num_frames, y_shape, x_shape))
            for i in range(num_frames):
                data_scaled[i] = resize(
                    self.plot_data[i], (y_shape, x_shape)
                )
            self.plot_data = data_scaled

        self.ax = self.canvas.figure.subplots()
        self.img_plot = self.ax.imshow(
            self.plot_data[self.current_frame_num],
            interpolation="none",
            cmap=self.colormap,
        )

        self.img_plot.set_clim(self.ft.data.min(), self.ft.data.max())
        # self.ax.get_xaxis().set_visible(False)
        # self.ax.get_yaxis().set_visible(False)
        self.img_plot.figure.tight_layout(pad=0)
        self.connect_selection()

    def recreate_plot(self) -> None:
        """Recreates an image plot of the `FastMovie`'s data.'"""
        self.current_frame_num = 0
        if self.img_plot:
            self.img_plot.remove()
        self.ax.cla()
        self.canvas.figure.clf()
        self.create_plot()

    def set_clim(self, lower_limit: float, upper_limit: float) -> None:
        """Sets upper limit and lower limit of the image plot."""
        if self.img_plot is not None:
            self.img_plot.set_clim(lower_limit, upper_limit)

    def set_colormap(self, new_colormap: str) -> None:
        """Sets the colormap of the image plot.

        Args:
            new_colormap: The name of the colormap to set. (Any colormap
                support by matplotlib is valid)
        """
        if self.img_plot is not None:
            self.img_plot.set_cmap(new_colormap)
        self.colormap = new_colormap

    def connect_selection(self) -> None:
        """Connects a `RetangleSelector` to the image plot's canvas."""
        self.rectangle_selection = RectangleSelector(
            self.ax,
            minspanx=5,
            minspany=5,
            useblit=True,
            use_data_coordinates=True,
            interactive=True,
            drag_from_anywhere=True,
        )
        self.rectangle_selection.add_state("square")
        self.rectangle_selection.set_active(False)

    def selection_set_active(self, value: bool):
        """Sets the rectangle selection on the image plot's canvas as active
            or non-active.

        Args:
            value: `True` for activating, `False` for non-active.
        """

        self.info.is_selection_active = value
        if self.rectangle_selection is not None:
            self.rectangle_selection.set_active(value)

    def get_selection(
        self,
    ) -> (
        tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
            tuple[int, int],
        ]
        | None
    ):
        """Get the coordinates of the `RectangleSelection` if one exists.

        Returns:
            The coordinates of `RectangleSelection`'s corners start from upper
            left, moving clockwise.
        """
        # Corners of rectangle in data coordinates from lower left, moving clockwise.
        if self.rectangle_selection is not None:
            corners = self.rectangle_selection.corners
            ul = (round(corners[0][0]), round(corners[1][0]))
            ur = (round(corners[0][1]), round(corners[1][1]))
            lr = (round(corners[0][2]), round(corners[1][2]))
            ll = (round(corners[0][3]), round(corners[1][3]))
            return (ul, ur, lr, ll)
        return None

    def start_processing(self, message: str) -> None:
        """Shows an indicator on `MovieWindow`, indicating that a processing
        function is running.

        Args:
            message: The message displayed next to the indicator.
        """
        self.process_indicator.status_label.setText(message)
        self.process_indicator.show()

    def end_processing(self) -> None:
        """Hides the processing indicator and recreates the plot."""
        self.process_indicator.hide()
        self.recreate_plot()

    def update_frame(self) -> None:
        """Update the image plot to show the next frame and the inputs to show
        this current frame number.
        """
        self.img_plot.set_data(self.plot_data[self.current_frame_num])
        self.img_plot.figure.canvas.draw()
        self.movie_controls.curr_frame_input.setValue(self.current_frame_num)
        self.movie_controls.current_frame_lbl.setText(f"/{self.num_frames - 1}")

    def start_playing(self) -> None:
        """Starts playing the movie."""
        fps = self.movie_controls.fps_input.value()
        update_time = 1 / fps * 1000

        self.timer = self.canvas.new_timer(update_time)
        self.timer.add_callback(self.on_next_btn_clicked)
        self.timer.start()

    def stop_playing(self) -> None:
        """Stops playing the movie."""
        self.timer.stop()

    def on_prev_btn_clicked(self) -> None:
        """Callback for the previous-frame-button (<<)."""
        if self.current_frame_num > 0:
            self.current_frame_num -= 1
        else:
            self.current_frame_num = self.num_frames - 1

        self.update_frame()

    def on_next_btn_clicked(self) -> None:
        """Callback for the next-frame-button (>>)."""
        if self.current_frame_num < self.num_frames - 1:
            self.current_frame_num += 1
        else:
            self.current_frame_num = 0

        self.update_frame()

    def on_first_btn_clicked(self) -> None:
        """Callback for the first-frame-button (>>|)."""
        self.current_frame_num = 0
        self.update_frame()

    def on_last_btn_clicked(self) -> None:
        """Callback for the last-frame-button (|<<)."""
        self.current_frame_num = self.num_frames - 1
        self.update_frame()

    def on_current_frame_changed(self, value: int) -> None:
        """Callback for the current-frame input (on change).

        Args:
            value: The new frame number.
        """
        if value > self.num_frames - 1:
            self.current_frame_num = self.num_frames - 1
        elif value < 0:
            self.current_frame_num = 0
        else:
            self.current_frame_num = value

        self.update_frame()

    def on_play_btn_clicked(self) -> None:
        """Callback for the play-button (|>)."""
        if self.movie_controls.play_btn.isChecked():
            self.start_playing()
        else:
            self.stop_playing()

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        """Overrides the `focusInEvent` of `super`. Additionally to the
        default behavior, a signal is sent that the window is focused.
        """
        print(f"Focused: {self.info.id_}")
        self.window_focused.emit(self.info)
        super().focusInEvent(event)

    @override
    def closeEvent(self, event: QCloseEvent) -> None:
        """Overrides the `closeEvent` of `super`. Additionally to the
        default behavior, a signal is sent that the window was is closed.
        """
        print(f"Closed {self.info}")
        self.timer.stop()
        self.window_closed.emit(self.info)
        super().closeEvent(event)


@final
class MovieControlButton(QPushButton):
    """A button of [`MovieControls`][pyfast_ui.movie_window.MovieControls].

    Args:
        icon: The button's icon.
        checkable: Decides if the button is checkable or not.
    """

    def __init__(self, icon: QStyle.StandardPixmap, checkable: bool = False) -> None:
        super().__init__()
        self.setIcon(self.style().standardIcon(icon))
        self.setIconSize(QSize(30, 30))
        self.setFixedSize(35, 35)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setCheckable(checkable)


@final
class MovieControlSpinBox(QSpinBox):
    """A spinbox of [`MovieControls`][pyfast_ui.movie_window.MovieControls].

    Args:
        focus_signal: The signal for sending if the `MovieWindow` which
            contains this `MovieControlsSpinBox` is focused.
        movie_info: The `MovieInfo` of the `MovieWindow` which contains this
            `MovieControlSpinBox`.
    """

    def __init__(self, focus_signal: SignalInstance, movie_info: MovieInfo) -> None:
        super().__init__()
        self.focus_signal = focus_signal
        self.movie_info = movie_info
        self.setFocusPolicy(Qt.StrongFocus)

    @override
    def focusInEvent(self, event: QFocusEvent) -> None:
        """Overrides the `focusInEvent` of `super`. Additionally, a signal gets
        emitted.
        """
        self.focus_signal.emit(self.movie_info)
        super().focusInEvent(event)


@final
class MovieControls(QWidget):
    """The controls of a [`MovieWindow`][pyfast_ui.movie_window.MovieWindow].

    Args:
        num_frames: The number of frames of the `MovieWindows`'s `FastMovie`.
        fps: The default frames-per-second.
        focus_signal: The signal that gets emitted when this `MovieControls`'s
            is focused.
        movie_info: The `MovieInfo` of the `MovieWindow` that contains this
            `MovieControls`.
    """

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
        """Sets the frames-number label to the new total number of frames.

        Args:
            new_num_frames: The new total number of frames.
        """
        self.current_frame_lbl.setText(f"/{new_num_frames}")


@final
class ProcessIndicator(QWidget):
    """An indicator shown when a processing function is running.

    Args:
        label: The message next to the processing indicator.
    """

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
