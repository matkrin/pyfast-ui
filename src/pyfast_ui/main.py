import sys
from typing import override
from PySide6.QtCore import QSize, Slot
from PySide6.QtGui import QIntValidator, QKeyEvent, QKeySequence, QShortcut, Qt
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

# from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.figure import Figure

import pyfastspm as pf

FAST_FILE = "/Users/matthias/github/pyfastspm/examples/F20190424_1.h5"


class MainGui(QMainWindow):
    """Main GUI"""

    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("PyfastSPM")
        self.setGeometry(100, 100, 700, 400)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_layout = QVBoxLayout()
        self.central_widget.setLayout(self.central_layout)

        self.open_btn = QPushButton("Open")
        _ = self.open_btn.clicked.connect(self.on_open_btn_click)

        self.central_layout.addWidget(self.open_btn)

        self.plot_windows: list[QWidget] = []

    def on_open_btn_click(self) -> None:
        print("clicked")
        plot_window = MovieWindow(FAST_FILE)
        self.plot_windows.append(plot_window)
        plot_window.show()


class MovieWindow(QWidget):
    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.ft = pf.FastMovie(str(filename), y_phase=0)
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

        self.curr_frame_input = QLineEdit(str(self.current_frame_num))
        intValidator = QIntValidator()
        intValidator.setRange(0, self.num_frames - 1)
        self.curr_frame_input.setValidator(intValidator)
        self.curr_frame_input.setFixedWidth(40)
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

        self.fps_input = QLineEdit("24")
        self.fps_input.setFixedWidth(35)
        intValidator = QIntValidator()
        intValidator.setRange(0, 50)
        self.fps_input.setValidator(intValidator)
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

        _ = self.curr_frame_input.returnPressed.connect(self.on_current_frame_return)

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
        self.curr_frame_input.setText(str(self.current_frame_num))
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
    def on_current_frame_return(self) -> None:
        input = int(self.curr_frame_input.text())
        if input > self.num_frames - 1:
            self.current_frame_num = self.num_frames - 1
        elif input < 0:
            self.current_frame_num = 0
        else:
            self.current_frame_num = input

        self.update_frame()

    @Slot()
    def on_play_btn_clicked(self) -> None:
        fps = int(self.fps_input.text())
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
