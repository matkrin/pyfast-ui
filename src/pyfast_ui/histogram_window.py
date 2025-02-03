from typing import final

from PySide6.QtGui import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from pyfastspm import FastMovie

from pyfast_ui.movie_window import MovieInfo


@final
class HistogramWindow(QWidget):
    def __init__(self, fast_movie: FastMovie, movie_info: MovieInfo) -> None:
        super().__init__()
        self.data = fast_movie.data
        self.info = movie_info
        self.setWindowTitle(f"{self.info.filename}({self.info.id_})")
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.canvas = FigureCanvas(Figure(figsize=(4, 4)))

        self.ax = None
        self.hist_plot = None
        self.create_plot()

        # Layout
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        # Strech the canvas when window resizes
        layout.setStretch(1, 2)

    def create_plot(self) -> None:
        self.ax = self.canvas.figure.subplots()
        self.hist_plot = self.ax.hist(self.data, bins=500)

        # self.hist_plot.figure.tight_layout(pad=0)
