from typing import Callable, final
from scipy.stats import percentileofscore
import numpy as np

from PySide6.QtGui import Qt
from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout, QWidget
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider
from pyfast_ui.custom_widgets import LabeledDoubleSpinBoxes, LabeledSpinBoxes
from pyfastspm import FastMovie

from pyfast_ui.movie_window import MovieInfo


@final
class HistogramWindow(QWidget):
    def __init__(
        self,
        fast_movie: FastMovie,
        movie_info: MovieInfo,
        update_callback: Callable[[float, float], None],
    ) -> None:
        super().__init__()
        self.data = fast_movie.data.flatten()
        self.info = movie_info
        self.update_callback = update_callback
        self.setWindowTitle(f"{self.info.filename}({self.info.id_})")
        self.setFocusPolicy(Qt.StrongFocus)

        # Guard flag to avoid recursive updates.
        self._updating = False

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.canvas = FigureCanvas(Figure(figsize=(6, 2)))

        self.ax = None
        self.hist_plot = None
        self.lower_limit_line = None
        self.upper_limit_line = None

        self.data_min, self.data_max = self.data.min(), self.data.max()
        self.limit_absolute = LabeledDoubleSpinBoxes("Limit absolute", (0, 0))
        self.limit_absolute.spinbox_left.setMinimum(self.data_min)
        self.limit_absolute.spinbox_right.setMaximum(self.data_max)
        self.limit_absolute.setValue((self.data_min, self.data_max))
        self.limit_absolute.setFixedWidth(300)

        self.limit_percent = LabeledDoubleSpinBoxes("Limit percentile", (0, 100))
        self.limit_percent.spinbox_left.setMinimum(0)
        self.limit_percent.spinbox_right.setMaximum(100)
        self.limit_percent.setFixedWidth(300)

        inputs_layout = QVBoxLayout()
        inputs_layout.addWidget(self.limit_absolute)
        inputs_layout.addWidget(self.limit_percent)

        # Layout
        layout.addWidget(NavigationToolbar(self.canvas, self))
        layout.addWidget(self.canvas)
        layout.addLayout(inputs_layout)
        layout.setStretch(1, 2)

        self.limit_absolute.value_changed.connect(self.on_change_limit_absolute)
        self.limit_percent.value_changed.connect(self.on_change_limit_percent)

        self.create_plot()

    def contrast_percent(self) -> tuple[float, float]:
        min, max = self.limit_percent.value()
        return (min / 100, max / 100)

    def create_plot(self) -> None:
        self.ax = self.canvas.figure.subplots()
        self.canvas.figure.subplots_adjust(bottom=0.3)
        _, _, self.hist_plot = self.ax.hist(self.data, bins=500)

        slider_ax = self.canvas.figure.add_axes([0.20, 0.1, 0.60, 0.03])
        self.slider = RangeSlider(
            slider_ax,
            "Limits",
            self.data_min,
            self.data_max,
            valinit=(self.data_min, self.data_max),
        )
        self.slider.on_changed(self.slider_update)

        self.lower_limit_line = self.ax.axvline(self.slider.val[0], color="black")
        self.upper_limit_line = self.ax.axvline(self.slider.val[1], color="black")

    def slider_update(self, value) -> None:
        # Prevent handling slider callback when we're already updating from spinboxes.
        if self._updating:
            return

        min_absolute, max_absolute = value

        self.lower_limit_line.set_xdata([min_absolute, min_absolute])
        self.upper_limit_line.set_xdata([max_absolute, max_absolute])
        self.canvas.draw_idle()

        # Block spin box signals while updating them.
        self._updating = True
        try:
            self.update_callback(min_absolute, max_absolute)
            self.limit_absolute.blockSignals(True)
            self.limit_percent.blockSignals(True)
            try:
                self.limit_absolute.setValue((min_absolute, max_absolute))
                # Convert to percentile based on data range.
                # min_percent = ( (min_val - self.data_min) / (self.data_max - self.data_min)) * 100
                # max_percent = ( (max_val - self.data_min) / (self.data_max - self.data_min)) * 100

                min_percentile = percentileofscore(self.data, min_absolute, kind="rank")
                max_percentile = percentileofscore(self.data, max_absolute, kind="rank")
                self.limit_percent.setValue((min_percentile, max_percentile))

            finally:
                self.limit_absolute.blockSignals(False)
                self.limit_percent.blockSignals(False)
        finally:
            self._updating = False

    def on_change_limit_absolute(self, new_values: tuple[float, float]) -> None:
        # Avoid recursion if we're already updating.
        if self._updating:
            return
        self._updating = True
        try:
            min_absolute, max_absolute = new_values
            self.update_callback(min_absolute, max_absolute)
            self.lower_limit_line.set_xdata([min_absolute, min_absolute])
            self.upper_limit_line.set_xdata([max_absolute, max_absolute])
            self.canvas.draw_idle()

            # min_percent = ( (min_absolute - self.data_min) / (self.data_max - self.data_min)) * 100
            # max_percent = ( (max_absolute - self.data_min) / (self.data_max - self.data_min)) * 100

            min_percentile = percentileofscore(self.data, min_absolute, kind="rank")
            max_percentile = percentileofscore(self.data, max_absolute, kind="rank")

            # Block the percent spin box signals before updating.
            self.limit_percent.blockSignals(True)
            try:
                self.limit_percent.setValue((min_percentile, max_percentile))
            finally:
                self.limit_percent.blockSignals(False)

            # Update the slider; its callback will check the _updating flag.
            self.slider.set_val([min_absolute, max_absolute])
        finally:
            self._updating = False

    def on_change_limit_percent(self, new_value: tuple[int, int]) -> None:
        # Avoid recursion if we're already updating.
        if self._updating:
            return
        self._updating = True
        try:
            min_percent, max_percent = new_value
            # min_absolute = self.data_min + (min_percent / 100) * ( self.data_max - self.data_min)
            # max_absolute = self.data_min + (max_percent / 100) * ( self.data_max - self.data_min)
            min_absolute = np.percentile(self.data, min_percent)
            max_absolute = np.percentile(self.data, max_percent)
            self.update_callback(min_absolute, max_absolute)
            self.lower_limit_line.set_xdata([min_absolute, min_absolute])
            self.upper_limit_line.set_xdata([max_absolute, max_absolute])
            self.canvas.draw_idle()

            # Block the absolute spin box signals before updating.
            self.limit_absolute.blockSignals(True)
            try:
                self.limit_absolute.setValue((min_absolute, max_absolute))
            finally:
                self.limit_absolute.blockSignals(False)

            # Update the slider; its callback will check the _updating flag.
            self.slider.set_val([min_absolute, max_absolute])
        finally:
            self._updating = False
