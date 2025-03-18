from typing import Callable, final

import numpy as np
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar
from matplotlib.backends.backend_qtagg import FigureCanvas
from matplotlib.figure import Figure
from matplotlib.widgets import RangeSlider
from PySide6.QtGui import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget
from scipy.stats import percentileofscore

from pyfast_ui.custom_widgets import LabeledDoubleSpinBoxes
from pyfast_ui.movie_window import MovieInfo
from pyfast_ui.pyfast_re.fast_movie import FastMovie


@final
class HistogramWindow(QWidget):
    """A window displaying a histogram plot of the intensity values of all
    frames of a `FastMovie` and input elements to adjust the color range
    (contrast) of the `FastMovie`.

    Args:
        fast_movie: `FastMovie` to adjust color range.
        movie_info: App specific info about the `FastMovie`.
        update_callback: Callback that runs to update the `FastMovie`'s color
            range.
    """

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

        self._canvas = FigureCanvas(Figure(figsize=(6, 2)))

        self._ax = None
        self._hist_plot = None
        self._lower_limit_line = None
        self._upper_limit_line = None

        self._data_min, self._data_max = self.data.min(), self.data.max()
        self._limit_absolute = LabeledDoubleSpinBoxes("Limit absolute", (0, 0))
        self._limit_absolute.spinbox_left.setMinimum(self._data_min)
        self._limit_absolute.spinbox_right.setMaximum(self._data_max)
        self._limit_absolute.setValue((self._data_min, self._data_max))
        self._limit_absolute.setFixedWidth(300)

        self._limit_percentile = LabeledDoubleSpinBoxes("Limit percentile", (0, 100))
        self._limit_percentile.spinbox_left.setMinimum(0)
        self._limit_percentile.spinbox_right.setMaximum(100)
        self._limit_percentile.setFixedWidth(300)

        inputs_layout = QVBoxLayout()
        inputs_layout.addWidget(self._limit_absolute)
        inputs_layout.addWidget(self._limit_percentile)

        # Layout
        layout.addWidget(NavigationToolbar(self._canvas, self))
        layout.addWidget(self._canvas)
        layout.addLayout(inputs_layout)
        layout.setStretch(1, 2)

        self._limit_absolute.value_changed.connect(self._on_change_limit_absolute)
        self._limit_percentile.value_changed.connect(self._on_change_limit_percentile)

        self._create_plot()

    def contrast_percentile(self) -> tuple[float, float]:
        """Get the currently selected contrast percentile values as a fraction of 1."""
        min, max = self._limit_percentile.value()
        return (min / 100, max / 100)

    def _create_plot(self) -> None:
        """Create the histogram plot."""
        self._ax = self._canvas.figure.subplots()
        self._canvas.figure.subplots_adjust(bottom=0.3)
        _, _, self._hist_plot = self._ax.hist(self.data, bins=500)

        slider_ax = self._canvas.figure.add_axes([0.20, 0.1, 0.60, 0.03])
        self.slider = RangeSlider(
            slider_ax,
            "Limits",
            self._data_min,
            self._data_max,
            valinit=(self._data_min, self._data_max),
        )
        self.slider.on_changed(self._slider_update)

        self._lower_limit_line = self._ax.axvline(self.slider.val[0], color="black")
        self._upper_limit_line = self._ax.axvline(self.slider.val[1], color="black")

    def _slider_update(self, value) -> None:
        """Callback that gets called when the range slider values change."""
        # Prevent handling slider callback when we're already updating from spinboxes.
        if self._updating:
            return

        min_absolute, max_absolute = value

        self._lower_limit_line.set_xdata([min_absolute, min_absolute])
        self._upper_limit_line.set_xdata([max_absolute, max_absolute])
        self._canvas.draw_idle()

        # Block spin box signals while updating them.
        self._updating = True
        try:
            self.update_callback(min_absolute, max_absolute)
            self._limit_absolute.blockSignals(True)
            self._limit_percentile.blockSignals(True)
            try:
                self._limit_absolute.setValue((min_absolute, max_absolute))
                # Convert to percentile based on data range.
                # min_percent = ( (min_val - self.data_min) / (self.data_max - self.data_min)) * 100
                # max_percent = ( (max_val - self.data_min) / (self.data_max - self.data_min)) * 100

                min_percentile = percentileofscore(self.data, min_absolute, kind="rank")
                max_percentile = percentileofscore(self.data, max_absolute, kind="rank")
                self._limit_percentile.setValue((min_percentile, max_percentile))

            finally:
                self._limit_absolute.blockSignals(False)
                self._limit_percentile.blockSignals(False)
        finally:
            self._updating = False

    def _on_change_limit_absolute(self, new_values: tuple[float, float]) -> None:
        """Callback that gets called when the values of the absolute input change."""
        # Avoid recursion if we're already updating.
        if self._updating:
            return
        self._updating = True
        try:
            min_absolute, max_absolute = new_values
            self.update_callback(min_absolute, max_absolute)
            self._lower_limit_line.set_xdata([min_absolute, min_absolute])
            self._upper_limit_line.set_xdata([max_absolute, max_absolute])
            self._canvas.draw_idle()

            # min_percent = ( (min_absolute - self.data_min) / (self.data_max - self.data_min)) * 100
            # max_percent = ( (max_absolute - self.data_min) / (self.data_max - self.data_min)) * 100

            min_percentile = percentileofscore(self.data, min_absolute, kind="rank")
            max_percentile = percentileofscore(self.data, max_absolute, kind="rank")

            # Block the percent spin box signals before updating.
            self._limit_percentile.blockSignals(True)
            try:
                self._limit_percentile.setValue((min_percentile, max_percentile))
            finally:
                self._limit_percentile.blockSignals(False)

            # Update the slider; its callback will check the _updating flag.
            self.slider.set_val([min_absolute, max_absolute])
        finally:
            self._updating = False

    def _on_change_limit_percentile(self, new_value: tuple[int, int]) -> None:
        """Callback that gets called when values of the percentile input change."""
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
            self._lower_limit_line.set_xdata([min_absolute, min_absolute])
            self._upper_limit_line.set_xdata([max_absolute, max_absolute])
            self._canvas.draw_idle()

            # Block the absolute spin box signals before updating.
            self._limit_absolute.blockSignals(True)
            try:
                self._limit_absolute.setValue((min_absolute, max_absolute))
            finally:
                self._limit_absolute.blockSignals(False)

            # Update the slider; its callback will check the _updating flag.
            self.slider.set_val([min_absolute, max_absolute])
        finally:
            self._updating = False
