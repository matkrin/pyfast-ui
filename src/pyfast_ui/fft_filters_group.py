import dataclasses
from typing import Self, final

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
)

from pyfast_ui.config import FftFilterConfig
from pyfast_ui.custom_widgets import (
    LabeledDoubleSpinBoxes,
    LabeledSpinBox,
    LabeledSpinBoxes,
)


@final
class FFTFiltersGroup(QGroupBox):
    def __init__(
        self,
        filter_x: bool,
        filter_y: bool,
        filter_x_overtones: bool,
        filter_high_pass: bool,
        filter_pump: bool,
        filter_noise: bool,
        display_spectrum: bool,
        filter_broadness: int | None,
        num_x_overtones: int,
        high_pass_params: tuple[float, float],
        num_pump_overtones: int,
        pump_freqs: tuple[float, float],
        fft_display_range: tuple[int, int],
    ) -> None:
        super().__init__("FFT Filters")
        layout = QGridLayout()
        self.setLayout(layout)

        self._filter_x = QCheckBox("Filter x")
        self._filter_x.setChecked(filter_x)

        self._filter_y = QCheckBox("Filter y")
        self._filter_y.setChecked(filter_y)

        self._filter_x_overtones = QCheckBox("Filter x Overtones")
        self._filter_x_overtones.setChecked(filter_x_overtones)

        self._filter_high_pass = QCheckBox("Filter High Pass")
        self._filter_high_pass.setChecked(filter_high_pass)

        self._filter_pump = QCheckBox("Filter pump")
        self._filter_pump.setChecked(filter_pump)

        self._filter_noise = QCheckBox("Filter noise")
        self._filter_noise.setChecked(filter_noise)

        self._display_spectrum = QCheckBox("Display Spectrum")
        self._display_spectrum.setChecked(display_spectrum)

        # Advanced settings
        self._filter_broadness = LabeledSpinBox(
            "Filter broadness", filter_broadness or 0
        )
        self._num_x_overtones = LabeledSpinBox("Num x overtones", num_x_overtones)
        self._high_pass_params = LabeledDoubleSpinBoxes(
            "High pass params", high_pass_params
        )
        self._num_pump_overtones = LabeledSpinBox("Num pump overtones", num_x_overtones)

        pump_freqs_lbl = QLabel("Pump freqs")
        self._pump_freqs = QLineEdit()
        self._pump_freqs.setText(",".join(str(x) for x in pump_freqs))
        pump_freqs_layout = QHBoxLayout()
        pump_freqs_layout.addWidget(pump_freqs_lbl)
        pump_freqs_layout.addWidget(self._pump_freqs)

        self._fft_display_range = LabeledSpinBoxes(
            "FFT display range", fft_display_range
        )

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        # Add Widgets to Layout
        layout.addWidget(self._filter_x, 0, 0)
        layout.addWidget(self._filter_y, 1, 0)
        layout.addWidget(self._filter_x_overtones, 2, 0)
        layout.addWidget(self._filter_high_pass, 0, 1)
        layout.addWidget(self._filter_pump, 1, 1)
        layout.addWidget(self._filter_noise, 2, 1)

        layout.addWidget(self._display_spectrum, 3, 0)

        layout.addWidget(self._filter_broadness, 4, 0)
        layout.addWidget(self._num_x_overtones, 4, 1)
        layout.addWidget(self._high_pass_params, 5, 0, 1, 2)
        layout.addWidget(self._fft_display_range, 6, 0, 1, 2)
        layout.addLayout(pump_freqs_layout, 7, 0, 1, 2)

        layout.addWidget(self.apply_btn, 8, 0)
        layout.addWidget(self.new_btn, 8, 1)

    @classmethod
    def from_config(cls, fft_filter_config: FftFilterConfig) -> Self:
        return cls(**dataclasses.asdict(fft_filter_config))

    @property
    def filterparams(self) -> list[bool]:
        return [
            self.filter_x,
            self.filter_y,
            self.filter_x_overtones,
            self.filter_high_pass,
            self.filter_pump,
            self.filter_noise,
            self.display_spectrum,
        ]

    @property
    def filter_x(self) -> bool:
        return self._filter_x.isChecked()

    @property
    def filter_y(self) -> bool:
        return self._filter_y.isChecked()

    @property
    def filter_x_overtones(self) -> bool:
        return self._filter_x_overtones.isChecked()

    @property
    def filter_high_pass(self) -> bool:
        return self._filter_high_pass.isChecked()

    @property
    def filter_pump(self) -> bool:
        return self._filter_pump.isChecked()

    @property
    def filter_noise(self) -> bool:
        return self._filter_noise.isChecked()

    @property
    def display_spectrum(self) -> bool:
        return self._display_spectrum.isChecked()

    @property
    def filter_broadness(self) -> int | None:
        filter_broadness = self._filter_broadness.value()
        return None if filter_broadness == 0 else filter_broadness

    @property
    def num_x_overtones(self) -> int:
        return self._num_x_overtones.value()

    @property
    def high_pass_params(self) -> tuple[float, float]:
        return self._high_pass_params.value()

    @property
    def num_pump_overtones(self) -> int:
        return self._num_pump_overtones.value()

    @property
    def pump_freqs(self) -> list[float]:
        pump_freqs_text = self._pump_freqs.text()
        return [float(x) for x in pump_freqs_text.split(",")]

    @property
    def fft_display_range(self) -> tuple[int, int]:
        return self._fft_display_range.value()
