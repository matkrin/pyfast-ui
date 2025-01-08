from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QPushButton,
)


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
    ) -> None:
        super().__init__("FFT Filters")
        layout = QGridLayout()
        self.setLayout(layout)

        self._filter_x = QCheckBox("Filter x", self)
        self._filter_x.setChecked(filter_x)

        self._filter_y = QCheckBox("Filter y", self)
        self._filter_y.setChecked(filter_y)

        self._filter_x_overtones = QCheckBox("Filter x Overtones", self)
        self._filter_x_overtones.setChecked(filter_x_overtones)

        self._filter_high_pass = QCheckBox("Filter High Pass", self)
        self._filter_high_pass.setChecked(filter_high_pass)

        self._filter_pump = QCheckBox("Filter pump", self)
        self._filter_pump.setChecked(filter_pump)

        self._filter_noise = QCheckBox("Filter noise", self)
        self._filter_noise.setChecked(filter_noise)

        self._display_spectrum = QCheckBox("Display Spectrum", self)
        self._display_spectrum.setChecked(display_spectrum)

        self.apply_btn = QPushButton("Apply")

        # Add Widgets to Layout
        # Column 0
        layout.addWidget(self._filter_x, 0, 0)
        layout.addWidget(self._filter_y, 1, 0)
        layout.addWidget(self._filter_x_overtones, 2, 0)
        layout.addWidget(self._display_spectrum, 3, 0)
        # Column 1
        layout.addWidget(self._filter_high_pass, 0, 1)
        layout.addWidget(self._filter_pump, 1, 1)
        layout.addWidget(self._filter_noise, 2, 1)
        layout.addWidget(self.apply_btn, 4, 0, 1, 2)

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
