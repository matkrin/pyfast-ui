from PySide6.QtWidgets import QCheckBox, QGroupBox, QPushButton, QVBoxLayout


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
        layout = QVBoxLayout()
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
        layout.addWidget(self._filter_x)
        layout.addWidget(self._filter_y)
        layout.addWidget(self._filter_x_overtones)
        layout.addWidget(self._filter_high_pass)
        layout.addWidget(self._filter_pump)
        layout.addWidget(self._filter_noise)
        layout.addWidget(self._display_spectrum)
        layout.addWidget(self.apply_btn)

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
