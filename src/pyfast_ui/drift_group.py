from typing import Literal, TypeAlias
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QRadioButton, QSpinBox, QVBoxLayout


DriftType: TypeAlias = Literal["common", "full"]


class DriftGroup(QGroupBox):
    def __init__(
        self, fft_drift: bool, drifttype: DriftType, stepsize: int, known_drift: bool
    ) -> None:
        super().__init__("Drift")
        layout = QVBoxLayout()
        self.setLayout(layout)

        ## FFT Drift
        self._fft_drift = QCheckBox("FFT Drift", self)
        self._fft_drift.setChecked(fft_drift)

        ## Drift Type
        drift_type_layout = QHBoxLayout()
        self._drift_type_common = QRadioButton("common", self)
        self._drift_type_full =  QRadioButton("full", self)
        drift_type_layout.addWidget(self._drift_type_common)
        drift_type_layout.addWidget(self._drift_type_full)


        self._button_group = QButtonGroup(self)
        self._button_group.addButton(self._drift_type_common)
        self._button_group.addButton(self._drift_type_full)

        match drifttype:
            case "common":
                self._drift_type_common.setChecked(True)
            case "full":
                self._drift_type_full.setChecked(True)

        # Stepsize
        self._stepsize = QSpinBox(self)
        self._stepsize.setRange(0, 5000)
        self._stepsize.setValue(stepsize)
        self._stepsize_lbl = QLabel("Stepsize")
        stepsize_layout = QHBoxLayout()

        # Known Drift
        self._known_drift = QCheckBox("Known Drift", self)
        self._known_drift.setChecked(known_drift)

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        # Add drift type widgets to the drift type group box
        drift_type_layout.addWidget(self._drift_type_common)
        drift_type_layout.addWidget(self._drift_type_full)

        # Add stepsize widgets to stepsize Layout
        stepsize_layout.addWidget(self._stepsize_lbl)
        stepsize_layout.addWidget(self._stepsize)

        # Add Widgets and Layouts to main group box
        layout.addWidget(self._fft_drift)
        layout.addLayout(drift_type_layout)
        layout.addLayout(stepsize_layout)
        layout.addWidget(self._known_drift)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.new_btn)
        layout.addLayout(btn_layout)

    @property
    def fft_drift(self) -> bool:
        return self._fft_drift.isChecked()

    @property
    def drift_type(self) -> DriftType:
        selected_button = self._button_group.checkedButton()
        return selected_button.text()

    @property
    def stepsize(self) -> int:
        return self._stepsize.value()

    @property
    def known_drift(self) -> bool:
        return self._known_drift.isChecked()
