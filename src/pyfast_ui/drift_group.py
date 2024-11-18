from typing import Literal, TypeAlias
from PySide6.QtWidgets import QButtonGroup, QCheckBox, QGroupBox, QRadioButton, QSpinBox, QVBoxLayout


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
        self._drift_type_group = QGroupBox("Drift Type", self)
        drift_type_layout = QVBoxLayout()
        self._drift_type_group.setLayout(drift_type_layout)

        self._drift_type_common = QRadioButton("common", self)
        self._drift_type_full =  QRadioButton("full", self)

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

        # Known Drift
        self._known_drift = QCheckBox("Known Drift", self)
        self._known_drift.setChecked(known_drift)

        # Add drift type widgets to the drift type group box
        drift_type_layout.addWidget(self._drift_type_common)
        drift_type_layout.addWidget(self._drift_type_full)

        # Add Widgets to main group box
        layout.addWidget(self._fft_drift)
        layout.addWidget(self._drift_type_group)
        layout.addWidget(self._stepsize)
        layout.addWidget(self._known_drift)
