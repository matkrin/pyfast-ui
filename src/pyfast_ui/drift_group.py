from typing import Literal, TypeAlias, final
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
)


@final
class DriftGroup(QGroupBox):
    def __init__(
        self,
        drift_algorithm: str,
        fft_drift: bool,
        drifttype: str,
        stepsize: int,
        known_drift: bool,
        stackreg_reference: str,
    ) -> None:
        super().__init__("Drift")
        layout = QVBoxLayout()
        self.setLayout(layout)

        ## Drift Type
        self._drift_type_common = QRadioButton("common", self)
        self._drift_type_full = QRadioButton("full", self)

        self._drift_type_button_group = QButtonGroup(self)
        self._drift_type_button_group.addButton(self._drift_type_common)
        self._drift_type_button_group.addButton(self._drift_type_full)

        drift_type_layout = QHBoxLayout()
        drift_type_layout.addWidget(self._drift_type_common)
        drift_type_layout.addWidget(self._drift_type_full)

        match drifttype:
            case "common":
                self._drift_type_common.setChecked(True)
            case "full":
                self._drift_type_full.setChecked(True)

        self.correlation_group = CorreclationGroup(
            fft_drift, drifttype, stepsize, known_drift
        )

        self._drift_algo_correlation = QRadioButton("correlation", self)
        self._drift_algo_stackreg = QRadioButton("stackreg", self)

        self._drift_algo_button_group = QButtonGroup(self)
        self._drift_algo_button_group.addButton(self._drift_algo_correlation)
        self._drift_algo_button_group.addButton(self._drift_algo_stackreg)

        drift_algo_layout = QHBoxLayout()
        drift_algo_layout.addWidget(self._drift_algo_correlation)
        drift_algo_layout.addWidget(self._drift_algo_stackreg)

        match drift_algorithm:
            case "correlation":
                self._drift_algo_correlation.setChecked(True)
            case "stackreg":
                self._drift_algo_stackreg.setChecked(True)

        self.correlation_group = CorreclationGroup(
            fft_drift, drifttype, stepsize, known_drift
        )

        self.stackreg_group = StackregGroup(stackreg_reference)

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addLayout(drift_type_layout)
        layout.addLayout(drift_algo_layout)
        layout.addWidget(self.correlation_group)
        layout.addWidget(self.stackreg_group)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.new_btn)
        layout.addLayout(btn_layout)

    @property
    def drift_algorithm(self) -> str:
        selected_button = self._drift_algo_button_group.checkedButton()
        return selected_button.text()

    @property
    def stackreg_reference(self) -> str:
        return self.stackreg_group.reference

    @property
    def fft_drift(self) -> bool:
        return self.correlation_group.fft_drift.isChecked()

    @property
    def drift_type(self) -> str:
        selected_button = self._drift_type_button_group.checkedButton()
        return selected_button.text()

    @property
    def stepsize(self) -> int:
        return self.correlation_group.stepsize.value()

    @property
    def known_drift(self) -> bool:
        return self.correlation_group.known_drift.isChecked()


@final
class CorreclationGroup(QGroupBox):
    def __init__(
        self, fft_drift: bool, drifttype: str, stepsize: int, known_drift: bool
    ) -> None:
        super().__init__("Correlation")
        layout = QVBoxLayout()
        self.setLayout(layout)

        ## FFT Drift
        self.fft_drift = QCheckBox("FFT Drift", self)
        self.fft_drift.setChecked(fft_drift)

        # Stepsize
        self.stepsize = QSpinBox(self)
        self.stepsize.setRange(0, 5000)
        self.stepsize.setValue(stepsize)
        self._stepsize_lbl = QLabel("Stepsize")
        stepsize_layout = QHBoxLayout()

        # Known Drift
        self.known_drift = QCheckBox("Known Drift", self)
        self.known_drift.setChecked(known_drift)

        # Add stepsize widgets to stepsize Layout
        stepsize_layout.addWidget(self._stepsize_lbl)
        stepsize_layout.addWidget(self.stepsize)

        # Add Widgets and Layouts to main group box
        layout.addWidget(self.fft_drift)
        layout.addLayout(stepsize_layout)
        layout.addWidget(self.known_drift)


@final
class StackregGroup(QGroupBox):
    def __init__(self, reference: str) -> None:
        super().__init__("StackReg")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self._reference_previous = QRadioButton("previous", self)
        self._reference_first = QRadioButton("first", self)
        self._reference_mean = QRadioButton("mean", self)

        self._reference_btn_group = QButtonGroup(self)
        self._reference_btn_group.addButton(self._reference_previous)
        self._reference_btn_group.addButton(self._reference_first)
        self._reference_btn_group.addButton(self._reference_mean)

        match reference:
            case "previous":
                self._reference_previous.setChecked(True)
            case "first":
                self._reference_first.setChecked(True)
            case "mean":
                self._reference_mean.setChecked(True)

        reference_layout = QHBoxLayout()
        reference_layout.addWidget(self._reference_previous)
        reference_layout.addWidget(self._reference_first)
        reference_layout.addWidget(self._reference_mean)

        layout.addLayout(reference_layout)

    @property
    def reference(self) -> str:
        selected_button = self._reference_btn_group.checkedButton()
        return selected_button.text()
