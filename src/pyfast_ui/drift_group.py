from typing import Self, final

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

from pyfast_ui.config import DriftConfig
from pyfast_ui.custom_widgets import LabeledSpinBox


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
        boxcar: int,
        median_filter: bool,
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

        self._drift_algo_correlation = QRadioButton("correlation", self)
        self._drift_algo_stackreg = QRadioButton("stackreg", self)
        self._known_drift = QRadioButton("known", self)

        self._drift_algo_button_group = QButtonGroup(self)
        self._drift_algo_button_group.addButton(self._drift_algo_correlation)
        self._drift_algo_button_group.addButton(self._drift_algo_stackreg)
        self._drift_algo_button_group.addButton(self._known_drift)

        drift_algo_layout = QVBoxLayout()
        drift_algo_layout.addWidget(QLabel("Mode"))
        drift_algo_layout.addWidget(self._drift_algo_correlation)
        drift_algo_layout.addWidget(self._drift_algo_stackreg)
        drift_algo_layout.addWidget(self._known_drift)

        match drift_algorithm:
            case "correlation" if not known_drift:
                self._drift_algo_correlation.setChecked(True)
            case "stackreg" if not known_drift:
                self._drift_algo_stackreg.setChecked(True)
            case _ if known_drift:
                self._known_drift.setChecked(True)

        self._drift_filter_group = DriftFilterGroup(boxcar, median_filter)

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
        layout.addWidget(self._drift_filter_group)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.new_btn)
        layout.addLayout(btn_layout)

    @property
    def drift_algorithm(self) -> str:
        selected_button = self._drift_algo_button_group.checkedButton()
        return selected_button.text()

    @drift_algorithm.setter
    def drift_algorithm(self, value: str) -> None:
        match value:
            case "correlation":
                self._drift_algo_correlation.setChecked(True)
            case "stackreg":
                self._drift_algo_stackreg.setChecked(True)
            case _:
                self._known_drift.setChecked(True)

    @property
    def stackreg_reference(self) -> str:
        return self.stackreg_group.reference

    @property
    def fft_drift(self) -> bool:
        # return self.correlation_group.fft_drift.isChecked()
        return True

    @fft_drift.setter
    def fft_drift(self, value: bool) -> None:
        self.correlation_group.fft_drift = value

    @property
    def drift_type(self) -> str:
        selected_button = self._drift_type_button_group.checkedButton()
        return selected_button.text()

    @drift_type.setter
    def drift_type(self, value: str) -> None:
        match value:
            case "common":
                self._drift_type_common.setChecked(True)
            case "full":
                self._drift_type_full.setChecked(True)
            case _:
                raise ValueError("drift type must be 'common' or 'full'")

    @property
    def stepsize(self) -> int:
        return self.correlation_group.stepsize.value()

    @stepsize.setter
    def stepsize(self, value: int) -> None:
        self.correlation_group.stepsize.setValue(value)

    @property
    def known_drift(self) -> bool:
        # return self.correlation_group.known_drift.isChecked()
        return self._known_drift.isChecked()

    @known_drift.setter
    def known_drift(self, value: bool) -> None:
        self._known_drift.setChecked(value)

    @property
    def boxcar(self) -> int:
        return self._drift_filter_group.boxcar.value()

    @boxcar.setter
    def boxcar(self, value: int) -> None:
        self._drift_filter_group.boxcar.setValue(value)

    @property
    def median_filter(self) -> bool:
        return self._drift_filter_group.median_filter.isChecked()

    @median_filter.setter
    def median_filter(self, value: bool) -> None:
        self._drift_filter_group.median_filter.setChecked(value)

    @classmethod
    def from_config(cls, drift_config: DriftConfig) -> Self:
        return cls(**drift_config.model_dump())

    def update_from_config(self, drift_config: DriftConfig) -> None:
        self.drift_algorithm = drift_config.drift_algorithm
        self.stackreg_group.reference = drift_config.stackreg_reference
        self.fft_drift = drift_config.fft_drift
        self.drift_type = drift_config.drifttype
        self.stepsize = drift_config.stepsize
        self.known_drift = drift_config.known_drift
        self.boxcar = drift_config.boxcar
        self.median_filter = drift_config.median_filter

    def to_config(self) -> DriftConfig:
        return DriftConfig(
            drift_algorithm=self.drift_algorithm,
            fft_drift=self.fft_drift,
            drifttype=self.drift_type,
            stepsize=self.stepsize,
            known_drift=self.known_drift,
            stackreg_reference=self.stackreg_reference,
            boxcar=self.boxcar,
            median_filter=self.median_filter,
        )


@final
class CorreclationGroup(QGroupBox):
    def __init__(
        self, fft_drift: bool, drifttype: str, stepsize: int, known_drift: bool
    ) -> None:
        super().__init__("Correlation")
        layout = QVBoxLayout()
        self.setLayout(layout)

        ## FFT Drift
        # self.fft_drift = QCheckBox("FFT Drift", self)
        # self.fft_drift.setChecked(fft_drift)
        self.fft_drift = fft_drift

        # Stepsize
        self.stepsize = QSpinBox(self)
        self.stepsize.setRange(0, 5000)
        self.stepsize.setValue(stepsize)
        self._stepsize_lbl = QLabel("Stepsize")
        stepsize_layout = QHBoxLayout()

        # Known Drift
        # self.known_drift = QCheckBox("Known Drift", self)
        # self.known_drift.setChecked(known_drift)

        # Add stepsize widgets to stepsize Layout
        stepsize_layout.addWidget(self._stepsize_lbl)
        stepsize_layout.addWidget(self.stepsize)

        # Add Widgets and Layouts to main group box
        # layout.addWidget(self.fft_drift)
        layout.addLayout(stepsize_layout)
        # layout.addWidget(self.known_drift)


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
            case _:
                raise ValueError(
                    "stack reference must be 'previous', 'first' or 'mean'"
                )

        reference_layout = QHBoxLayout()
        reference_layout.addWidget(self._reference_previous)
        reference_layout.addWidget(self._reference_first)
        reference_layout.addWidget(self._reference_mean)

        layout.addLayout(reference_layout)

    @property
    def reference(self) -> str:
        selected_button = self._reference_btn_group.checkedButton()
        return selected_button.text()

    @reference.setter
    def reference(self, value: str) -> None:
        match value:
            case "previous":
                self._reference_previous.setChecked(True)
            case "first":
                self._reference_first.setChecked(True)
            case "mean":
                self._reference_mean.setChecked(True)
            case _:
                raise ValueError(
                    "stack reference must be 'previous', 'first' or 'mean'"
                )


@final
class DriftFilterGroup(QGroupBox):
    def __init__(self, boxcar: int, median_filter: bool) -> None:
        super().__init__("Filter")
        layout = QHBoxLayout()
        self.setLayout(layout)

        self.boxcar = LabeledSpinBox("Boxcar width", boxcar)

        median_filter_label = QLabel("Median filter")
        self.median_filter = QCheckBox()
        self.median_filter.setChecked(median_filter)
        median_filter_layout = QHBoxLayout()
        median_filter_layout.addWidget(self.median_filter)
        median_filter_layout.addWidget(median_filter_label)

        layout.addWidget(self.boxcar)
        layout.addLayout(median_filter_layout)
