from typing import Self, final

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from pyfast_ui.config import PhaseConfig


@final
class PhaseGroup(QGroupBox):
    """Grouping of phase related widgets.

    Args:
        apply_auto_xphase: Whether to apply autmotically determined x-phase correction.
        additional_x_phase: Manually chosen additional x-phase.
        manual_y_phase: Manually chosen additional y-phase.
        index_frame_to_correlate: Index of the frame to correlate to.
        sigma_gauss:
    """
    def __init__(
        self,
        apply_auto_xphase: bool,
        additional_x_phase: int,
        manual_y_phase: int,
        index_frame_to_correlate: int,
        sigma_gauss: int,
    ) -> None:
        super().__init__("Phase Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        self._apply_auto_xphase = QCheckBox("Apply auto X-Phase", self)
        self._apply_auto_xphase.setChecked(apply_auto_xphase)

        additional_x_phase_lbl = QLabel("Additional x-phase")
        self._additional_x_phase = QSpinBox(self)
        self._additional_x_phase.setRange(-10, 10)
        self._additional_x_phase.setFixedWidth(50)
        self._additional_x_phase.setValue(additional_x_phase)
        additional_x_phase_layout = QHBoxLayout()
        additional_x_phase_layout.addWidget(additional_x_phase_lbl)
        additional_x_phase_layout.addWidget(self._additional_x_phase)

        manual_y_phase_lbl = QLabel("Manual y-phase")
        self._manual_y_phase = QSpinBox(self)
        self._manual_y_phase.setRange(-10, 10)
        self._manual_y_phase.setFixedWidth(50)
        self._manual_y_phase.setValue(manual_y_phase)
        manual_y_phase_layout = QHBoxLayout()
        manual_y_phase_layout.addWidget(manual_y_phase_lbl)
        manual_y_phase_layout.addWidget(self._manual_y_phase)

        index_frame_to_correlate_lbl = QLabel("Index frame to correlate")
        self._index_frame_to_correlate = QSpinBox(self)
        self._index_frame_to_correlate.setFixedWidth(50)
        self._index_frame_to_correlate.setValue(index_frame_to_correlate)
        index_frame_to_correlate_layout = QHBoxLayout()
        index_frame_to_correlate_layout.addWidget(index_frame_to_correlate_lbl)
        index_frame_to_correlate_layout.addWidget(self._index_frame_to_correlate)

        sigma_gauss_lbl = QLabel("Sigma gauss")
        self._sigma_gauss = QSpinBox(self)
        self._sigma_gauss.setFixedWidth(50)
        self._sigma_gauss.setValue(sigma_gauss)
        sigma_gause_layout = QHBoxLayout()
        sigma_gause_layout.addWidget(sigma_gauss_lbl)
        sigma_gause_layout.addWidget(self._sigma_gauss)

        layout.addWidget(self._apply_auto_xphase, 0, 0)
        layout.addLayout(additional_x_phase_layout, 1, 0)
        layout.addLayout(manual_y_phase_layout, 2, 0)
        layout.addLayout(index_frame_to_correlate_layout, 1, 1)
        layout.addLayout(sigma_gause_layout, 2, 1)

        btn_layout = QHBoxLayout()
        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        btn_layout.addWidget(self.apply_btn)
        btn_layout.addWidget(self.new_btn)
        layout.addLayout(btn_layout, 3, 0, 1, 2)

    @property
    def apply_auto_xphase(self) -> bool:
        return self._apply_auto_xphase.isChecked()

    @apply_auto_xphase.setter
    def apply_auto_xphase(self, value: bool) -> None:
        self._apply_auto_xphase.setChecked(value)

    @property
    def additional_x_phase(self) -> int:
        return self._additional_x_phase.value()

    @additional_x_phase.setter
    def additional_x_phase(self, value: int) -> None:
        self._additional_x_phase.setValue(value)

    @property
    def manual_y_phase(self) -> int:
        return self._manual_y_phase.value()

    @manual_y_phase.setter
    def manual_y_phase(self, value: int) -> None:
        self._manual_y_phase.setValue(value)

    @property
    def index_frame_to_correlate(self) -> int:
        return self._index_frame_to_correlate.value()

    @index_frame_to_correlate.setter
    def index_frame_to_correlate(self, value: int) -> None:
        self._index_frame_to_correlate.setValue(value)

    @property
    def sigma_gauss(self) -> int:
        return self._sigma_gauss.value()

    @sigma_gauss.setter
    def sigma_gauss(self, value: int) -> None:
        self._sigma_gauss.setValue(value)

    @classmethod
    def from_config(cls, phase_config: PhaseConfig) -> Self:
        return cls(**phase_config.model_dump())

    def update_from_config(self, phase_config: PhaseConfig) -> None:
        self.apply_auto_xphase = phase_config.apply_auto_xphase
        self.additional_x_phase = phase_config.additional_x_phase
        self.manual_y_phase = phase_config.manual_y_phase
        self.index_frame_to_correlate = phase_config.index_frame_to_correlate
        self.sigma_gauss = phase_config.sigma_gauss

    def to_config(self) -> PhaseConfig:
        return PhaseConfig(
            apply_auto_xphase=self.apply_auto_xphase,
            additional_x_phase=self.additional_x_phase,
            manual_y_phase=self.manual_y_phase,
            index_frame_to_correlate=self.index_frame_to_correlate,
            sigma_gauss=self.sigma_gauss,
        )
