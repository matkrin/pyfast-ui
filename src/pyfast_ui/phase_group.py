from typing import final

from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)


@final
class PhaseGroup(QGroupBox):
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

        self.apply_btn = QPushButton("Apply")

        layout.addWidget(self._apply_auto_xphase, 0, 0)
        layout.addLayout(additional_x_phase_layout, 1, 0)
        layout.addLayout(manual_y_phase_layout, 2, 0)
        layout.addLayout(index_frame_to_correlate_layout, 1, 1)
        layout.addLayout(sigma_gause_layout, 2, 1)
        layout.addWidget(self.apply_btn, 3, 0, 1, 2)

    @property
    def apply_auto_xphase(self) -> bool:
        return self._apply_auto_xphase.isChecked()

    @property
    def additional_x_phase(self) -> int:
        return self._additional_x_phase.value()

    @property
    def manual_y_phase(self) -> int:
        return self._manual_y_phase.value()

    @property
    def index_frame_to_correlate(self) -> int:
        return self._index_frame_to_correlate.value()

    @property
    def sigma_gauss(self) -> int:
        return self._sigma_gauss.value()
