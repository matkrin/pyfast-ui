from dataclasses import dataclass
import sys
from typing import final
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QCheckBox,
    QPushButton,
    QHBoxLayout,
)

from pyfast_ui.config import BatchConfig


@dataclass
class BatchDialogResult:
    phase: bool
    fft_filter: bool
    creep: bool
    image_correction: bool
    drift: bool
    image_filter: bool

    def to_config(self) -> BatchConfig:
        return BatchConfig(
            phase=self.phase,
            fft_filter=self.fft_filter,
            creep=self.creep,
            image_correction=self.image_correction,
            drift=self.drift,
            image_filter=self.image_filter,
        )


@final
class BatchDialog(QDialog):
    """The dialog that selects which steps a batch run applies.

    Args:
        config: The ticks to start from. They come from the config file, so a
            selection that has been saved there is offered again on the next
            start instead of the built-in one.
    """

    def __init__(self, config: BatchConfig | None = None) -> None:
        super().__init__()

        if config is None:
            config = BatchConfig()

        self._selected_options = BatchDialogResult(
            config.phase,
            config.fft_filter,
            config.creep,
            config.image_correction,
            config.drift,
            config.image_filter,
        )

        # Set up the window
        self.setWindowTitle("Select Operations")

        # Create a layout
        layout = QVBoxLayout(self)

        # Create six checkboxes
        self._phase_checkbox = QCheckBox("Phase")
        self._phase_checkbox.setChecked(config.phase)

        self._fft_filter_checkbox = QCheckBox("FFT Filter")
        self._fft_filter_checkbox.setChecked(config.fft_filter)

        self._creep_checkbox = QCheckBox("Creep")
        self._creep_checkbox.setChecked(config.creep)

        self._image_correction_checkbox = QCheckBox("Image Correction")
        self._image_correction_checkbox.setChecked(config.image_correction)

        self._drift_checkbox = QCheckBox("Drift")
        self._drift_checkbox.setChecked(config.drift)

        self._image_filter_checkbox = QCheckBox("Image Filter")
        self._image_filter_checkbox.setChecked(config.image_filter)

        # Add checkboxes to the layout
        layout.addWidget(self._phase_checkbox)
        layout.addWidget(self._fft_filter_checkbox)
        layout.addWidget(self._creep_checkbox)
        layout.addWidget(self._image_correction_checkbox)
        layout.addWidget(self._drift_checkbox)
        layout.addWidget(self._image_filter_checkbox)

        # Create a button box with 'OK' and 'Cancel'
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)

        # Add button box to layout
        layout.addLayout(button_box)

        # Connect button actions
        _ = self.ok_button.clicked.connect(self._on_ok_clicked)
        _ = self.cancel_button.clicked.connect(self._on_cancel_clicked)

    def _on_ok_clicked(self):
        # Collect the checkbox states
        self._selected_options.phase = self._phase_checkbox.isChecked()
        self._selected_options.fft_filter= self._fft_filter_checkbox.isChecked()
        self._selected_options.creep= self._creep_checkbox.isChecked()
        self._selected_options.image_correction= self._image_correction_checkbox.isChecked()
        self._selected_options.drift= self._drift_checkbox.isChecked()
        self._selected_options.image_filter= self._image_filter_checkbox.isChecked()

        # Close the dialog and indicate the user clicked 'OK'
        self.accept()

    def _on_cancel_clicked(self):
        # Close the dialog without saving selections
        self.reject()

    def get_selected_options(self) -> BatchDialogResult:
        return self._selected_options
