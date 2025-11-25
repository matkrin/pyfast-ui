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


@dataclass
class BatchDialogResult:
    phase: bool
    fft_filter: bool
    creep: bool
    image_correction: bool
    drift: bool
    image_filter: bool


@final
class BatchDialog(QDialog):
    def __init__(self) -> None:
        super().__init__()

        self._selected_options = BatchDialogResult(
            False,
            False,
            False,
            False,
            False,
            False,
        )

        # Set up the window
        self.setWindowTitle("Select Operations")

        # Create a layout
        layout = QVBoxLayout(self)

        # Create six checkboxes
        self._phase_checkbox = QCheckBox("Phase")
        self._phase_checkbox.setChecked(True)

        self._fft_filter_checkbox = QCheckBox("FFT Filter")
        self._fft_filter_checkbox.setChecked(True)

        self._creep_checkbox = QCheckBox("Creep")
        self._creep_checkbox.setChecked(True)

        self._image_correction_checkbox = QCheckBox("Image Correction")
        self._image_correction_checkbox.setChecked(True)

        self._drift_checkbox = QCheckBox("Drift")
        self._drift_checkbox.setChecked(False)

        self._image_filter_checkbox = QCheckBox("Image Filter")
        self._image_filter_checkbox.setChecked(False)

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
