from typing import Self, final

from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from pyfast_ui.config import ImageCorrectionConfig
from pyfast_ui.custom_widgets import LabeledCombobox


@final
class ImageCorrectionGroup(QGroupBox):
    def __init__(self, correction_type: str, align_type: str) -> None:
        super().__init__("Image Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        self._correction_type = LabeledCombobox(
            "Correction type", ["align", "plane", "fixzero"]
        )
        self._align_type = LabeledCombobox(
            "Align type", ["median", "mean", "poly2", "poly3"]
        )

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._correction_type, 0, 0, 1, 2)
        layout.addWidget(self._align_type, 1, 0, 1, 2)

        layout.addWidget(self.apply_btn, 2, 0)
        layout.addWidget(self.new_btn, 2, 1)

    @classmethod
    def from_config(cls, image_correction_config: ImageCorrectionConfig) -> Self:
        return cls(**image_correction_config.model_dump())

    @property
    def correction_type(self) -> str:
        return self._correction_type.value()

    @property
    def align_type(self) -> str:
        return self._align_type.value()
