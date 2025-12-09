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
            "Align type", ["median", "median of diff", "mean", "poly2", "poly3"]
        )

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._correction_type, 0, 0, 1, 2)
        layout.addWidget(self._align_type, 1, 0, 1, 2)

        layout.addWidget(self.apply_btn, 2, 0)
        layout.addWidget(self.new_btn, 2, 1)

    @property
    def correction_type(self) -> str:
        return self._correction_type.value()

    @correction_type.setter
    def correction_type(self, value: str) -> None:
        self._correction_type.set_value(value)

    @property
    def align_type(self) -> str:
        return self._align_type.value()

    @align_type.setter
    def align_type(self, value: str) -> None:
        self._align_type.set_value(value)

    @classmethod
    def from_config(cls, image_correction_config: ImageCorrectionConfig) -> Self:
        return cls(**image_correction_config.model_dump())

    def update_from_config(
        self, image_correction_config: ImageCorrectionConfig
    ) -> None:
        self.correction_type = image_correction_config.correction_type
        self.align_type = image_correction_config.align_type

    def to_config(self) -> ImageCorrectionConfig:
        return ImageCorrectionConfig(
            correction_type=self.correction_type,
            align_type=self.align_type,
        )
