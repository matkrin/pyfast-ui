from typing import final
from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from pyfast_ui.custom_widgets import LabeledCombobox, LabeledSpinBox


@final
class ImageFilterGroup(QGroupBox):
    def __init__(self, filter_type: str, pixel_width: int) -> None:
        super().__init__("Image Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        self._filter_type = LabeledCombobox(
            "Filter type", ["gaussian2d", "median2d", "mean2d"]
        )
        self._pixel_width = LabeledSpinBox("Pixel width", pixel_width)

        self.apply_btn = QPushButton("Apply")

        layout.addWidget(self._filter_type)
        layout.addWidget(self._pixel_width)
        layout.addWidget(self.apply_btn)

    @property
    def filter_type(self) -> str:
        return self._filter_type.value()

    @property
    def pixel_width(self) -> int:
        return self._pixel_width.value()
