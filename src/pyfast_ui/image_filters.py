import dataclasses
from typing import Self, final
from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from pyfast_ui.config import ImageFilterConfig
from pyfast_ui.custom_widgets import LabeledCombobox, LabeledSpinBox


@final
class ImageFilterGroup(QGroupBox):
    def __init__(self, filter_type: str, pixel_width: int) -> None:
        super().__init__("Image Filter")
        layout = QGridLayout()
        self.setLayout(layout)

        self._filter_type = LabeledCombobox(
            "Filter type", ["gaussian2d", "median2d", "mean2d"]
        )
        self._pixel_width = LabeledSpinBox("Pixel width", pixel_width)

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._filter_type, 0, 0, 1, 2)
        layout.addWidget(self._pixel_width, 1, 0, 1, 2)

        layout.addWidget(self.apply_btn, 2, 0)
        layout.addWidget(self.new_btn, 2, 1)

    @classmethod
    def from_config(cls, image_filer_config: ImageFilterConfig) -> Self:
        return cls(**dataclasses.asdict(image_filer_config))

    @property
    def filter_type(self) -> str:
        return self._filter_type.value()

    @property
    def pixel_width(self) -> int:
        return self._pixel_width.value()
