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
            "Filter type", ["gauss", "median", "mean"]
        )
        self._pixel_width = LabeledSpinBox("Pixel width", pixel_width)

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._filter_type, 0, 0, 1, 2)
        layout.addWidget(self._pixel_width, 1, 0, 1, 2)

        layout.addWidget(self.apply_btn, 2, 0)
        layout.addWidget(self.new_btn, 2, 1)

    @property
    def filter_type(self) -> str:
        return self._filter_type.value()

    @filter_type.setter
    def filter_type(self, value: str) -> None:
        self._filter_type.set_value(value)

    @property
    def pixel_width(self) -> int:
        return self._pixel_width.value()

    @pixel_width.setter
    def pixel_width(self, value: int) -> None:
        self._pixel_width.setValue(value)

    @classmethod
    def from_config(cls, image_filer_config: ImageFilterConfig) -> Self:
        return cls(**image_filer_config.model_dump())

    def update_from_config(self, image_filer_config: ImageFilterConfig) -> None:
        self.filter_type = image_filer_config.filter_type
        self.pixel_width = image_filer_config.pixel_width

    def to_config(self) -> ImageFilterConfig:
        return ImageFilterConfig(
            filter_type=self.filter_type,
            pixel_width=self.pixel_width,
        )
