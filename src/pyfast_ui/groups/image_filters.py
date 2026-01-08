from typing import Self, final

from PySide6.QtWidgets import (
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
)

from pyfast_ui.config import ImageFilterConfig
from pyfast_ui.custom_widgets import (
    LabeledSpinBox,
    LabeledDoubleSpinBox,
)


@final
class ImageFilterGroup(QGroupBox):
    def __init__(self, filter_type: str, pixel_width: int, gauss_sigma: float) -> None:
        super().__init__("Image Filter")
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        self._filter_type_layout = QHBoxLayout()
        self._filter_type_label = QLabel("Filter type")
        self._filter_type = QComboBox()
        self._filter_type.addItems(["gauss", "median", "mean"])
        self._filter_type_layout.addWidget(self._filter_type_label)
        self._filter_type_layout.addWidget(self._filter_type)

        # For mean and median
        self._pixel_width = LabeledSpinBox("Pixel width", pixel_width)
        # Form gauss
        self._sigma = LabeledDoubleSpinBox("Sigma", gauss_sigma)
        self._sigma.spinbox.setDecimals(2)
        self._sigma.spinbox.setSingleStep(0.1)

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        self.layout.addLayout(self._filter_type_layout, 0, 0, 1, 2)

        if self._filter_type.currentText() == "gauss":
            self.layout.addWidget(self._sigma, 1, 0, 1, 2)
        else:
            self.layout.addWidget(self._pixel_width, 1, 0, 1, 2)

        self.layout.addWidget(self.apply_btn, 2, 0)
        self.layout.addWidget(self.new_btn, 2, 1)

        _ = self._filter_type.currentTextChanged.connect(self._on_combobox_change)

    def _on_combobox_change(self, text: str):
        if text == "gauss":
            self.layout.removeWidget(self._pixel_width)
            self._pixel_width.setParent(None)
            self.layout.addWidget(self._sigma, 1, 0, 1, 2)
        else:
            self.layout.removeWidget(self._sigma)
            self._sigma.setParent(None)
            self.layout.addWidget(self._pixel_width, 1, 0, 1, 2)

    @property
    def filter_type(self) -> str:
        return self._filter_type.currentText()

    @filter_type.setter
    def filter_type(self, value: str) -> None:
        self._filter_type.setCurrentText(value)

    @property
    def pixel_width(self) -> int:
        return self._pixel_width.value()

    @pixel_width.setter
    def pixel_width(self, value: int) -> None:
        self._pixel_width.setValue(value)

    @property
    def gauss_sigma(self) -> float:
        return self._sigma.value()

    @gauss_sigma.setter
    def gauss_sigma(self, value: float) -> None:
        self._sigma.setValue(value)

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
