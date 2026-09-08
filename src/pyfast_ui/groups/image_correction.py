from typing import Self, final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from pyfast_ui.config import ImageCorrectionConfig
from pyfast_ui.custom_widgets import LabeledCombobox


@final
class ImageCorrectionGroup(QGroupBox):
    area_selection_changed = Signal(str)

    def __init__(
        self, correction_type: str, align_type: str, area_selection: str
    ) -> None:
        super().__init__("Image Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        self._correction_type = LabeledCombobox(
            "Correction type", ["align", "plane", "fixzero"]
        )
        self._align_type = LabeledCombobox(
            "Align type",
            ["median", "median of diff", "mean", "line", "poly2", "poly3"],
        )
        self._area_selection = LabeledCombobox(
            "Area selection",
            [
                "whole image",
                "lowest 50%",
                "middle 50%",
                "highest 50%",
                "exclude ROI",
            ],
        )
        self._area_selection.setToolTip(
            "Which pixels the background fit is allowed to use. The height"
            " based choices take a band of the height histogram of each frame,"
            " so that a large feature does not drag the background towards"
            " itself. \"exclude ROI\" leaves out the rectangle drawn on the"
            " movie. Applies to the align types and to plane subtraction."
        )

        self.correction_type = correction_type
        self.align_type = align_type
        self.area_selection = area_selection

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._correction_type, 0, 0, 1, 2)
        layout.addWidget(self._align_type, 1, 0, 1, 2)
        layout.addWidget(self._area_selection, 2, 0, 1, 2)

        layout.addWidget(self.apply_btn, 3, 0)
        layout.addWidget(self.new_btn, 3, 1)

        _ = self._correction_type.value_changed.connect(self._update_enabled_controls)
        _ = self._area_selection.value_changed.connect(self.area_selection_changed)
        self._update_enabled_controls()

    def _update_enabled_controls(self) -> None:
        """The align type only means something for the align correction, and
        neither selection means anything for "fixzero", which just moves the
        minimum to zero."""
        correction = self._correction_type.value()
        self._align_type.setEnabled(correction == "align")
        self._area_selection.setEnabled(correction in ("align", "plane"))

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

    @property
    def area_selection(self) -> str:
        return self._area_selection.value()

    @area_selection.setter
    def area_selection(self, value: str) -> None:
        self._area_selection.set_value(value)

    @classmethod
    def from_config(cls, image_correction_config: ImageCorrectionConfig) -> Self:
        return cls(**image_correction_config.model_dump())

    def update_from_config(
        self, image_correction_config: ImageCorrectionConfig
    ) -> None:
        self.correction_type = image_correction_config.correction_type
        self.align_type = image_correction_config.align_type
        self.area_selection = image_correction_config.area_selection

    def to_config(self) -> ImageCorrectionConfig:
        return ImageCorrectionConfig(
            correction_type=self.correction_type,
            align_type=self.align_type,
            area_selection=self.area_selection,
        )
