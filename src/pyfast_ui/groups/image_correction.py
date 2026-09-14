from typing import Self, final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QGridLayout, QGroupBox, QPushButton

from pyfast_ui.config import ImageCorrectionConfig
from pyfast_ui.custom_widgets import LabeledCombobox, LabeledDoubleSpinBox


@final
class ImageCorrectionGroup(QGroupBox):
    area_selection_changed = Signal(str)

    def __init__(
        self,
        correction_type: str,
        align_type: str,
        area_selection: str,
        area_percent: float = 30.0,
    ) -> None:
        super().__init__("Image Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        # "frame level" needs no fit, so neither selection applies to it.
        self._correction_type = LabeledCombobox(
            "Correction type", ["align", "plane", "fixzero", "frame level"]
        )
        self._align_type = LabeledCombobox(
            "Align type",
            [
                "median",
                "median of diff",
                "median of diff (multi-lag)",
                "mean",
                "line",
                "poly2",
                "poly3",
            ],
        )
        self._area_selection = LabeledCombobox(
            "Area selection",
            [
                "whole image",
                "lowest x%",
                "middle x%",
                "highest x%",
                "exclude ROI",
            ],
        )
        self._area_selection.setToolTip(
            "Which pixels the background fit is allowed to use. The height"
            " based choices take a band of the height histogram of each frame,"
            " x % wide, so that a large feature does not drag the background"
            " towards itself. \"exclude ROI\" leaves out the rectangle drawn on"
            " the movie. Applies to the align types and to plane subtraction."
        )

        self._area_percent = LabeledDoubleSpinBox("Area x [%]", area_percent)
        self._area_percent.spinbox.setRange(0.0, 100.0)
        self._area_percent.spinbox.setSingleStep(5.0)
        self._area_percent.setToolTip(
            "How much of the height histogram the band covers. \"lowest\" and"
            " \"highest\" start at their end of the histogram, \"middle\" is"
            " centred on the median. Small values pick the flattest background"
            " but leave few pixels for the fit; large values approach the whole"
            " image.\n\nFor the align corrections the band is taken from each"
            " row separately, since the correction is fitted per row; a row"
            " that keeps fewer than five percent of its pixels takes the"
            " background of the row above. Plane subtraction fits the frame as"
            " a whole and takes the band from the whole frame."
        )

        self.correction_type = correction_type
        self.align_type = align_type
        self.area_selection = area_selection
        self.area_percent = area_percent

        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        layout.addWidget(self._correction_type, 0, 0, 1, 2)
        layout.addWidget(self._align_type, 1, 0, 1, 2)
        layout.addWidget(self._area_selection, 2, 0, 1, 2)
        layout.addWidget(self._area_percent, 3, 0, 1, 2)

        layout.addWidget(self.apply_btn, 4, 0)
        layout.addWidget(self.new_btn, 4, 1)

        _ = self._correction_type.value_changed.connect(self._update_enabled_controls)
        _ = self._area_selection.value_changed.connect(self.area_selection_changed)
        _ = self._area_selection.value_changed.connect(
            lambda _value: self._update_enabled_controls()
        )
        self._update_enabled_controls()

    def _update_enabled_controls(self) -> None:
        """The align type only means something for the align correction, and
        neither selection means anything for "fixzero", which just moves the
        minimum to zero. The percentage in turn only means something for the
        three height bands."""
        correction = self._correction_type.value()
        selects_area = correction in ("align", "plane")
        self._align_type.setEnabled(correction == "align")
        self._area_selection.setEnabled(selects_area)
        self._area_percent.setEnabled(
            selects_area and self._area_selection.value().endswith("x%")
        )

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
        self._update_enabled_controls()

    @property
    def area_percent(self) -> float:
        return self._area_percent.value()

    @area_percent.setter
    def area_percent(self, value: float) -> None:
        self._area_percent.setValue(value)

    @classmethod
    def from_config(cls, image_correction_config: ImageCorrectionConfig) -> Self:
        return cls(**image_correction_config.model_dump())

    def update_from_config(
        self, image_correction_config: ImageCorrectionConfig
    ) -> None:
        self.correction_type = image_correction_config.correction_type
        self.align_type = image_correction_config.align_type
        self.area_selection = image_correction_config.area_selection
        self.area_percent = image_correction_config.area_percent

    def to_config(self) -> ImageCorrectionConfig:
        return ImageCorrectionConfig(
            correction_type=self.correction_type,
            align_type=self.align_type,
            area_selection=self.area_selection,
            area_percent=self.area_percent,
        )
