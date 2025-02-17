from typing import final

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout

from pyfast_ui.custom_widgets import LabeledSpinBoxes


@final
class ModifyGroup(QGroupBox):
    """Group for movie modification functionality.

    Args:
        cut_range: Initial values (start, end) for the cut range
    """

    def __init__(self, cut_range: tuple[int, int]) -> None:
        super().__init__("Modify")
        layout = QVBoxLayout()
        self.setLayout(layout)

        crop_layout = QHBoxLayout()
        self.toggle_selection_btn = QPushButton("Toggle crop selection")
        self.crop_btn = QPushButton("Crop")
        crop_layout.addWidget(self.toggle_selection_btn)
        crop_layout.addWidget(self.crop_btn)

        cut_layout = QHBoxLayout()
        self._cut_range = LabeledSpinBoxes("Cut frames", cut_range)
        self.cut_btn = QPushButton("Cut")
        cut_layout.addWidget(self._cut_range)
        cut_layout.addWidget(self.cut_btn)

        layout.addLayout(crop_layout)
        layout.addLayout(cut_layout)

    @property
    def cut_range(self) -> tuple[int, int]:
        return self._cut_range.value()

    @cut_range.setter
    def cut_range(self, value: tuple[int, int]) -> None:
        self._cut_range.setValue(value)
