from typing import final

from PySide6.QtWidgets import QGroupBox, QPushButton, QVBoxLayout

from pyfast_ui.custom_widgets import LabeledSpinBoxes


@final
class ModifyGroup(QGroupBox):
    def __init__(self, cut_range: tuple[int, int]) -> None:
        super().__init__("Modify")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self.toggle_selection_btn = QPushButton("Toggle crop selection")
        self.crop_btn = QPushButton("Crop")

        self._cut_range = LabeledSpinBoxes("Cut frames", cut_range)
        self._cut_btn = QPushButton("Cut")

        layout.addWidget(self.toggle_selection_btn)
        layout.addWidget(self.crop_btn)
        layout.addWidget(self._cut_range)
        layout.addWidget(self._cut_btn)

    @property
    def cut_range(self) -> tuple[int, int]:
        return self._cut_range.value()

    @cut_range.setter
    def cut_range(self, value: tuple[int, int]) -> None:
        self._cut_range.setValue(value)
