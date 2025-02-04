from typing import final

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton

from pyfast_ui.custom_widgets import LabeledSpinBoxes


@final
class ModifyGroup(QGroupBox):
    def __init__(self, cut_range: tuple[int, int]) -> None:
        super().__init__("Select Channel")
        layout = QHBoxLayout()
        self.setLayout(layout)

        self._cut_range = LabeledSpinBoxes("Cut", (0, cut_range[1]))
        self.new_btn = QPushButton("New")

        layout.addWidget(self._cut_range)
        layout.addWidget(self.new_btn)

    @property
    def cut_range(self) -> tuple[int, int]:
        return self._cut_range.value()
