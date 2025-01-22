from typing import final
from PySide6.QtWidgets import QGroupBox, QPushButton, QVBoxLayout
from pyfast_ui.custom_widgets import LabeledCombobox, LabeledSpinBoxes


@final
class ModifyGroup(QGroupBox):
    def __init__(self) -> None:
        super().__init__("Modify")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self._channel = LabeledCombobox(
            "Channel",
            [
                "udi",
                "udf",
                "udb",
                "uf",
                "ub",
                "df",
                "db",
                "ui",
                "di",
            ],
        )
        self._slice = LabeledSpinBoxes("Slice frames", (0, 0))
        self.new_btn = QPushButton("New")

        layout.addWidget(self._channel)
        layout.addWidget(self._slice)
        layout.addWidget(self.new_btn)

    @property
    def channel(self) -> str:
        return self._channel.value()

    @property
    def slice(self) -> tuple[int, int]:
        return self._slice.value()
