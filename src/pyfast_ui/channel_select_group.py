from typing import final

from PySide6.QtWidgets import QGroupBox, QHBoxLayout, QPushButton, QWidget

from pyfast_ui.custom_widgets import LabeledCombobox


@final
class ChannelSelectGroup(QWidget):
    def __init__(self) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
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
        self.new_btn = QPushButton("New")

        layout.addWidget(self._channel)
        layout.addWidget(self.new_btn)

    @property
    def channel(self) -> str:
        return self._channel.value()
