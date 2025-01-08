from typing import Literal, TypeAlias
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)


CreepModeType: TypeAlias = None | Literal["sin", "bezier", "root"]


class CreepGroup(QGroupBox):
    def __init__(self, creep_mode: CreepModeType):
        super().__init__("Creep Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        # Create four radio buttons
        self._none = QRadioButton("None", self)
        self._sin = QRadioButton("sin", self)
        self._bezier = QRadioButton("bezier", self)
        self._root = QRadioButton("root", self)

        # Create a QButtonGroup to group the radio buttons
        self.button_group = QButtonGroup(self)

        # Add the radio buttons to the button group
        self.button_group.addButton(self._none)
        self.button_group.addButton(self._sin)
        self.button_group.addButton(self._bezier)
        self.button_group.addButton(self._root)

        match creep_mode:
            case None:
                self._none.setChecked(True)
            case "sin":
                self._sin.setChecked(True)
            case "bezier":
                self._bezier.setChecked(True)
            case "root":
                self._root.setChecked(True)

        self.apply_btn = QPushButton("Apply")

        # Add radio buttons to the layout
        layout.addWidget(self._none, 0, 0)
        layout.addWidget(self._sin, 1, 0)
        layout.addWidget(self._bezier, 0, 1)
        layout.addWidget(self._root, 1, 1)
        layout.addWidget(self.apply_btn, 2, 0, 1, 2)

    @property
    def creep_mode(self) -> CreepModeType:
        selected_button = self.button_group.checkedButton()
        if selected_button:
            button_text = selected_button.text()
            if button_text == "None":
                return None

            return button_text
