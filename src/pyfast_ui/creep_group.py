from typing import Literal, TypeAlias
from PySide6.QtWidgets import (
    QButtonGroup,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from pyfast_ui.custom_widgets import LabeledDoubleSpinBox, LabeledSpinBox


CreepModeType: TypeAlias = None | Literal["sin", "bezier", "root"]


class CreepGroup(QGroupBox):
    def __init__(
        self,
        creep_mode: CreepModeType,
        weight_boundry: float,
        creep_num_cols: int,
        known_input: tuple[float, float, float] | None,
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
    ):
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

        # Bezier group
        bezier_group = QGroupBox("Bezier")
        bezier_group_layout = QVBoxLayout()
        bezier_group.setLayout(bezier_group_layout)
        self._weight_boundry = LabeledDoubleSpinBox("Weight_boundry", weight_boundry)
        self._creep_num_cols = LabeledSpinBox("Creep num cols", creep_num_cols)
        bezier_group_layout.addWidget(self._weight_boundry)
        bezier_group_layout.addWidget(self._creep_num_cols)

        # Non-bezier group
        non_bezier_group = QGroupBox("Non-bezier")
        non_bezier_group_layout = QVBoxLayout()
        non_bezier_group.setLayout(non_bezier_group_layout)
        self._initial_guess = LabeledDoubleSpinBox("Initial guess", initial_guess)
        self._guess_ind = LabeledDoubleSpinBox("Guess ind", guess_ind)
        non_bezier_group_layout.addWidget(self._initial_guess)
        non_bezier_group_layout.addWidget(self._guess_ind)

        # Apply button
        self.apply_btn = QPushButton("Apply")
        self.new_btn = QPushButton("New")

        # Add radio buttons to the layout
        layout.addWidget(self._none, 0, 0)
        layout.addWidget(self._sin, 0, 1)
        layout.addWidget(self._bezier, 0, 2)
        layout.addWidget(self._root, 0, 3)
        layout.addWidget(bezier_group, 1, 0, 1, 2)
        layout.addWidget(non_bezier_group, 1, 2, 1, 2)

        layout.addWidget(self.apply_btn, 2, 0, 1, 2)
        layout.addWidget(self.new_btn, 2, 2, 1, 2)

    @property
    def creep_mode(self) -> CreepModeType:
        selected_button = self.button_group.checkedButton()
        if selected_button:
            button_text = selected_button.text()
            if button_text == "None":
                return None

            return button_text

        # creep_mode: CreepModeType,
        # weight_boundry: float,
        # creep_num_cols: int,
        # known_input: tuple[float, float, float] | None,
        # initial_guess: float,
        # guess_ind: float,
        # known_params: float | None,
    @property
    def weight_boundry(self) -> float:
        return self._weight_boundry.value()

    @property
    def creep_num_cols(self) -> int:
        return self._creep_num_cols.value()

    @property
    def initial_guess(self) -> float:
        return self._initial_guess.value()

    @property
    def guess_ind(self) -> float:
        return self._guess_ind.value()


