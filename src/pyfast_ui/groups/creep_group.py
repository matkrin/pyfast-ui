from typing import Self, final

from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
)

from pyfast_ui.config import CreepConfig
from pyfast_ui.custom_widgets import LabeledDoubleSpinBox, LabeledSpinBox


@final
class CreepGroup(QGroupBox):
    def __init__(
        self,
        creep_mode: str,
        weight_boundry: float,
        creep_num_cols: int,
        known_input: tuple[float, float, float] | None,
        initial_guess: float,
        guess_ind: float,
        known_params: float | None,
        align_updown: bool = False,
        use_previous_creep: bool = False,
    ):
        super().__init__("Creep Correction")
        layout = QGridLayout()
        self.setLayout(layout)

        self.known_input = known_input
        self.known_params = known_params

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
        self._weight_boundry.spinbox.setDecimals(2)
        self._weight_boundry.spinbox.setSingleStep(0.1)
        self._creep_num_cols = LabeledSpinBox("Creep num cols", creep_num_cols)
        bezier_group_layout.addWidget(self._weight_boundry)
        bezier_group_layout.addWidget(self._creep_num_cols)

        # Non-bezier group
        non_bezier_group = QGroupBox("Non-bezier")
        non_bezier_group_layout = QVBoxLayout()
        non_bezier_group.setLayout(non_bezier_group_layout)
        self._initial_guess = LabeledDoubleSpinBox("Initial guess", initial_guess)
        self._initial_guess.spinbox.setDecimals(2)
        self._initial_guess.spinbox.setSingleStep(0.1)
        self._guess_ind = LabeledDoubleSpinBox("Guess ind", guess_ind)
        self._guess_ind.spinbox.setDecimals(2)
        self._guess_ind.spinbox.setSingleStep(0.1)
        non_bezier_group_layout.addWidget(self._initial_guess)
        non_bezier_group_layout.addWidget(self._guess_ind)

        # Reuse of an earlier evaluation
        self._use_previous_creep = QCheckBox("Use previous ud evaluation")
        self._use_previous_creep.setChecked(use_previous_creep)
        self._use_previous_creep.setToolTip(
            "Take the creep parameters from the '.creep.txt' file that an"
            " earlier run of the same movie in a channel with both sweeps wrote"
            " next to the .h5 file. A movie of one sweep, ui or di, cannot be"
            " fitted on its own, because the fit compares an up frame with the"
            " following down frame."
        )

        # Alignment of the two sweeps against each other
        self._align_updown = QCheckBox("Align up/down frames")
        self._align_updown.setChecked(align_updown)
        self._align_updown.setToolTip(
            "After the creep correction, shift the up and the down frames onto"
            " each other by interpolation, vertically and horizontally. Removes"
            " what the y-phase cannot reach, it moves in steps of four rows, and"
            " a horizontal offset between the two sweeps, which nothing else"
            " corrects."
        )

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

        layout.addWidget(self._use_previous_creep, 2, 0, 1, 4)
        layout.addWidget(self._align_updown, 3, 0, 1, 4)
        layout.addWidget(self.apply_btn, 4, 0, 1, 2)
        layout.addWidget(self.new_btn, 4, 2, 1, 2)

    @property
    def creep_mode(self) -> str:
        selected_button = self.button_group.checkedButton()
        button_text = selected_button.text()
        if button_text == "None":
            return "None"

        return button_text

    @creep_mode.setter
    def creep_mode(self, value: str) -> None:
        match value:
            case "None" | None:
                self._none.setChecked(True)
            case "sin":
                self._sin.setChecked(True)
            case "bezier":
                self._bezier.setChecked(True)
            case "root":
                self._root.setChecked(True)

    @property
    def weight_boundry(self) -> float:
        return self._weight_boundry.value()

    @weight_boundry.setter
    def weight_boundry(self, value: float) -> None:
        self._weight_boundry.setValue(value)

    @property
    def creep_num_cols(self) -> int:
        return self._creep_num_cols.value()

    @creep_num_cols.setter
    def creep_num_cols(self, value: int) -> None:
        self._creep_num_cols.setValue(value)

    @property
    def initial_guess(self) -> float:
        return self._initial_guess.value()

    @initial_guess.setter
    def initial_guess(self, value: float) -> None:
        self._initial_guess.setValue(value)

    @property
    def use_previous_creep(self) -> bool:
        return self._use_previous_creep.isChecked()

    @use_previous_creep.setter
    def use_previous_creep(self, value: bool) -> None:
        self._use_previous_creep.setChecked(value)

    @property
    def align_updown(self) -> bool:
        return self._align_updown.isChecked()

    @align_updown.setter
    def align_updown(self, value: bool) -> None:
        self._align_updown.setChecked(value)

    @property
    def guess_ind(self) -> float:
        return self._guess_ind.value()

    @guess_ind.setter
    def guess_ind(self, value: float) -> None:
        self._guess_ind.setValue(value)

    @classmethod
    def from_config(cls, creep_config: CreepConfig) -> Self:
        return cls(**creep_config.model_dump())

    def update_from_config(self, creep_config: CreepConfig) -> None:
        self.creep_mode = creep_config.creep_mode
        self.weight_boundry = creep_config.weight_boundry
        self.creep_num_cols = creep_config.creep_num_cols
        self.initial_guess = creep_config.initial_guess
        self.guess_ind = creep_config.guess_ind
        self.known_input = creep_config.known_input
        self.known_params = creep_config.known_params
        self.align_updown = creep_config.align_updown
        self.use_previous_creep = creep_config.use_previous_creep

    def to_config(self) -> CreepConfig:
        return CreepConfig(
            creep_mode=self.creep_mode,
            weight_boundry=self.weight_boundry,
            creep_num_cols=self.creep_num_cols,
            known_input=self.known_input,
            initial_guess=self.initial_guess,
            guess_ind=self.guess_ind,
            known_params=self.known_params,
            align_updown=self.align_updown,
            use_previous_creep=self.use_previous_creep,
        )
