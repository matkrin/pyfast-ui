from typing import final

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSpinBox,
    QWidget,
)


@final
class LabeledSpinBox(QWidget):
    def __init__(self, label_text: str, spinbox_value: int) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = QLabel(label_text)
        self.spinbox = QSpinBox()
        self.spinbox.setFixedWidth(50)
        self.spinbox.setValue(spinbox_value)
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)

    def value(self) -> int:
        return self.spinbox.value()

    def setValue(self, new_spinbox_value: int) -> None:
        self.spinbox.setValue(new_spinbox_value)


@final
class LabeledSpinBoxes(QWidget):
    value_changed = Signal(tuple)

    def __init__(self, label_text: str, spinbox_values: tuple[int, int]) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)

        self.label = QLabel(label_text)

        self.spinbox_left = QSpinBox()
        self.spinbox_left.setFixedWidth(80)
        self.spinbox_left.setRange(0, 100_000)
        self.spinbox_left.setValue(spinbox_values[0])
        _ = self.spinbox_left.valueChanged.connect(self._emit_value_changed)

        self.spinbox_right = QSpinBox()
        self.spinbox_right.setRange(0, 100_000)
        self.spinbox_right.setFixedWidth(80)
        self.spinbox_right.setValue(spinbox_values[1])
        _ = self.spinbox_right.valueChanged.connect(self._emit_value_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.spinbox_left)
        layout.addWidget(self.spinbox_right)

        self.spinbox_right

    def _emit_value_changed(self) -> None:
        self.value_changed.emit((self.spinbox_left.value(), self.spinbox_right.value()))

    def value(self) -> tuple[int, int]:
        return self.spinbox_left.value(), self.spinbox_right.value()

    def setValue(self, new_spinbox_value: tuple[int, int]) -> None:
        self.spinbox_left.setValue(new_spinbox_value[0])
        self.spinbox_right.setValue(new_spinbox_value[1])


@final
class LabeledDoubleSpinBox(QWidget):
    def __init__(self, label_text: str, spinbox_value: float) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = QLabel(label_text)
        self.spinbox = QDoubleSpinBox()
        self.spinbox.setFixedWidth(50)
        self.spinbox.setValue(spinbox_value)
        layout.addWidget(self.label)
        layout.addWidget(self.spinbox)

    def value(self) -> float:
        return self.spinbox.value()

    def setValue(self, new_spinbox_value: float) -> None:
        self.spinbox.setValue(new_spinbox_value)


@final
class LabeledDoubleSpinBoxes(QWidget):
    value_changed = Signal(tuple)

    def __init__(self, label_text: str, spinbox_values: tuple[float, float]) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = QLabel(label_text)

        self.spinbox_left = QDoubleSpinBox()
        self.spinbox_left.setFixedWidth(80)
        self.spinbox_left.setRange(0.0, 10_000.0)
        self.spinbox_left.setValue(spinbox_values[0])
        _ = self.spinbox_left.valueChanged.connect(self._emit_value_changed)

        self.spinbox_right = QDoubleSpinBox()
        self.spinbox_right.setFixedWidth(80)
        self.spinbox_right.setRange(0.0, 10_000.0)
        self.spinbox_right.setValue(spinbox_values[1])
        _ = self.spinbox_right.valueChanged.connect(self._emit_value_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.spinbox_left)
        layout.addWidget(self.spinbox_right)

    def _emit_value_changed(self) -> None:
        self.value_changed.emit((self.spinbox_left.value(), self.spinbox_right.value()))

    def value(self) -> tuple[float, float]:
        return self.spinbox_left.value(), self.spinbox_right.value()

    def setValue(self, new_spinbox_value: tuple[float, float]) -> None:
        self.spinbox_left.setValue(new_spinbox_value[0])
        self.spinbox_right.setValue(new_spinbox_value[1])


@final
class LabeledCombobox(QWidget):
    value_changed = Signal(str)

    def __init__(self, label_text: str, combobox_values: list[str]) -> None:
        super().__init__()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        self.label = QLabel(label_text)

        self.combobox = QComboBox()
        self.combobox.addItems(combobox_values)
        _ = self.combobox.currentTextChanged.connect(self._emit_value_changed)

        layout.addWidget(self.label)
        layout.addWidget(self.combobox)

    def _emit_value_changed(self) -> None:
        self.value_changed.emit(self.combobox.currentText())

    def value(self) -> str:
        return self.combobox.currentText()

    def set_value(self, value: str) -> None:
        self.combobox.setCurrentText(value)
