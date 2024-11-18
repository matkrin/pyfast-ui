from PySide6.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)


class ImportGroup(QGroupBox):
    def __init__(
        self, image_range: tuple[int, int] | None, apply_auto_xphase: bool
    ) -> None:
        super().__init__("Import")
        layout = QVBoxLayout()
        self.setLayout(layout)

        self._use_image_range = QCheckBox("Use image range", self)
        self._use_image_range.setChecked(False if image_range is None else True)

        self._image_range_start = QSpinBox(self)
        self._image_range_start_lbl = QLabel("Start")
        self._image_range_end = QSpinBox(self)
        self._image_range_end_lbl = QLabel("End")
        start_layout = QHBoxLayout()
        end_layout = QHBoxLayout()

        if image_range is None:
            self._image_range_start.setValue(0)
            self._image_range_end.setValue(0)
        else:
            self._image_range_start.setValue(image_range[0])
            self._image_range_end.setValue(image_range[1])

        self._apply_auto_xphase = QCheckBox("Apply auto X-Phase", self)
        self._apply_auto_xphase.setChecked(apply_auto_xphase)

        self.apply_btn = QPushButton("Import")

        # Add widgets to the start_layout and end_layout
        start_layout.addWidget(self._image_range_start_lbl)
        start_layout.addWidget(self._image_range_start)
        end_layout.addWidget(self._image_range_end_lbl)
        end_layout.addWidget(self._image_range_end)

        # Add Widgets and Layouts to main Layout
        layout.addWidget(self._use_image_range)
        layout.addLayout(start_layout)
        layout.addLayout(end_layout)
        layout.addWidget(self._apply_auto_xphase)
        layout.addWidget(self.apply_btn)

        # Connect Signals
        _ = self._use_image_range.toggled.connect(self._update_image_range_state)

        # Initialize the state of the SpinBoxes based on the checkbox state
        self._update_image_range_state(self._use_image_range.isChecked())

    @property
    def is_image_range(self) -> bool:
        return self._use_image_range.isChecked()

    @property
    def image_range(self) -> tuple[int, int]:
        return (self._image_range_start.value(), self._image_range_end.value())

    @property
    def apply_auto_xphase(self) -> bool:
        return self._apply_auto_xphase.isChecked()

    def _update_image_range_state(self, checked: bool) -> None:
        """Enable or disable the image range spin boxes based on the checkbox state."""
        self._image_range_start.setEnabled(checked)
        self._image_range_end.setEnabled(checked)
        self._image_range_start_lbl.setEnabled(checked)
        self._image_range_end_lbl.setEnabled(checked)
