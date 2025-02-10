import dataclasses
from typing import Self
import matplotlib.pyplot as plt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from pyfast_ui.config import ExportConfig


class ExportGroup(QGroupBox):
    def __init__(
        self,
        export_movie: bool,
        export_tiff: bool,
        export_frames: bool,
        frame_export_images: tuple[int, int],
        frame_export_channel: str,
        scaling: tuple[float, float],
        fps_factor: int,
        frame_export_format: str,
        auto_label: bool,
    ) -> None:
        super().__init__("Export")
        layout = QGridLayout()
        self.setLayout(layout)

        self._export_movie = QCheckBox("MP4", self)
        self._export_movie.setChecked(export_movie)

        self._export_tiff = QCheckBox("TIFF", self)
        self._export_tiff.setChecked(export_tiff)

        self._export_frames = QCheckBox("Frames", self)
        self._export_frames.setChecked(export_frames)

        frame_export_images_lbl = QLabel("Frame export images")
        self._frame_export_images_start = QSpinBox(self)
        self._frame_export_images_start.setValue(frame_export_images[0])
        self._frame_export_images_end = QSpinBox(self)
        self._frame_export_images_end.setValue(frame_export_images[1])
        frame_export_images_layout = QHBoxLayout()
        frame_export_images_layout.addWidget(frame_export_images_lbl)
        frame_export_images_layout.addWidget(self._frame_export_images_start)
        frame_export_images_layout.addWidget(self._frame_export_images_end)

        frame_export_channel_lbl = QLabel("Frame export channel")
        self._frame_export_channel = QComboBox()
        self._frame_export_channel.addItems(["ui", "di", "uf", "df", "ub", "db"])
        self._frame_export_channel.setCurrentText(frame_export_channel)
        frame_export_channel_layout = QHBoxLayout()
        frame_export_channel_layout.addWidget(frame_export_channel_lbl)
        frame_export_channel_layout.addWidget(self._frame_export_channel)

        scaling_lbl = QLabel("Scaling")
        self._scaling_start = QDoubleSpinBox(self)
        self._scaling_start.setValue(scaling[0])
        self._scaling_end = QDoubleSpinBox(self)
        self._scaling_end.setValue(scaling[1])
        scaling_layout = QHBoxLayout()
        scaling_layout.addWidget(scaling_lbl)
        scaling_layout.addWidget(self._scaling_start)
        scaling_layout.addWidget(self._scaling_end)

        fps_factor_lbl = QLabel("FPS factor")
        self._fps_factor = QSpinBox(self)
        self._fps_factor.setValue(fps_factor)
        fps_factor_layout = QHBoxLayout()
        fps_factor_layout.addWidget(fps_factor_lbl)
        fps_factor_layout.addWidget(self._fps_factor)

        frame_export_format_lbl = QLabel("Frame export format")
        self._frame_export_format = QComboBox()
        self._frame_export_format.addItems(["png", "jpg", "bmp", "gwy"])
        self._frame_export_format.setCurrentText(frame_export_format)
        frame_export_format_layout = QHBoxLayout()
        frame_export_format_layout.addWidget(frame_export_format_lbl)
        frame_export_format_layout.addWidget(self._frame_export_format)

        self._auto_label = QCheckBox("Auto label", self)
        self._auto_label.setChecked(auto_label)

        self.apply_btn = QPushButton("Export")

        # Add Widgets to Layout
        # Column 0
        layout.addWidget(self._export_movie, 0, 0)
        layout.addWidget(self._export_tiff, 0, 1)
        layout.addWidget(self._export_frames, 0, 2)
        layout.addLayout(scaling_layout, 2, 0, 1, 3)
        layout.addLayout(fps_factor_layout, 3, 0, 1, 3)
        layout.addLayout(frame_export_images_layout, 4, 0, 1, 3)
        layout.addLayout(frame_export_channel_layout, 5, 0, 1, 3)
        layout.addLayout(frame_export_format_layout, 6, 0, 1, 3)
        layout.addWidget(self._auto_label, 7, 0)
        layout.addWidget(self.apply_btn, 8, 0, 1, 3)

    @classmethod
    def from_config(cls, export_config: ExportConfig) -> Self:
        return cls(**dataclasses.asdict(export_config))

    @property
    def export_movie(self) -> bool:
        return self._export_movie.isChecked()

    @property
    def export_tiff(self) -> bool:
        return self._export_tiff.isChecked()

    @property
    def export_frames(self) -> bool:
        return self._export_frames.isChecked()

    @property
    def frame_export_images(self) -> tuple[int, int]:
        return (
            self._frame_export_images_start.value(),
            self._frame_export_images_end.value(),
        )

    @property
    def frame_export_channel(self) -> str:
        return self._frame_export_channel.currentText()

    @property
    def scaling(self) -> tuple[float, float]:
        return (self._scaling_start.value(), self._scaling_end.value())

    @property
    def fps_factor(self) -> int:
        return self._fps_factor.value()

    @property
    def color_map(self) -> str:
        return self._color_map.currentText()

    @property
    def frame_export_format(self) -> str:
        return self._frame_export_format.currentText()

    @property
    def auto_label(self) -> bool:
        return self._auto_label.isChecked()
