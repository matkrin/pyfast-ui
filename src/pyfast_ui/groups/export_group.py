from typing import Self, final

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
)

from pyfast_ui.config import ExportConfig
from pyfast_ui.custom_widgets import (
    LabeledSpinBox,
    LabeledSpinBoxes,
)


@final
class ExportGroup(QGroupBox):
    def __init__(
        self,
        export_movie: bool,
        export_tiff: bool,
        export_frames: bool,
        frame_export_images: tuple[int, int],
        # frame_export_channel: str,
        double_x_pixels_tiff: bool,
        scaling: int,
        smooth_scaling: bool,
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

        self._frame_export_images = LabeledSpinBoxes(
            "Frame export images", frame_export_images
        )

        self._double_x_pixels_tiff = QCheckBox(
            "Double xpixels in interlaced TIFF files", self
        )
        self._double_x_pixels_tiff.setChecked(double_x_pixels_tiff)
        self._double_x_pixels_tiff.setToolTip(
            "Interlaced channels (udi, ui, di) hold twice as many rows as"
            " columns for the same scan area. Doubling the number of x pixels"
            " on export makes the pixels square, so a square scan stays square"
            " in programs that ignore the resolution tags."
        )

        self._scaling = LabeledSpinBox("Scaling (movie and images)", scaling)

        self._smooth_scaling = QCheckBox("Smooth (bilinear) scaling", self)
        self._smooth_scaling.setChecked(smooth_scaling)
        self._smooth_scaling.setToolTip(
            "Off: every measured pixel is drawn as a block, so what you see are"
            " the values that were measured. On: the picture is smoothed"
            " between the pixel centres, which looks better for presentation"
            " but shows values that were never measured. The data written to"
            " the file is the same either way."
        )

        fps_factor_lbl = QLabel("FPS factor")
        self._fps_factor = QSpinBox(self)
        self._fps_factor.setValue(fps_factor)
        fps_factor_layout = QHBoxLayout()
        fps_factor_layout.addWidget(fps_factor_lbl)
        fps_factor_layout.addWidget(self._fps_factor)

        frame_export_format_lbl = QLabel("Frame export format")
        self._frame_export_format = QComboBox()
        self._frame_export_format.addItems(
            [
                "gwy",
                "txt",
                "png",
                "jpg",
                "svg",
                "pdf",
                "eps",
                "webp",
            ]
        )
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
        layout.addWidget(self._double_x_pixels_tiff, 1, 0, 1, 3)
        # layout.addLayout(scaling_layout, 2, 0, 1, 3)
        layout.addWidget(self._scaling, 2, 0, 1, 3)
        layout.addWidget(self._smooth_scaling, 3, 0, 1, 3)
        layout.addLayout(fps_factor_layout, 4, 0, 1, 3)
        layout.addWidget(self._frame_export_images, 5, 0, 1, 3)
        # layout.addLayout(frame_export_channel_layout, 6, 0, 1, 3)
        layout.addLayout(frame_export_format_layout, 6, 0, 1, 3)
        layout.addWidget(self._auto_label, 7, 0)
        layout.addWidget(self.apply_btn, 8, 0, 1, 3)

    @property
    def export_movie(self) -> bool:
        return self._export_movie.isChecked()

    @export_movie.setter
    def export_movie(self, value: bool) -> None:
        self._export_movie.setChecked(value)

    @property
    def export_tiff(self) -> bool:
        return self._export_tiff.isChecked()

    @export_tiff.setter
    def export_tiff(self, value: bool) -> None:
        self._export_tiff.setChecked(value)

    @property
    def export_frames(self) -> bool:
        return self._export_frames.isChecked()

    @export_frames.setter
    def export_frames(self, value: bool) -> None:
        self._export_frames.setChecked(value)

    @property
    def frame_export_images(self) -> tuple[int, int]:
        return self._frame_export_images.value()

    @frame_export_images.setter
    def frame_export_images(self, value: tuple[int, int]) -> None:
        self._frame_export_images.setValue(value)

    # @property
    # def frame_export_channel(self) -> str:
    #     return self._frame_export_channel.currentText()

    # @frame_export_channel.setter
    # def frame_export_channel(self, value: str) -> None:
    #     self._frame_export_channel.setCurrentText(value)

    @property
    def double_x_pixels_tiff(self) -> bool:
        return self._double_x_pixels_tiff.isChecked()

    @double_x_pixels_tiff.setter
    def double_x_pixels_tiff(self, value: bool) -> None:
        self._double_x_pixels_tiff.setChecked(value)

    @property
    def scaling(self) -> int:
        return self._scaling.value()

    @scaling.setter
    def scaling(self, value: int) -> None:
        self._scaling.setValue(value)

    @property
    def smooth_scaling(self) -> bool:
        return self._smooth_scaling.isChecked()

    @smooth_scaling.setter
    def smooth_scaling(self, value: bool) -> None:
        self._smooth_scaling.setChecked(value)

    @property
    def fps_factor(self) -> int:
        return self._fps_factor.value()

    @fps_factor.setter
    def fps_factor(self, value: int) -> None:
        self._fps_factor.setValue(value)

    @property
    def frame_export_format(self) -> str:
        return self._frame_export_format.currentText()

    @frame_export_format.setter
    def frame_export_format(self, value: str) -> None:
        self._frame_export_format.setCurrentText(value)

    @property
    def auto_label(self) -> bool:
        return self._auto_label.isChecked()

    @auto_label.setter
    def auto_label(self, value: bool) -> None:
        self._auto_label.setChecked(value)

    @classmethod
    def from_config(cls, export_config: ExportConfig) -> Self:
        return cls(**export_config.model_dump())

    def update_from_config(self, export_config: ExportConfig) -> None:
        self.export_movie = export_config.export_movie
        self.export_tiff = export_config.export_tiff
        self.export_frames = export_config.export_frames
        self.double_x_pixels_tiff = export_config.double_x_pixels_tiff
        self.scaling = export_config.scaling
        self.smooth_scaling = export_config.smooth_scaling
        self.fps_factor = export_config.fps_factor
        self.auto_label = export_config.auto_label
        self.frame_export_images = export_config.frame_export_images
        # self.frame_export_channel = export_config.frame_export_channel
        self.frame_export_format = export_config.frame_export_format

    def to_config(self) -> ExportConfig:
        return ExportConfig(
            export_movie=self.export_movie,
            export_tiff=self.export_tiff,
            export_frames=self.export_frames,
            double_x_pixels_tiff=self.double_x_pixels_tiff,
            scaling=self.scaling,
            smooth_scaling=self.smooth_scaling,
            fps_factor=self.fps_factor,
            auto_label=self.auto_label,
            frame_export_images=self.frame_export_images,
            # frame_export_channel=self.frame_export_channel,
            frame_export_format=self.frame_export_format,
        )
