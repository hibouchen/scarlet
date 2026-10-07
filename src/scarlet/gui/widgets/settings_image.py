from __future__ import annotations

import math

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets


class SettingsImageView(QtWidgets.QWidget):
    """Small detector-image view with editable center and ROI overlays."""

    center_edited = QtCore.Signal(float, float)
    roi_edited = QtCore.Signal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(360, 280)
        self.setMouseTracking(True)
        self._image: np.ndarray | None = None
        self._qimage: QtGui.QImage | None = None
        self._center: tuple[float, float] | None = None
        self._roi: tuple[int, int, int, int] | None = None
        self._edit_mode = "center"
        self._roi_origin: tuple[float, float] | None = None

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._image = None
            self._qimage = None
            self.update()
            return
        values = np.asarray(image, dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"Settings image must be 2D, got shape {values.shape}")
        self._image = values
        self._qimage = self._to_qimage(values)
        self.update()

    def set_overlays(
        self,
        *,
        center: tuple[float, float] | None,
        roi: tuple[int, int, int, int] | None,
    ) -> None:
        self._center = center
        self._roi = roi
        self.update()

    def set_edit_mode(self, mode: str) -> None:
        if mode not in {"center", "roi"}:
            raise ValueError(mode)
        self._edit_mode = mode
        self.setCursor(
            QtCore.Qt.CursorShape.CrossCursor
            if self._image is not None
            else QtCore.Qt.CursorShape.ArrowCursor
        )

    @property
    def image_shape(self) -> tuple[int, int] | None:
        return None if self._image is None else self._image.shape

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        super().paintEvent(event)
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self.palette().color(QtGui.QPalette.ColorRole.Base))
        if self._qimage is None:
            painter.setPen(self.palette().color(QtGui.QPalette.ColorRole.PlaceholderText))
            painter.drawText(
                self.rect(),
                QtCore.Qt.AlignmentFlag.AlignCenter,
                "Load an empty-beam image",
            )
            return
        image_rect = self._image_rect()
        painter.drawImage(image_rect, self._qimage)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        if self._roi is not None:
            x0, x1, y0, y1 = self._roi
            top_left = self._data_to_widget(float(x0), float(y0))
            bottom_right = self._data_to_widget(float(x1), float(y1))
            painter.setPen(QtGui.QPen(QtGui.QColor("#ffd54f"), 2.0))
            painter.drawRect(QtCore.QRectF(top_left, bottom_right).normalized())
        if self._center is not None:
            point = self._data_to_widget(*self._center)
            painter.setPen(QtGui.QPen(QtGui.QColor("#ef5350"), 2.0))
            painter.drawLine(
                QtCore.QPointF(image_rect.left(), point.y()),
                QtCore.QPointF(image_rect.right(), point.y()),
            )
            painter.drawLine(
                QtCore.QPointF(point.x(), image_rect.top()),
                QtCore.QPointF(point.x(), image_rect.bottom()),
            )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        point = self._widget_to_data(event.position())
        if point is None or event.button() != QtCore.Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        if self._edit_mode == "center":
            self._center = point
            self.center_edited.emit(*point)
            self.update()
        else:
            self._roi_origin = point

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._edit_mode != "roi" or self._roi_origin is None:
            return super().mouseMoveEvent(event)
        point = self._widget_to_data(event.position())
        if point is not None:
            self._roi = self._normalized_roi(self._roi_origin, point)
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._edit_mode != "roi"
            or self._roi_origin is None
            or event.button() != QtCore.Qt.MouseButton.LeftButton
        ):
            return super().mouseReleaseEvent(event)
        point = self._widget_to_data(event.position())
        origin = self._roi_origin
        self._roi_origin = None
        if point is None:
            return
        roi = self._normalized_roi(origin, point)
        if roi[0] < roi[1] and roi[2] < roi[3]:
            self._roi = roi
            self.roi_edited.emit(roi)
            self.update()

    def _image_rect(self) -> QtCore.QRectF:
        if self._image is None:
            return QtCore.QRectF()
        ny, nx = self._image.shape
        available = QtCore.QRectF(self.rect()).adjusted(8.0, 8.0, -8.0, -8.0)
        scale = min(available.width() / nx, available.height() / ny)
        width = nx * scale
        height = ny * scale
        return QtCore.QRectF(
            available.center().x() - width / 2.0,
            available.center().y() - height / 2.0,
            width,
            height,
        )

    def _data_to_widget(self, x: float, y: float) -> QtCore.QPointF:
        if self._image is None:
            return QtCore.QPointF()
        ny, nx = self._image.shape
        rect = self._image_rect()
        return QtCore.QPointF(
            rect.left() + x * rect.width() / nx,
            rect.top() + y * rect.height() / ny,
        )

    def _widget_to_data(self, point: QtCore.QPointF) -> tuple[float, float] | None:
        if self._image is None:
            return None
        rect = self._image_rect()
        if not rect.contains(point):
            return None
        ny, nx = self._image.shape
        x = min(float(nx), max(0.0, (point.x() - rect.left()) * nx / rect.width()))
        y = min(float(ny), max(0.0, (point.y() - rect.top()) * ny / rect.height()))
        return x, y

    def _normalized_roi(
        self,
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> tuple[int, int, int, int]:
        if self._image is None:
            return 0, 0, 0, 0
        ny, nx = self._image.shape
        x0 = max(0, min(nx, math.floor(min(first[0], second[0]))))
        x1 = max(0, min(nx, math.ceil(max(first[0], second[0]))))
        y0 = max(0, min(ny, math.floor(min(first[1], second[1]))))
        y1 = max(0, min(ny, math.ceil(max(first[1], second[1]))))
        return x0, x1, y0, y1

    @staticmethod
    def _to_qimage(values: np.ndarray) -> QtGui.QImage:
        finite = np.isfinite(values)
        if not finite.any():
            display = np.zeros(values.shape, dtype=np.uint8)
        else:
            minimum = float(np.nanmin(values[finite]))
            transformed = np.log1p(np.maximum(values - minimum, 0.0))
            valid = transformed[finite]
            low, high = np.percentile(valid, (1.0, 99.5))
            if not math.isfinite(high) or high <= low:
                high = low + 1.0
            display = np.clip((transformed - low) / (high - low), 0.0, 1.0)
            display = np.where(finite, display, 0.0)
            display = np.ascontiguousarray(np.round(display * 255.0), dtype=np.uint8)
        height, width = display.shape
        return QtGui.QImage(
            display.data,
            width,
            height,
            width,
            QtGui.QImage.Format.Format_Grayscale8,
        ).copy()
