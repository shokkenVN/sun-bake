import os
from typing import Optional
import numpy as np
import cv2

from PySide6.QtCore import Qt, Signal, QPoint, QRectF
from PySide6.QtGui import (
    QPainter,
    QPixmap,
    QImage,
    QColor,
    QBrush,
    QWheelEvent,
    QMouseEvent,
    QDragEnterEvent,
    QDropEvent,
)
from PySide6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QFileDialog,
)


def _make_checker_brush(size: int = 16) -> QBrush:
    pix = QPixmap(size * 2, size * 2)
    pix.fill(QColor(24, 24, 28))
    p = QPainter(pix)
    c2 = QColor(34, 34, 40)
    p.fillRect(size, 0, size, size, c2)
    p.fillRect(0, size, size, size, c2)
    p.end()
    return QBrush(pix)


class DropZoneWidget(QFrame):
    file_selected = Signal(str)

    def __init__(
        self,
        title: str = "Drop file here",
        subtitle: str = ".model3.json, .moc3, or .psd",
        file_filter: str = "Supported Files (*.json *.moc3 *.psd);;All Files (*.*)",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.file_filter = file_filter
        self.setAcceptDrops(True)
        self.setObjectName("DropZone")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(6)

        top_row = QHBoxLayout()
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 600; font-size: 13px;")

        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setFixedWidth(90)
        self.browse_btn.clicked.connect(self._on_browse_clicked)

        top_row.addWidget(self.title_label)
        top_row.addStretch()
        top_row.addWidget(self.browse_btn)

        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("color: #8b949e; font-size: 11px;")
        self.subtitle_label.setWordWrap(True)

        layout.addLayout(top_row)
        layout.addWidget(self.subtitle_label)

    def set_loaded_file(self, path: str):
        self.title_label.setText(os.path.basename(path))
        self.subtitle_label.setText(path)

    def _on_browse_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open File", "", self.file_filter)
        if file_path:
            self.file_selected.emit(file_path)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if file_path:
                self.file_selected.emit(file_path)
                event.acceptProposedAction()


class InteractiveImageWidget(QWidget):
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._pixmap: Optional[QPixmap] = None
        self._zoom: float = 1.0
        self._pan: QPoint = QPoint(0, 0)
        self._drag_start: Optional[QPoint] = None
        self._user_zoomed: bool = False
        self._checker_brush = _make_checker_brush(16)

        self.setMouseTracking(True)

    def set_image(self, img_bgra: Optional[np.ndarray], keep_view: bool = False):
        if img_bgra is None or img_bgra.size == 0:
            self._pixmap = None
            self.update()
            return

        h, w = img_bgra.shape[:2]
        if img_bgra.shape[2] == 4:
            rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2RGBA)
        else:
            rgba = cv2.cvtColor(img_bgra, cv2.COLOR_BGR2RGBA)

        q_img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        old_size = self._pixmap.size() if self._pixmap else None
        self._pixmap = QPixmap.fromImage(q_img.copy())

        if keep_view and old_size == self._pixmap.size() and self._user_zoomed:
            self.update()
        else:
            self.reset_view()

    def reset_view(self):
        self._user_zoomed = False
        if not self._pixmap or self._pixmap.isNull():
            self.update()
            return
        vw, vh = self.width(), self.height()
        iw, ih = self._pixmap.width(), self._pixmap.height()
        if vw <= 0 or vh <= 0 or iw <= 0 or ih <= 0:
            return

        scale_x = (vw * 0.94) / iw
        scale_y = (vh * 0.94) / ih
        self._zoom = min(scale_x, scale_y, 1.0)
        self._pan = QPoint(
            int((vw - iw * self._zoom) * 0.5),
            int((vh - ih * self._zoom) * 0.5),
        )
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self._user_zoomed and self._drag_start is None:
            self.reset_view()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self._checker_brush)

        if self._pixmap and not self._pixmap.isNull():
            painter.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 2.0)
            iw = self._pixmap.width() * self._zoom
            ih = self._pixmap.height() * self._zoom
            target_rect = QRectF(self._pan.x(), self._pan.y(), iw, ih)
            painter.drawPixmap(target_rect, self._pixmap, QRectF(self._pixmap.rect()))

    def wheelEvent(self, event: QWheelEvent):
        if not self._pixmap:
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        mouse_pos = event.position().toPoint()
        new_zoom = max(0.03, min(16.0, self._zoom * factor))
        self._pan = mouse_pos - (mouse_pos - self._pan) * (new_zoom / self._zoom)
        self._zoom = new_zoom
        self._user_zoomed = True
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag_start = event.pos()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._drag_start is not None and (event.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            self._pan += event.pos() - self._drag_start
            self._drag_start = event.pos()
            self._user_zoomed = True
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._drag_start = None

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        self.reset_view()
