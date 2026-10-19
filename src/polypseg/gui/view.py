"""Visualizador de imagem com zoom, arraste e captura de cliques."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
)

POSITIVE_COLOR = QColor(46, 204, 113)
NEGATIVE_COLOR = QColor(231, 76, 60)
BOX_COLOR = QColor(52, 152, 219)


def numpy_to_qimage(array: np.ndarray) -> QImage:
    """Converte (H, W, 3) uint8 RGB ou (H, W, 4) RGBA em QImage (com cópia)."""
    array = np.ascontiguousarray(array)
    h, w = array.shape[:2]
    if array.shape[2] == 3:
        fmt, stride = QImage.Format.Format_RGB888, 3 * w
    else:
        fmt, stride = QImage.Format.Format_RGBA8888, 4 * w
    return QImage(array.data, w, h, stride, fmt).copy()


class ImageView(QGraphicsView):
    """Mostra imagem + sobreposição. Esquerdo = pólipo, direito = fundo, roda = zoom, meio = arrastar.

    Com Ctrl: arrastar com o esquerdo desenha uma caixa; o direito remove a caixa sob o cursor.
    """

    clicked = Signal(int, int, bool)  # linha, coluna, positivo
    box_drawn = Signal(int, int, int, int)  # linha0, coluna0, linha1, coluna1 (fim exclusivo)
    box_removed = Signal(int, int)  # linha, coluna do clique

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(30, 30, 30)))
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self._image_item = QGraphicsPixmapItem()
        self._overlay_item = QGraphicsPixmapItem()
        self._image_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._image_item)
        self.scene().addItem(self._overlay_item)
        self._markers: list[QGraphicsEllipseItem] = []
        self._boxes: list[QGraphicsRectItem] = []
        self._box_start: QPointF | None = None
        self._rubber = QGraphicsRectItem()
        self._rubber.setPen(self._box_pen(Qt.PenStyle.DashLine))
        self._rubber.setZValue(9)
        self._rubber.hide()
        self.scene().addItem(self._rubber)
        self._pan_start: QPointF | None = None
        self._shape: tuple[int, int] | None = None
        self.interactive = True

    def set_image(self, rgb: np.ndarray) -> None:
        self._image_item.setPixmap(QPixmap.fromImage(numpy_to_qimage(rgb)))
        self._overlay_item.setPixmap(QPixmap())
        self._shape = rgb.shape[:2]
        self.scene().setSceneRect(self._image_item.boundingRect())
        self.set_markers([])
        self.set_boxes([])
        self.fit()

    def set_overlay(self, rgba: np.ndarray | None) -> None:
        self._overlay_item.setPixmap(QPixmap() if rgba is None else QPixmap.fromImage(numpy_to_qimage(rgba)))

    def set_markers(self, clicks: list[tuple[int, int, bool]]) -> None:
        for item in self._markers:
            self.scene().removeItem(item)
        self._markers = []
        for row, col, positive in clicks:
            color = POSITIVE_COLOR if positive else NEGATIVE_COLOR
            item = QGraphicsEllipseItem(-6, -6, 12, 12)
            item.setPos(col + 0.5, row + 0.5)
            item.setBrush(QBrush(color))
            item.setPen(QPen(QColor(255, 255, 255), 2))
            item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setZValue(10)
            self.scene().addItem(item)
            self._markers.append(item)

    @staticmethod
    def _box_pen(style: Qt.PenStyle = Qt.PenStyle.SolidLine) -> QPen:
        pen = QPen(BOX_COLOR, 2, style)
        pen.setCosmetic(True)  # mesma espessura em qualquer zoom
        return pen

    def set_boxes(self, boxes: list[tuple[int, int, int, int]]) -> None:
        for item in self._boxes:
            self.scene().removeItem(item)
        self._boxes = []
        for r0, c0, r1, c1 in boxes:
            item = QGraphicsRectItem(c0, r0, c1 - c0, r1 - r0)
            item.setPen(self._box_pen())
            item.setZValue(9)
            self.scene().addItem(item)
            self._boxes.append(item)

    def _scene_pixel(self, event) -> tuple[int, int] | None:
        pos = self.mapToScene(event.position().toPoint())
        row, col = int(pos.y()), int(pos.x())
        h, w = self._shape
        return (row, col) if 0 <= row < h and 0 <= col < w else None

    def fit(self) -> None:
        if self._shape is not None:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    # ---------------------------------------------------------------- eventos
    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return
        if self._shape is None or not self.interactive:
            return
        pixel = self._scene_pixel(event)
        if pixel is None:
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.button() == Qt.MouseButton.LeftButton:
                self._box_start = QPointF(pixel[1], pixel[0])
                self._rubber.setRect(QRectF(self._box_start, self._box_start))
                self._rubber.show()
            elif event.button() == Qt.MouseButton.RightButton:
                self.box_removed.emit(*pixel)
            return
        if event.button() in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            self.clicked.emit(*pixel, event.button() == Qt.MouseButton.LeftButton)

    def mouseMoveEvent(self, event):
        if self._pan_start is not None:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
            return
        if self._box_start is not None:
            pos = self.mapToScene(event.position().toPoint())
            self._rubber.setRect(QRectF(self._box_start, pos).normalized())
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_start = None
            self.unsetCursor()
            return
        if self._box_start is not None and event.button() == Qt.MouseButton.LeftButton:
            rect = self._rubber.rect()
            self._box_start = None
            self._rubber.hide()
            h, w = self._shape
            r0, r1 = max(0, int(rect.top())), min(h, int(np.ceil(rect.bottom())))
            c0, c1 = max(0, int(rect.left())), min(w, int(np.ceil(rect.right())))
            if r1 - r0 >= 4 and c1 - c0 >= 4:  # ignora Ctrl+clique sem arrastar
                self.box_drawn.emit(r0, c0, r1, c1)
            return
        super().mouseReleaseEvent(event)
