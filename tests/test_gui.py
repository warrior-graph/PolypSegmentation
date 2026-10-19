"""Teste de fumaça da interface em modo offscreen."""

import os
import time

import numpy as np
import pytest
from skimage import io

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets", exc_type=ImportError)

from polypseg.gui.window import MainWindow, render_overlay  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def wait_ready(app, window, timeout=30.0):
    start = time.time()
    while not window.ready and time.time() - start < timeout:
        app.processEvents()
        time.sleep(0.01)
    assert window.ready


def test_render_overlay_layers():
    labels = np.zeros((10, 10), int)
    labels[:, 5:] = 1
    mask = np.zeros((10, 10), bool)
    mask[2:8, 2:8] = True
    out = render_overlay(labels, mask, mask, show_gt=True)
    assert out.shape == (10, 10, 4) and out.dtype == np.uint8
    assert out[..., 3].max() == 255
    custom = render_overlay(labels, mask, None, show_boundaries=False, colors={"mask": (10, 20, 30)})
    assert tuple(custom[4, 4, :3]) == (10, 20, 30)
    thin = render_overlay(labels, None, None, show_mask=False)
    thick = render_overlay(labels, None, None, show_mask=False, boundary_width=3)
    assert (thick[..., 3] > 0).sum() > (thin[..., 3] > 0).sum()


def test_window_folder_click_and_dice(app, synthetic, tmp_path):
    image, gt = synthetic
    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    io.imsave(tmp_path / "images" / "a.png", (image * 255).astype(np.uint8), check_contrast=False)
    io.imsave(tmp_path / "masks" / "a.png", (gt * 255).astype(np.uint8), check_contrast=False)

    window = MainWindow()
    window.open_folder(tmp_path)
    wait_ready(app, window)
    assert window.gt is not None

    window.view.clicked.emit(80, 110, True)
    window.view.clicked.emit(10, 10, False)
    app.processEvents()
    assert window.mask is not None
    assert "Dice" in window.info_label.text()

    window.overlay_colors["boundaries"] = (0, 0, 255)
    window.refresh_overlay()
    window._update_color_buttons()
    assert "rgb(0, 0, 255)" in window.color_buttons["boundaries"].styleSheet()

    window.undo()
    assert len(window.segmenter.clicks) == 1
    window.clear()
    assert window.mask is None
    window.close()


def test_window_boxes(app, synthetic, tmp_path):
    from polypseg.boxes import BoxSegmenter

    image, gt = synthetic
    io.imsave(tmp_path / "a.png", (image * 255).astype(np.uint8), check_contrast=False)
    window = MainWindow()
    window.load_image(tmp_path / "a.png")
    wait_ready(app, window)
    window.view.clicked.emit(80, 110, True)

    window.view.box_drawn.emit(40, 70, 120, 150)  # caixa em volta do pólipo
    wait_ready(app, window)
    assert isinstance(window.segmenter, BoxSegmenter)
    assert len(window.segmenter.clicks) == 1  # o clique dentro da caixa foi mantido
    window.view.clicked.emit(5, 5, False)  # fora da caixa: ignorado
    app.processEvents()
    assert len(window.segmenter.clicks) == 1
    assert window.mask is not None and not window.mask[:30].any()

    window.view.box_removed.emit(80, 110)
    wait_ready(app, window)
    assert not isinstance(window.segmenter, BoxSegmenter)
    window.close()


def test_view_ctrl_drag_draws_box(app):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtTest import QTest

    from polypseg.gui.view import ImageView

    view = ImageView()
    view.resize(300, 300)
    view.show()
    view.set_image(np.zeros((100, 120, 3), np.uint8))
    app.processEvents()
    boxes, removed, clicks = [], [], []
    view.box_drawn.connect(lambda *b: boxes.append(b))
    view.box_removed.connect(lambda *p: removed.append(p))
    view.clicked.connect(lambda *c: clicks.append(c))
    port = view.viewport()
    ctrl = Qt.KeyboardModifier.ControlModifier
    a, b = view.mapFromScene(QPointF(20, 10)), view.mapFromScene(QPointF(80, 60))
    QTest.mousePress(port, Qt.MouseButton.LeftButton, ctrl, a)
    QTest.mouseMove(port, b)
    QTest.mouseRelease(port, Qt.MouseButton.LeftButton, ctrl, b)
    QTest.mouseClick(port, Qt.MouseButton.RightButton, ctrl, b)
    QTest.mouseClick(port, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, b)
    assert len(boxes) == 1
    r0, c0, r1, c1 = boxes[0]
    assert abs(r0 - 10) <= 2 and abs(c0 - 20) <= 2 and abs(r1 - 60) <= 2 and abs(c1 - 80) <= 2
    assert len(removed) == 1 and len(clicks) == 1 and clicks[0][2] is True
    view.close()
