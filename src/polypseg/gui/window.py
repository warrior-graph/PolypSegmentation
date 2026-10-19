"""Janela principal da ferramenta de segmentação assistida."""

from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from scipy import ndimage as ndi
from skimage import io

from polypseg.features import FeatureConfig
from polypseg.gui.view import ImageView
from polypseg.metrics import dice, iou
from polypseg.preprocessing import load_image, load_mask
from polypseg.segmenter import CLASSIFIERS, InteractiveSegmenter, SegmenterConfig
from polypseg.superpixels import METHODS, SuperpixelConfig, boundaries

IMAGE_FILTER = "Imagens (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
CLASSIFIER_LABELS = {
    "graph": "Grafo (random walker)",
    "hybrid": "Híbrido (SVM + grafo)",
    "svm": "SVM",
    "rf": "Random Forest",
}
BORDER_LABELS = {"always": "Sempre", "auto": "Só sem cliques de fundo", "never": "Nunca"}


def render_overlay(
    labels: np.ndarray | None,
    mask: np.ndarray | None,
    gt: np.ndarray | None,
    show_boundaries: bool = True,
    show_mask: bool = True,
    show_gt: bool = False,
    opacity: float = 0.45,
) -> np.ndarray:
    """Camada RGBA uint8 com máscara, fronteiras dos superpixels e contorno do gabarito."""
    shape = (labels if labels is not None else mask).shape
    out = np.zeros((*shape, 4), dtype=np.uint8)
    if show_mask and mask is not None and mask.any():
        out[mask] = (46, 204, 113, int(255 * opacity))
        edge = mask & ~ndi.binary_erosion(mask, iterations=2)
        out[edge] = (46, 204, 113, 255)
    if show_boundaries and labels is not None:
        out[boundaries(labels)] = (255, 221, 87, 140)
    if show_gt and gt is not None and gt.any():
        edge = gt & ~ndi.binary_erosion(gt, iterations=2)
        out[edge] = (236, 64, 255, 255)
    return out


class _Signals(QObject):
    done = Signal(int, object)
    failed = Signal(int, str)


class _SetupJob(QRunnable):
    """Calcula superpixels e descritores fora da thread da interface.

    Cada job usa seu próprio segmentador; a janela só o adota se o job ainda for o atual,
    então trocar de imagem rapidamente não mistura estados.
    """

    def __init__(self, config: SegmenterConfig, image: np.ndarray, clicks: list, token: int):
        super().__init__()
        self.config, self.image, self.clicks, self.token = config, image, clicks, token
        self.signals = _Signals()

    def run(self):
        try:
            segmenter = InteractiveSegmenter(self.config)
            segmenter.set_image(self.image)
            segmenter.clicks = list(self.clicks)
            self.signals.done.emit(self.token, segmenter)
        except Exception:  # noqa: BLE001 - erro vai para a interface
            self.signals.failed.emit(self.token, traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PolypSeg · segmentação assistida de pólipos")
        self.resize(1400, 900)
        self.segmenter = InteractiveSegmenter(SegmenterConfig())
        self.pool = QThreadPool.globalInstance()
        self.image: np.ndarray | None = None
        self.image_path: Path | None = None
        self.gt: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.files: list[Path] = []
        self.ready = False
        self._token = 0
        self._image_started = 0.0
        self._last_predict = 0.0

        self.view = ImageView(self)
        self.view.clicked.connect(self.on_click)
        self.setCentralWidget(self.view)
        self._build_actions()
        self._build_panel()
        self.statusBar().showMessage("Abra uma imagem ou uma pasta (Ctrl+O / Ctrl+Shift+O).")

    # ---------------------------------------------------------------- construção
    def _build_actions(self):
        bar = self.addToolBar("Principal")
        bar.setMovable(False)

        def action(text, shortcut, slot):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.triggered.connect(slot)
            bar.addAction(act)
            return act

        action("Abrir imagem", "Ctrl+O", self.open_image_dialog)
        action("Abrir pasta", "Ctrl+Shift+O", self.open_folder_dialog)
        action("Abrir gabarito", "Ctrl+G", self.open_gt_dialog)
        action("Salvar máscara", "Ctrl+S", self.save_mask)
        bar.addSeparator()
        action("Desfazer", "Ctrl+Z", self.undo)
        action("Limpar", "Esc", self.clear)
        action("Ajustar", "F", self.view.fit)
        bar.addSeparator()
        action("◀ Anterior", "PgUp", lambda: self.step(-1))
        action("Próxima ▶", "PgDown", lambda: self.step(1))

    def _build_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        files_box = QGroupBox("Imagens")
        files_layout = QVBoxLayout(files_box)
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(140)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        files_layout.addWidget(self.file_list)
        layout.addWidget(files_box)

        sp_box = QGroupBox("Superpixels")
        sp_form = QFormLayout(sp_box)
        self.method_combo = QComboBox()
        self.method_combo.addItems(METHODS)
        self.n_spin = QSpinBox()
        self.n_spin.setRange(20, 2000)
        self.n_spin.setSingleStep(25)
        self.n_spin.setValue(self.segmenter.config.superpixels.n_segments)
        self.comp_spin = QDoubleSpinBox()
        self.comp_spin.setRange(0.1, 100.0)
        self.comp_spin.setValue(self.segmenter.config.superpixels.compactness)
        sp_form.addRow("Método", self.method_combo)
        sp_form.addRow("Quantidade", self.n_spin)
        sp_form.addRow("Compacidade", self.comp_spin)
        layout.addWidget(sp_box)

        feat_box = QGroupBox("Descritores")
        feat_layout = QVBoxLayout(feat_box)
        self.feat_checks = {}
        for key, text in (("color", "Cor (RGB, Lab, HSV)"), ("lbp", "Textura LBP"), ("glcm", "Textura GLCM (Haralick)"), ("gabor", "Banco de Gabor")):
            box = QCheckBox(text)
            box.setChecked(True)
            feat_layout.addWidget(box)
            self.feat_checks[key] = box
        layout.addWidget(feat_box)

        self.recompute_button = QPushButton("Recalcular superpixels e descritores")
        self.recompute_button.clicked.connect(self.recompute)
        layout.addWidget(self.recompute_button)

        model_box = QGroupBox("Modelo")
        model_form = QFormLayout(model_box)
        self.clf_combo = QComboBox()
        for key in CLASSIFIERS:
            self.clf_combo.addItem(CLASSIFIER_LABELS[key], key)
        self.border_combo = QComboBox()
        for key, text in BORDER_LABELS.items():
            self.border_combo.addItem(text, key)
        self.clf_combo.currentIndexChanged.connect(self._on_model_changed)
        self.border_combo.currentIndexChanged.connect(self._on_model_changed)
        model_form.addRow("Classificador", self.clf_combo)
        model_form.addRow("Borda como fundo", self.border_combo)
        layout.addWidget(model_box)

        vis_box = QGroupBox("Visualização")
        vis_layout = QVBoxLayout(vis_box)
        self.show_boundaries = QCheckBox("Fronteiras dos superpixels")
        self.show_boundaries.setChecked(True)
        self.show_mask = QCheckBox("Máscara")
        self.show_mask.setChecked(True)
        self.show_gt = QCheckBox("Contorno do gabarito")
        self.opacity = QSlider(Qt.Orientation.Horizontal)
        self.opacity.setRange(0, 100)
        self.opacity.setValue(45)
        for w in (self.show_boundaries, self.show_mask, self.show_gt):
            w.toggled.connect(self.refresh_overlay)
            vis_layout.addWidget(w)
        self.opacity.valueChanged.connect(self.refresh_overlay)
        vis_layout.addWidget(QLabel("Opacidade da máscara"))
        vis_layout.addWidget(self.opacity)
        layout.addWidget(vis_box)

        info_box = QGroupBox("Resultado")
        info_layout = QVBoxLayout(info_box)
        self.info_label = QLabel("—")
        self.info_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)
        layout.addWidget(info_box)

        help_label = QLabel(
            "<b>Botão esquerdo</b>: pólipo · <b>direito</b>: fundo<br>"
            "<b>Roda</b>: zoom · <b>botão do meio</b>: arrastar<br>"
            "<b>Ctrl+Z</b>: desfazer · <b>Esc</b>: limpar · <b>PgUp/PgDn</b>: navegar"
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(320)
        dock = QDockWidget("Controles", self)
        dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # ---------------------------------------------------------------- configuração
    def current_config(self) -> SegmenterConfig:
        c = self.feat_checks
        return SegmenterConfig(
            superpixels=SuperpixelConfig(self.method_combo.currentText(), self.n_spin.value(), self.comp_spin.value()),
            features=FeatureConfig(
                color=c["color"].isChecked(), lbp=c["lbp"].isChecked(),
                glcm=c["glcm"].isChecked(), gabor=c["gabor"].isChecked(),
            ),
            classifier=self.clf_combo.currentData(),
            border_background=self.border_combo.currentData(),
        )

    # ---------------------------------------------------------------- arquivos
    def open_image_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Abrir imagem", "", IMAGE_FILTER)
        if path:
            self.files = []
            self.file_list.clear()
            self.load_image(Path(path))

    def open_folder_dialog(self):
        path = QFileDialog.getExistingDirectory(self, "Abrir pasta (ex.: Kvasir-SEG)")
        if path:
            self.open_folder(Path(path))

    def open_folder(self, folder: Path):
        images_dir = folder / "images" if (folder / "images").is_dir() else folder
        self.files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.addItems([p.name for p in self.files])
        self.file_list.blockSignals(False)
        if self.files:
            self.file_list.setCurrentRow(0)
        else:
            QMessageBox.warning(self, "Pasta vazia", f"Nenhuma imagem em {images_dir}")

    def _on_file_selected(self, row: int):
        if 0 <= row < len(self.files):
            self.load_image(self.files[row])

    def step(self, delta: int):
        if self.files:
            row = int(np.clip(self.file_list.currentRow() + delta, 0, len(self.files) - 1))
            self.file_list.setCurrentRow(row)

    @staticmethod
    def find_gt(image_path: Path) -> Path | None:
        """Procura o gabarito em ../masks/ com o mesmo nome-base."""
        masks_dir = image_path.parent.parent / "masks"
        if masks_dir.is_dir():
            for p in masks_dir.iterdir():
                if p.stem == image_path.stem and p.suffix.lower() in IMAGE_EXTS:
                    return p
        return None

    def load_image(self, path: Path):
        try:
            image = load_image(str(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro ao abrir", str(exc))
            return
        self.image, self.image_path, self.mask = image, path, None
        gt_path = self.find_gt(path)
        self.gt = load_mask(str(gt_path)) if gt_path else None
        if self.gt is not None and self.gt.shape != image.shape[:2]:
            self.gt = None
        self.show_gt.setEnabled(self.gt is not None)
        self.view.set_image((image * 255).astype(np.uint8))
        self._image_started = time.perf_counter()
        self.setWindowTitle(f"PolypSeg · {path.name}")
        self._start_setup(keep_clicks=False)

    def open_gt_dialog(self):
        if self.image is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Abrir gabarito", "", IMAGE_FILTER)
        if path:
            gt = load_mask(path)
            if gt.shape != self.image.shape[:2]:
                QMessageBox.warning(self, "Tamanho diferente", "O gabarito não tem o tamanho da imagem.")
                return
            self.gt = gt
            self.show_gt.setEnabled(True)
            self.show_gt.setChecked(True)
            self.update_info()

    def save_mask(self):
        if self.mask is None or self.image_path is None:
            self.statusBar().showMessage("Nada para salvar: faça ao menos um clique de pólipo.")
            return
        default = str(self.image_path.with_name(self.image_path.stem + "_mask.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Salvar máscara", default, "PNG (*.png)")
        if not path:
            return
        io.imsave(path, (self.mask.astype(np.uint8) * 255), check_contrast=False)
        # Metadados para medir custo de anotação no artigo.
        cfg = self.segmenter.config
        meta = {
            "image": str(self.image_path),
            "clicks": [{"row": c.row, "col": c.col, "positive": c.positive} for c in self.segmenter.clicks],
            "n_clicks": len(self.segmenter.clicks),
            "annotation_seconds": round(time.perf_counter() - self._image_started, 2),
            "superpixels": vars(cfg.superpixels),
            "features": {k: getattr(cfg.features, k) for k in ("color", "lbp", "glcm", "gabor")},
            "classifier": cfg.classifier,
            "border_background": cfg.border_background,
        }
        if self.gt is not None:
            meta.update(dice=dice(self.mask, self.gt), iou=iou(self.mask, self.gt))
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        self.statusBar().showMessage(f"Máscara salva em {path}")

    # ---------------------------------------------------------------- processamento
    def _start_setup(self, keep_clicks: bool):
        self.ready = False
        self.view.interactive = False
        self.recompute_button.setEnabled(False)
        self._token += 1
        clicks = list(self.segmenter.clicks) if keep_clicks else []
        self.view.set_overlay(None)
        self.view.set_markers([])
        self.statusBar().showMessage("Calculando superpixels e descritores…")
        job = _SetupJob(self.current_config(), self.image, clicks, self._token)
        job.signals.done.connect(self._on_setup_done)
        job.signals.failed.connect(self._on_setup_failed)
        self.pool.start(job)

    @Slot(int, object)
    def _on_setup_done(self, token: int, segmenter: InteractiveSegmenter):
        if token != self._token:
            return  # resultado de uma imagem ou configuração anterior
        self.segmenter = segmenter
        seconds = segmenter.setup_time
        self.ready = True
        self.view.interactive = True
        self.recompute_button.setEnabled(True)
        self.statusBar().showMessage(
            f"{self.segmenter.n_superpixels} superpixels, {len(self.segmenter.feature_names)} descritores em {seconds:.2f}s"
        )
        self.run_prediction()

    @Slot(int, str)
    def _on_setup_failed(self, token: int, message: str):
        if token != self._token:
            return
        self.recompute_button.setEnabled(True)
        QMessageBox.critical(self, "Erro no processamento", message)

    def recompute(self):
        if self.image is not None:
            self._start_setup(keep_clicks=True)

    def _on_model_changed(self):
        if self.image is None:
            return
        cfg = self.current_config()
        self.segmenter.config.classifier = cfg.classifier
        self.segmenter.config.border_background = cfg.border_background
        if self.ready:
            self.run_prediction()

    @Slot(int, int, bool)
    def on_click(self, row: int, col: int, positive: bool):
        if not self.ready:
            return
        self.segmenter.add_click(row, col, positive)
        self.run_prediction()

    def undo(self):
        if self.ready:
            self.segmenter.undo()
            self.run_prediction()

    def clear(self):
        if self.ready:
            self.segmenter.clear()
            self.run_prediction()

    def run_prediction(self):
        result = self.segmenter.predict()
        self.mask = result.mask if result.mask.any() else None
        self._last_predict = result.elapsed
        self.view.set_markers([(c.row, c.col, c.positive) for c in self.segmenter.clicks])
        self.refresh_overlay()
        self.update_info()

    def refresh_overlay(self):
        if self.image is None or not self.ready:
            return
        self.view.set_overlay(
            render_overlay(
                self.segmenter.labels, self.mask, self.gt,
                self.show_boundaries.isChecked(), self.show_mask.isChecked(),
                self.show_gt.isChecked() and self.gt is not None, self.opacity.value() / 100.0,
            )
        )

    def update_info(self):
        n_pos = sum(c.positive for c in self.segmenter.clicks)
        n_neg = len(self.segmenter.clicks) - n_pos
        lines = [
            f"Cliques: {n_pos} pólipo, {n_neg} fundo",
            f"Superpixels: {self.segmenter.n_superpixels}",
            f"Preparação: {self.segmenter.setup_time:.2f}s · predição: {self._last_predict * 1000:.0f} ms",
        ]
        if self.mask is not None:
            lines.append(f"Área da máscara: {100 * self.mask.mean():.1f}% da imagem")
        if self.gt is not None:
            pred = self.mask if self.mask is not None else np.zeros_like(self.gt)
            lines.append(f"<b>Dice {dice(pred, self.gt):.3f} · IoU {iou(pred, self.gt):.3f}</b> (gabarito)")
        self.info_label.setText("<br>".join(lines))
