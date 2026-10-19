"""Segmentação por caixas: o usuário desenha uma caixa em cada pólipo e cada caixa é segmentada à parte.

A caixa é uma restrição forte: fora dela não há pólipo. Cada caixa vira um recorte com uma
folga em volta; superpixels, descritores e grafo são calculados só no recorte, com o
orçamento inteiro de superpixels, e os superpixels da folga entram como fundo. A máscara
final é a união das máscaras das caixas, porque a saída é binária (pólipo ou não).

Caixas que se cruzam: um clique de fundo vale para todas as caixas que o contêm; um clique
de pólipo vai só para a caixa de centro mais próximo, medido em unidades do tamanho de cada
caixa, para não semear um pólipo dentro da caixa do vizinho.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from polypseg.segmenter import Click, InteractiveSegmenter, SegmentationResult, SegmenterConfig

Box = tuple[int, int, int, int]  # (linha0, coluna0, linha1, coluna1), fim exclusivo


@dataclass
class BoxRegion:
    box: Box  # caixa do usuário, em coordenadas da imagem
    crop: Box  # recorte com folga
    segmenter: InteractiveSegmenter

    def contains(self, row: int, col: int) -> bool:
        r0, c0, r1, c1 = self.box
        return r0 <= row < r1 and c0 <= col < c1

    def center_distance(self, row: int, col: int) -> float:
        """Distância ao centro da caixa, com cada eixo dividido pelo lado da caixa."""
        r0, c0, r1, c1 = self.box
        return float(np.hypot((row - (r0 + r1) / 2) / (r1 - r0), (col - (c0 + c1) / 2) / (c1 - c0)))


def boxes_from_mask(mask: np.ndarray) -> list[Box]:
    """Caixa justa de cada componente conexa da máscara (para simular o usuário)."""
    labels, _ = ndi.label(mask)
    return [(s[0].start, s[1].start, s[0].stop, s[1].stop) for s in ndi.find_objects(labels)]


class BoxSegmenter:
    """Um ``InteractiveSegmenter`` por caixa, com a mesma interface usada pela janela e pela simulação.

    Todas as caixas compartilham o mesmo ``config``, então mudar o modelo vale para todas.
    """

    def __init__(self, config: SegmenterConfig | None = None, margin: float = 0.15, min_margin: int = 8):
        self.config = config or SegmenterConfig()
        self.margin = margin
        self.min_margin = min_margin
        self.image: np.ndarray | None = None
        self.regions: list[BoxRegion] = []
        self.clicks: list[Click] = []  # cliques aceitos, em coordenadas da imagem
        self.history: list[list[int]] = []  # caixas que receberam cada clique, para desfazer
        self.labels: np.ndarray | None = None  # superpixels de todos os recortes, para desenhar
        self.setup_time = 0.0

    def set_image(self, image: np.ndarray, boxes: list[Box]) -> None:
        start = time.perf_counter()
        self.image = image
        self.regions = []
        self.clicks = []
        self.history = []
        h, w = image.shape[:2]
        for r0, c0, r1, c1 in boxes:
            r0, r1 = sorted((int(np.clip(r0, 0, h)), int(np.clip(r1, 0, h))))
            c0, c1 = sorted((int(np.clip(c0, 0, w)), int(np.clip(c1, 0, w))))
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            pr = max(self.min_margin, round(self.margin * (r1 - r0)))
            pc = max(self.min_margin, round(self.margin * (c1 - c0)))
            crop = (max(0, r0 - pr), max(0, c0 - pc), min(h, r1 + pr), min(w, c1 + pc))
            roi = np.zeros((crop[2] - crop[0], crop[3] - crop[1]), dtype=bool)
            roi[r0 - crop[0] : r1 - crop[0], c0 - crop[1] : c1 - crop[1]] = True
            seg = InteractiveSegmenter(self.config)
            seg.set_image(image[crop[0] : crop[2], crop[1] : crop[3]], roi=roi)
            self.regions.append(BoxRegion((r0, c0, r1, c1), crop, seg))
        # Rótulos únicos entre recortes (0 = fora de todos); onde os recortes se cruzam, vale o último.
        self.labels = np.zeros((h, w), dtype=np.int32)
        offset = 1
        for reg in self.regions:
            r0, c0, r1, c1 = reg.crop
            self.labels[r0:r1, c0:c1] = reg.segmenter.labels + offset
            offset += reg.segmenter.n_superpixels
        self.setup_time = time.perf_counter() - start

    @property
    def n_superpixels(self) -> int:
        return sum(reg.segmenter.n_superpixels for reg in self.regions)

    @property
    def feature_names(self) -> list[str]:
        return self.regions[0].segmenter.feature_names if self.regions else []

    @property
    def boxes(self) -> list[Box]:
        return [reg.box for reg in self.regions]

    def targets(self, row: int, col: int, positive: bool) -> list[int]:
        """Índices das caixas que recebem o clique (vazio se ele cair fora de todas)."""
        inside = [i for i, reg in enumerate(self.regions) if reg.contains(row, col)]
        if positive and len(inside) > 1:
            inside = [min(inside, key=lambda i: self.regions[i].center_distance(row, col))]
        return inside

    def add_click(self, row: int, col: int, positive: bool) -> list[int]:
        """Encaminha o clique; fora de todas as caixas ele é ignorado (já é fundo)."""
        targets = self.targets(row, col, positive)
        for i in targets:
            r0, c0 = self.regions[i].crop[:2]
            self.regions[i].segmenter.add_click(row - r0, col - c0, positive)
        if targets:
            self.clicks.append(Click(row, col, positive))
            self.history.append(targets)
        return targets

    def undo(self) -> None:
        if self.history:
            self.clicks.pop()
            for i in self.history.pop():
                self.regions[i].segmenter.undo()

    def clear(self) -> None:
        for reg in self.regions:
            reg.segmenter.clear()
        self.clicks = []
        self.history = []

    def predict(self) -> SegmentationResult:
        """União das máscaras das caixas; ``probability`` fica vazio (é por superpixel de cada caixa)."""
        start = time.perf_counter()
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        for reg in self.regions:
            r0, c0, r1, c1 = reg.crop
            mask[r0:r1, c0:c1] |= reg.segmenter.predict().mask
        return SegmentationResult(mask, np.zeros(0), time.perf_counter() - start)
