"""Segmentador interativo: cliques do usuário rotulam superpixels e um modelo leve completa a máscara."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from scipy import sparse
from scipy.sparse.linalg import spsolve
from skimage import color
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from polypseg.boundary import boundary_map, pair_strength
from polypseg.features import FeatureConfig, compute_features, feature_group
from polypseg.preprocessing import border_mask, preprocess
from polypseg.radial import soft_prior, star_contour, star_mask
from polypseg.refine import refine_band, smooth_mask
from polypseg.relief import relief_edges
from polypseg.superpixels import SuperpixelConfig, adjacency, compute_superpixels

CLASSIFIERS = ("graph", "hybrid", "svm", "rf")


@dataclass
class SegmenterConfig:
    superpixels: SuperpixelConfig = field(default_factory=SuperpixelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    # graph = random walker no grafo de superpixels; hybrid = SVM como prior + random walker.
    classifier: str = "graph"
    remove_specular: bool = True
    # Superpixels da borda da imagem como fundo: "always", "auto" (só sem cliques de fundo) ou "never".
    border_background: str = "always"
    keep_connected: bool = True
    fill_holes: bool = True
    graph_beta: float = 0.5
    # Cada grupo de descritores pesa o mesmo na distância do grafo, qualquer que seja o número de colunas.
    balance_groups: bool = False
    # Corta arestas cuja fronteira coincide com o mapa de borda (vale no canal B + gradiente).
    boundary_edges: bool = True
    boundary_gamma: float = 0.5
    boundary_valley_weight: float = 0.25
    # Soma ao mapa de borda a descontinuidade do relevo local (sombreamento coaxial + LRM).
    relief_map: bool = False
    # Prior em estrela: contorno radial de maior borda em volta de cada clique de pólipo.
    radial_prior: bool = True
    radial_weight: float = 0.1
    radial_tau: float = 5.0
    radial_max_step: int = 3
    radial_outward: float = 0.8
    radial_balloon: float = 0.1
    # Só a estrela do 1º clique de pólipo; os seguintes costumam cair perto da borda.
    radial_first_only: bool = True
    # Os demais cliques viram restrições do contorno da estrela (por fora dos de pólipo,
    # por dentro dos de fundo).
    radial_constrained: bool = True
    # Peso do prior dividido por (nº de cliques)^decay, para os cliques corrigirem a estrela.
    radial_decay: float = 1.0
    # Random walker por pixel na faixa de ±refine_band px em volta da borda da máscara.
    pixel_refine: bool = True
    refine_band: int = 14
    refine_beta: float = 4.0
    # Suavização final da borda (σ em pixels; 0 desliga), já que os pólipos são arredondados.
    smooth_sigma: float = 0.0
    graph_prior_weight: float = 0.05
    threshold: float = 0.5


@dataclass
class Click:
    row: int
    col: int
    positive: bool


@dataclass
class SegmentationResult:
    mask: np.ndarray
    probability: np.ndarray  # por superpixel, (K,)
    elapsed: float


def graph_propagate(
    weights: sparse.csr_matrix,
    seed_idx: np.ndarray,
    seed_val: np.ndarray,
    prior: np.ndarray | None = None,
    prior_weight: float = 1e-3,
) -> np.ndarray:
    """Random walker no grafo com sementes fixas e um termo opcional de prior por nó.

    Resolve (L_UU + λI) x_U = -L_US x_S + λ prior_U, com L = D - W.
    """
    n = weights.shape[0]
    if prior is None:
        prior = np.full(n, 0.5)
    degree = np.asarray(weights.sum(axis=1)).ravel()
    lap = sparse.diags(degree) - weights
    unseeded = np.setdiff1d(np.arange(n), seed_idx)
    x = prior.astype(np.float64).copy()
    x[seed_idx] = seed_val
    if unseeded.size == 0:
        return x
    lap = lap.tocsr()
    l_uu = lap[unseeded][:, unseeded] + prior_weight * sparse.identity(unseeded.size)
    l_us = lap[unseeded][:, seed_idx]
    rhs = -l_us @ seed_val.astype(np.float64) + prior_weight * prior[unseeded]
    x[unseeded] = np.atleast_1d(spsolve(l_uu.tocsc(), rhs))
    return np.clip(x, 0.0, 1.0)


class InteractiveSegmenter:
    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()
        self.image: np.ndarray | None = None
        self.work: np.ndarray | None = None
        self.labels: np.ndarray | None = None
        self.features: np.ndarray | None = None
        self.feature_names: list[str] = []
        self.weights: sparse.csr_matrix | None = None
        self.border_superpixels: np.ndarray = np.empty(0, dtype=int)
        self.boundary: np.ndarray | None = None
        self._star_cache: dict[tuple, np.ndarray] = {}
        self.luminance: np.ndarray | None = None
        self.tissue: np.ndarray | None = None
        self.clicks: list[Click] = []
        self.setup_time = 0.0

    # ---------------------------------------------------------------- preparação
    def set_image(self, image: np.ndarray) -> None:
        """Calcula superpixels, descritores e grafo. Apaga os cliques anteriores."""
        start = time.perf_counter()
        self.image = image
        work = preprocess(image, self.config.remove_specular)
        self.work = work
        self.labels = compute_superpixels(work, self.config.superpixels)
        raw, self.feature_names = compute_features(work, self.labels, self.config.features)
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        self.features = (raw - mean) / np.where(std > 1e-8, std, 1.0)
        self.boundary = None
        if self._needs_boundary():
            extra = relief_edges(work, ~border_mask(work)) if self.config.relief_map else None
            self.boundary = boundary_map(work, self.config.boundary_valley_weight, extra)
        self._star_cache = {}
        self.luminance = color.rgb2gray(work) if self.config.radial_prior else None
        self.tissue = ~ndi.binary_dilation(border_mask(image), iterations=4) if self.config.radial_prior else None
        self.weights = self._build_weights()
        self.border_superpixels = self._border_superpixels(image)
        self.clicks = []
        self.setup_time = time.perf_counter() - start

    def recompute(self) -> None:
        """Refaz superpixels e descritores com a configuração atual, mantendo os cliques."""
        clicks = list(self.clicks)
        self.set_image(self.image)
        self.clicks = clicks

    def _border_superpixels(self, image: np.ndarray) -> np.ndarray:
        """Superpixels na borda da imagem ou encostados na moldura preta do endoscópio.

        Em bases como CVC-ClinicDB a borda da imagem é quase toda moldura; sem incluir o
        tecido junto à moldura, o fundo automático não representa a mucosa.
        """
        lab = self.labels
        edge = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
        frame = border_mask(image)
        if frame.any():
            edge = np.concatenate([edge, lab[ndi.binary_dilation(frame, iterations=3)]])
        return np.unique(edge)

    def _needs_boundary(self) -> bool:
        return self.config.boundary_edges or self.config.radial_prior

    def star_prior(self) -> np.ndarray | None:
        """Prior em estrela por superpixel (união das estrelas dos cliques de pólipo)."""
        centers = [(c.row, c.col) for c in self.clicks if c.positive]
        if not (self.config.radial_prior and centers):
            return None
        if self.config.radial_first_only:
            centers = centers[:1]
        pixel = np.zeros(self.labels.shape)
        for center in centers:
            others = ()
            if self.config.radial_constrained:
                others = tuple((c.row, c.col, c.positive) for c in self.clicks if (c.row, c.col) != center)
            key = (center, others)
            if key not in self._star_cache:
                angles, radii = star_contour(
                    self.boundary, center, max_step=self.config.radial_max_step, luminance=self.luminance,
                    outward_weight=self.config.radial_outward, balloon=self.config.radial_balloon,
                    constraints=others, valid=self.tissue,
                )
                mask = star_mask(self.labels.shape, center, angles, radii)
                self._star_cache[key] = soft_prior(mask, self.config.radial_tau)
            pixel = np.maximum(pixel, self._star_cache[key])
        n = self.n_superpixels
        return np.bincount(self.labels.ravel(), weights=pixel.ravel(), minlength=n) / np.maximum(
            np.bincount(self.labels.ravel(), minlength=n), 1
        )

    def _column_weights(self) -> np.ndarray:
        if not self.config.balance_groups:
            return np.ones(len(self.feature_names))
        groups = [feature_group(n) for n in self.feature_names]
        sizes = {g: groups.count(g) for g in set(groups)}
        return np.array([1.0 / sizes[g] for g in groups])

    def _build_weights(self) -> sparse.csr_matrix:
        adj = adjacency(self.labels).tocoo()
        diff = self.features[adj.row] - self.features[adj.col]
        dist2 = (diff**2) @ self._column_weights()
        scale = self.config.graph_beta * max(float(np.median(dist2)), 1e-8)
        w = np.exp(-dist2 / scale)
        if self.config.boundary_edges:
            strength = np.asarray(pair_strength(self.labels, self.boundary)[adj.row, adj.col]).ravel()
            ref = self.config.boundary_gamma * max(float(np.median(strength)), 1e-8)
            w = w * np.exp(-strength / ref)
        n = self.labels.max() + 1
        return sparse.coo_matrix((w, (adj.row, adj.col)), shape=(n, n)).tocsr()

    @property
    def n_superpixels(self) -> int:
        return 0 if self.labels is None else int(self.labels.max()) + 1

    # ---------------------------------------------------------------- cliques
    def add_click(self, row: int, col: int, positive: bool) -> int:
        h, w = self.labels.shape
        row = int(np.clip(row, 0, h - 1))
        col = int(np.clip(col, 0, w - 1))
        self.clicks.append(Click(row, col, positive))
        return int(self.labels[row, col])

    def undo(self) -> None:
        if self.clicks:
            self.clicks.pop()

    def clear(self) -> None:
        self.clicks = []

    def seeds(self) -> tuple[np.ndarray, np.ndarray]:
        """Superpixels marcados e seus rótulos; o último clique em um superpixel prevalece."""
        marks: dict[int, int] = {}
        for c in self.clicks:
            marks[int(self.labels[c.row, c.col])] = int(c.positive)
        idx = np.fromiter(marks.keys(), dtype=int, count=len(marks))
        val = np.fromiter(marks.values(), dtype=int, count=len(marks))
        return idx, val

    # ---------------------------------------------------------------- predição
    def predict(self) -> SegmentationResult:
        start = time.perf_counter()
        n = self.n_superpixels
        idx, val = self.seeds()
        if not (val == 1).any():
            empty = np.zeros(self.labels.shape, dtype=bool)
            return SegmentationResult(empty, np.zeros(n), time.perf_counter() - start)

        mode = self.config.border_background
        if mode == "always" or (mode == "auto" and not (val == 0).any()):
            extra = np.setdiff1d(self.border_superpixels, idx)
            idx = np.concatenate([idx, extra])
            val = np.concatenate([val, np.zeros(extra.size, dtype=int)])

        prob = self._superpixel_probability(idx, val)
        prob[idx] = val
        mask = prob[self.labels] > self.config.threshold
        if self.config.pixel_refine:
            clicks = [(c.row, c.col, c.positive) for c in self.clicks]
            mask = refine_band(self.work, mask, clicks, self.config.refine_band, self.config.refine_beta)
        if self.config.smooth_sigma > 0:
            clicks = [(c.row, c.col, c.positive) for c in self.clicks]
            mask = smooth_mask(mask, self.config.smooth_sigma, clicks)
        mask = self._postprocess(mask)
        return SegmentationResult(mask, prob, time.perf_counter() - start)

    def _superpixel_probability(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        mode = self.config.classifier
        has_both = (val == 0).any() and (val == 1).any()
        if mode == "graph" or not has_both:
            prior = self.star_prior()
            if prior is None:
                return graph_propagate(self.weights, idx, val.astype(float))
            # Relativo ao grau médio, para o prior pesar o mesmo com ou sem as arestas de borda.
            degree = float(self.weights.sum()) / self.n_superpixels
            weight = degree * self.config.radial_weight / len(self.clicks) ** self.config.radial_decay
            return graph_propagate(self.weights, idx, val.astype(float), prior=prior, prior_weight=weight)
        x_train, y_train = self.features[idx], val
        if mode == "rf":
            model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=0)
            model.fit(x_train, y_train)
            return model.predict_proba(self.features)[:, 1]
        # Kernel linear: com poucos cliques o RBF só reconhece o próprio superpixel clicado.
        model = SVC(kernel="linear", C=0.1, class_weight="balanced")
        model.fit(x_train, y_train)
        prob = 1.0 / (1.0 + np.exp(-2.0 * model.decision_function(self.features)))
        if mode == "svm":
            return prob
        if mode == "hybrid":
            return graph_propagate(
                self.weights, idx, val.astype(float), prior=prob, prior_weight=self.config.graph_prior_weight
            )
        raise ValueError(f"classificador desconhecido: {mode!r}")

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if self.config.keep_connected and mask.any():
            comps, _ = ndi.label(mask)
            keep = {comps[c.row, c.col] for c in self.clicks if c.positive} - {0}
            if keep:
                mask = np.isin(comps, list(keep))
        if self.config.fill_holes:
            mask = ndi.binary_fill_holes(mask)
        return mask
