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

from polypseg.background import boundary_connectivity
from polypseg.boundary import boundary_map, pair_strength
from polypseg.features import FeatureConfig, compute_features, feature_group
from polypseg.preprocessing import border_mask, preprocess
from polypseg.radial import color_model, region_separation, soft_prior, star_contour, star_mask
from polypseg.refine import refine_band, smooth_mask, snake_band
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
    # "walker" (random walker) ou "cut" (corte mínimo s-t com o mesmo grafo e os mesmos priors).
    graph_solver: str = "walker"
    # Estrela geodésica (Gulshan et al., 2010) no grafo de superpixels: cada superpixel só fica
    # com a probabilidade de pólipo do seu pai na floresta de caminhos mínimos (IFT, Falcão et
    # al., 2004) que parte dos cliques de pólipo, com custo −log w por aresta.
    geodesic_star: bool = False
    # Prior geodésico (Bai e Sapiro, 2007; 0 desliga): d_fundo / (d_pólipo + d_fundo), com as
    # distâncias geodésicas aos cliques de pólipo e às sementes de fundo, peso relativo ao grau e
    # dividido pelo nº de cliques, como o da estrela.
    geodesic_prior: float = 0.2
    # Cada grupo de descritores pesa o mesmo na distância do grafo, qualquer que seja o número de colunas.
    balance_groups: bool = False
    # Corta arestas cuja fronteira coincide com o mapa de borda (vale no canal B + gradiente).
    boundary_edges: bool = True
    boundary_gamma: float = 0.5
    boundary_valley_weight: float = 0.25
    # Soma ao mapa de borda o gradiente do índice de hemoglobina, IHb = 32·log₂(R/G).
    boundary_hemoglobin: bool = False
    # Atenua o mapa de borda sobre vasos (filtro de Frangi); 0 desliga, 1 zera sobre vasos.
    boundary_vessel: float = 0.0
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
    # Escolha do contorno do 1º clique entre candidatos (duas polaridades e dois níveis de
    # expansão) pela separação de cor entre dentro e fora; False mantém um único contorno.
    radial_select: bool = False
    radial_select_balloons: tuple[float, ...] = (0.1, 0.25)
    radial_select_polarities: tuple[bool, ...] = (True, False)
    # O contorno de pólipo escuro só vence se separar a cor (1 + margem) vezes melhor:
    # com a luz junto da câmera, a cúpula clara é o caso comum.
    radial_dark_margin: float = 0.0
    # Peso do termo de região na DP da estrela do 1º clique (0 desliga): P(pólipo) por um modelo
    # de cor (GMM em Lab) ajustado ao disco do clique (pólipo) e aos superpixels de borda (fundo).
    # Só vale enquanto não há outros cliques; depois eles corrigem a estrela.
    radial_color_model: float = 0.0
    # Peso do prior dividido por (nº de cliques)^decay, para os cliques corrigirem a estrela.
    radial_decay: float = 1.0
    # Random walker por pixel na faixa de ±refine_band px em volta da borda da máscara.
    pixel_refine: bool = True
    refine_band: int = 14
    refine_beta: float = 4.0
    # Prior de fundo por conectividade com a borda (0 desliga): cada superpixel é puxado para
    # fundo com peso força · P(fundo), relativo ao grau médio, como o prior da estrela.
    background_prior: float = 0.0
    # Iterações do contorno ativo geodésico morfológico depois do refino (0 desliga).
    snake_iterations: int = 0
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
    prior_weight: float | np.ndarray = 1e-3,
) -> np.ndarray:
    """Random walker no grafo com sementes fixas e um termo opcional de prior por nó.

    Resolve (L_UU + Λ) x_U = -L_US x_S + Λ prior_U, com L = D - W e Λ = diag(λ); λ pode
    ser um escalar ou um peso por nó.
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
    lam = np.broadcast_to(np.asarray(prior_weight, dtype=np.float64), (n,))[unseeded]
    l_uu = lap[unseeded][:, unseeded] + sparse.diags(lam)
    l_us = lap[unseeded][:, seed_idx]
    rhs = -l_us @ seed_val.astype(np.float64) + lam * prior[unseeded]
    x[unseeded] = np.atleast_1d(spsolve(l_uu.tocsc(), rhs))
    return np.clip(x, 0.0, 1.0)


def graph_cut(
    weights: sparse.csr_matrix,
    seed_idx: np.ndarray,
    seed_val: np.ndarray,
    prior: np.ndarray | None = None,
    prior_weight: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Rótulos 0/1 pelo corte mínimo s-t (Boykov e Jolly, 2001), com a energia do random walker em L1.

    Minimiza Σ λᵢ |xᵢ − priorᵢ| + Σ wᵢⱼ [xᵢ ≠ xⱼ] com x binário e as sementes fixas. As
    capacidades são inteiras de 32 bits, como no max-flow do SciPy: os pesos são escalados para
    somarem 10⁹, e as sementes ganham capacidade maior que qualquer corte.
    """
    from scipy.sparse.csgraph import breadth_first_order, maximum_flow

    n = weights.shape[0]
    src, snk = n, n + 1
    if prior is None:
        prior = np.full(n, 0.5)
    lam = np.broadcast_to(np.asarray(prior_weight, dtype=np.float64), (n,))
    to_src = lam * prior  # custo de rotular fundo: cortado se o nó fica do lado do sorvedouro
    to_snk = lam * (1.0 - prior)
    scale = 1e9 / max(float(weights.sum() + to_src.sum() + to_snk.sum()), 1e-12)
    to_src[seed_idx[seed_val == 1]] = 1.1e9 / scale
    to_snk[seed_idx[seed_val == 0]] = 1.1e9 / scale
    w = weights.tocoo()
    nodes = np.arange(n)
    rows = np.concatenate([w.row, np.full(n, src), nodes])
    cols = np.concatenate([w.col, nodes, np.full(n, snk)])
    cap = np.rint(np.concatenate([w.data, to_src, to_snk]) * scale).astype(np.int64)
    keep = cap > 0
    graph = sparse.coo_matrix((cap[keep], (rows[keep], cols[keep])), shape=(n + 2, n + 2)).tocsr()
    flow = maximum_flow(graph, src, snk).flow
    residual = (graph - flow).tocsr()
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reach = breadth_first_order(residual, src, directed=True, return_predecessors=False)
    x = np.zeros(n)
    x[reach[reach < n]] = 1.0
    return x


def geodesic_forest(weights: sparse.csr_matrix, sources: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Floresta de caminhos mínimos a partir das fontes (IFT com custo aditivo −log w por aresta).

    Devolve a distância de cada nó à fonte mais próxima e o predecessor no caminho (−9999 nas
    fontes e nos nós inalcançáveis).
    """
    from scipy.sparse.csgraph import dijkstra

    cost = weights.tocsr(copy=True)
    cost.data = -np.log(np.clip(cost.data, 1e-300, 1.0)) + 1e-9  # zero explícito não é aresta
    dist, pred, _ = dijkstra(cost, directed=False, indices=sources, min_only=True, return_predecessors=True)
    return dist, pred


def geodesic_star(weights: sparse.csr_matrix, prob: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Impõe a convexidade em estrela geodésica: p(nó) ≤ p(pai) ao longo da floresta.

    Com o limiar, um superpixel só é pólipo se todo o caminho geodésico até o clique também for.
    """
    dist, pred = geodesic_forest(weights, sources)
    out = prob.copy()
    out[~np.isfinite(dist)] = 0.0
    for i in np.argsort(dist):
        if pred[i] >= 0:
            out[i] = min(out[i], out[pred[i]])
    return out


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
        self.background: np.ndarray | None = None
        self.boundary: np.ndarray | None = None
        self._star_cache: dict[tuple, np.ndarray] = {}
        self.luminance: np.ndarray | None = None
        self.lab: np.ndarray | None = None
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
            self.boundary = boundary_map(
                work, self.config.boundary_valley_weight, extra,
                self.config.boundary_hemoglobin, self.config.boundary_vessel,
            )
        self._star_cache = {}
        self.luminance = color.rgb2gray(work) if self.config.radial_prior else None
        needs_lab = self.config.radial_select or self.config.radial_color_model > 0
        self.lab = color.rgb2lab(work) if self.config.radial_prior and needs_lab else None
        self.tissue = ~ndi.binary_dilation(border_mask(image), iterations=4) if self.config.radial_prior else None
        self.weights = self._build_weights()
        self.border_superpixels = self._border_superpixels(image)
        self.background = None
        if self.config.background_prior > 0:
            self.background = boundary_connectivity(work, self.labels, self.border_superpixels)
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

    def _star_region(self, center: tuple[int, int], others: tuple) -> np.ndarray:
        """Máscara da estrela; com ``radial_select``, o candidato de maior separação de cor."""
        cfg = self.config
        options = [(True, cfg.radial_balloon)]
        if cfg.radial_select:
            options = [(bright, b) for bright in cfg.radial_select_polarities for b in cfg.radial_select_balloons]
        region = self._color_model(center) if cfg.radial_color_model > 0 and not others else None
        best, best_score = None, -np.inf
        for bright, balloon in options:
            angles, radii = star_contour(
                self.boundary, center, max_step=cfg.radial_max_step, luminance=self.luminance,
                outward_weight=cfg.radial_outward, balloon=balloon, constraints=others,
                valid=self.tissue, bright_dome=bright, region=region, region_weight=cfg.radial_color_model,
            )
            mask = star_mask(self.labels.shape, center, angles, radii)
            if len(options) == 1:
                return mask
            score = region_separation(self.lab, mask, self.tissue)
            if not bright:
                score /= 1.0 + cfg.radial_dark_margin
            if score > best_score:
                best, best_score = mask, score
        return best

    def _color_model(self, center: tuple[int, int]) -> np.ndarray | None:
        """P(pólipo) por pixel para o termo de região da estrela.

        Amostra de pólipo: disco do clique com o menor raio da estrela sem termo de região
        (ao menos 8 px). Amostra de fundo: superpixels de borda, como na regra de fundo.
        """
        cfg = self.config
        _, radii = star_contour(
            self.boundary, center, max_step=cfg.radial_max_step, luminance=self.luminance,
            outward_weight=cfg.radial_outward, balloon=cfg.radial_balloon, valid=self.tissue,
        )
        rr, cc = np.ogrid[: self.labels.shape[0], : self.labels.shape[1]]
        fg = ((rr - center[0]) ** 2 + (cc - center[1]) ** 2 <= max(radii.min(), 8.0) ** 2) & self.tissue
        bg = np.isin(self.labels, self.border_superpixels) & self.tissue & ~fg
        if fg.sum() < 50 or bg.sum() < 50:
            return None
        return color_model(self.lab, fg, bg)

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
                self._star_cache[key] = soft_prior(self._star_region(center, others), self.config.radial_tau)
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
        if self.config.geodesic_star:
            prob = geodesic_star(self.weights, prob, idx[val == 1])
        mask = prob[self.labels] > self.config.threshold
        if self.config.pixel_refine:
            clicks = [(c.row, c.col, c.positive) for c in self.clicks]
            mask = refine_band(self.work, mask, clicks, self.config.refine_band, self.config.refine_beta)
        if self.config.snake_iterations > 0:
            clicks = [(c.row, c.col, c.positive) for c in self.clicks]
            mask = snake_band(self.work, mask, clicks, self.config.snake_iterations)
        if self.config.smooth_sigma > 0:
            clicks = [(c.row, c.col, c.positive) for c in self.clicks]
            mask = smooth_mask(mask, self.config.smooth_sigma, clicks)
        mask = self._postprocess(mask)
        return SegmentationResult(mask, prob, time.perf_counter() - start)

    def _superpixel_probability(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
        mode = self.config.classifier
        has_both = (val == 0).any() and (val == 1).any()
        if mode == "graph" or not has_both:
            star = self.star_prior()
            geo = self._geodesic_prior(idx, val) if self.config.geodesic_prior > 0 else None
            solve = graph_cut if self.config.graph_solver == "cut" else graph_propagate
            if star is None and self.background is None and geo is None:
                return solve(self.weights, idx, val.astype(float))
            # Pesos relativos ao grau médio, para valerem o mesmo com ou sem as arestas de borda.
            # Os priors somam-se: λ·p = λ_e·estrela + λ_f·0 + λ_g·geodésico.
            degree = float(self.weights.sum()) / self.n_superpixels
            decay = len(self.clicks) ** self.config.radial_decay
            lam_star = degree * self.config.radial_weight / decay if star is not None else 0.0
            lam_bg = 0.0
            if self.background is not None:
                lam_bg = degree * self.config.background_prior * self.background / decay
            lam_geo = degree * self.config.geodesic_prior / decay if geo is not None else 0.0
            lam = lam_star + lam_bg + lam_geo
            pull = (0.0 if star is None else lam_star * star) + (0.0 if geo is None else lam_geo * geo)
            target = np.broadcast_to(pull, (self.n_superpixels,)) / np.maximum(lam, 1e-12)
            return solve(self.weights, idx, val.astype(float), prior=target, prior_weight=lam + 1e-12)
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

    def _geodesic_prior(self, idx: np.ndarray, val: np.ndarray) -> np.ndarray | None:
        """d_fundo / (d_pólipo + d_fundo) por superpixel, pelas distâncias geodésicas às sementes."""
        if not (val == 0).any():
            return None
        d_fg, _ = geodesic_forest(self.weights, idx[val == 1])
        d_bg, _ = geodesic_forest(self.weights, idx[val == 0])
        d_fg = np.minimum(d_fg, 1e6)
        d_bg = np.minimum(d_bg, 1e6)
        return d_bg / np.maximum(d_fg + d_bg, 1e-12)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if self.config.keep_connected and mask.any():
            comps, _ = ndi.label(mask)
            keep = {comps[c.row, c.col] for c in self.clicks if c.positive} - {0}
            if keep:
                mask = np.isin(comps, list(keep))
        if self.config.fill_holes:
            mask = ndi.binary_fill_holes(mask)
        return mask
