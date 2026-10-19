# TODO: técnicas clássicas da etapa 3

Proposta completa, com fundamento e referências: documento "Técnicas clássicas para a
próxima etapa da segmentação assistida de pólipos" (T1 a T11).

## Já testadas (branch `tecnicas-classicas`)

Avaliação rápida nas duas bases de desenvolvimento: 200 imagens do Kvasir-SEG
(`data/dev.txt`) e 120 do CVC-ClinicDB do treino do PraNet
(`python scripts/download_data.py --only cvc-dev`). Critério de corte: NoC@85 melhor em
pelo menos 0,2 numa base sem piorar a outra em mais de 0,1. **Nenhuma passou**; todas
ficaram como opções desligadas. Não houve rodada no teste.

| Técnica | Opção | Kvasir NoC@85 | CVC NoC@85 | Kvasir Dice@1 | CVC Dice@1 |
| --- | --- | --- | --- | --- | --- |
| Padrão atual | — | 4,92 | 7,02 | 0,685 | 0,546 |
| T3 Fundo por conectividade com a borda | `background_prior=0.15` | 4,97 | 7,02 | 0,681 | 0,542 |
| T1 Estrela escolhida pela separação de cor | `radial_select=True, radial_dark_margin=2` | 4,86 | 6,96 | 0,696 | 0,557 |
| T1, margem 1 | `radial_select=True, radial_dark_margin=1` | 4,89 | 6,92 | 0,680 | 0,562 |
| T5 Índice de hemoglobina no mapa de borda | `boundary_hemoglobin=True` | 4,91 | 7,15 | 0,682 | 0,549 |
| T5 Índice de hemoglobina como descritor | `FeatureConfig(hemoglobin=True)` | 4,93 | 6,98 | 0,685 | 0,546 |
| T7 Supressão de vasos (Frangi) | `boundary_vessel=1.0` | 4,89 | 7,08 | 0,686 | 0,543 |
| T8 Contorno ativo morfológico após o refino | `snake_iterations=10` | 4,99 | 7,22 | 0,685 | 0,548 |

Segunda rodada (Python 3.14, bibliotecas atuais; o padrão deu 4,96 e 7,03 em vez de 4,92 e
7,02, então as linhas abaixo comparam com a primeira linha desta tabela):

| Técnica | Opção | Kvasir NoC@85 | CVC NoC@85 | Kvasir Dice@1 | CVC Dice@1 |
| --- | --- | --- | --- | --- | --- |
| Padrão atual | — | 4,96 | 7,03 | 0,689 | 0,546 |
| T3 Fundo como limite da estrela (só penaliza, peso 10) | removida | 4,96 | 7,02 | 0,688 | 0,555 |
| T1 Termo de região GMM na DP, em todos os cliques | removida | 4,92 | 7,28 | 0,721 | 0,567 |
| T1 Termo de região GMM só no 1º clique | `radial_color_model=10` | 4,92 | 7,05 | 0,721 | 0,567 |
| T1 GMM no 1º clique + separação de cor | `radial_color_model=10, radial_select=True, radial_dark_margin=2` | 4,88 | 7,02 | 0,722 | 0,569 |
| T10 Corte mínimo s-t no lugar do random walker | `graph_solver="cut"` | 5,09 | 7,08 | 0,662 | 0,542 |
| T10 Corte mínimo, prior da estrela 0,03 | `graph_solver="cut", radial_weight=0.03` | 5,37 | 7,03 | 0,593 | 0,537 |
| T10 Corte mínimo, prior da estrela 0,3 | `graph_solver="cut", radial_weight=0.3` | 4,96 | 7,09 | 0,698 | 0,552 |
| T10 Corte mínimo, prior da estrela 1,0 | `graph_solver="cut", radial_weight=1.0` | 5,11 | 7,17 | 0,705 | 0,546 |
| T10 Pesos ao quadrado (família do power watershed) | removida | 5,17 | 7,20 | 0,702 | 0,545 |
| T10 Pesos na 4ª potência | removida | 5,46 | 7,41 | 0,704 | 0,545 |
| T10 Pesos na 8ª potência | removida | 5,54 | 7,41 | 0,704 | 0,545 |

Observações:

- [x] T3: o mapa de fundo separa bem pólipo de mucosa (AUC 0,84 no CVC), mas não muda a
      segmentação, porque a estrela e a regra de borda já cobrem esse papel. Como limite de
      tamanho da estrela (termo de região que só penaliza englobar fundo, pesos 3 a 30), a
      estrela do CVC melhora pouco (IoU 0,466 → 0,480, >2× de 31% para 26%) e o NoC@85 não
      muda. Usado como termo de região completo (também premia), a estrela cresce demais,
      porque quase toda a mucosa longe da borda tem P(fundo) baixo. Código removido.
- [x] T1: um termo de região na DP da estrela (Σ (0,5 − p) ao longo do raio, como na energia
      de Chan-Vese), com p de duas GMMs em Lab por imagem (disco do clique contra superpixels
      de borda), dá o maior Dice@1 até agora nas duas bases (+0,03 e +0,02), mas o NoC@85
      não muda. Aplicado em todos os cliques, piora o CVC (o modelo de cor briga com as
      correções); por isso só vale no 1º. Fica desligado (`radial_color_model`).
- [ ] No CVC, 32% das estrelas do 1º clique ainda saem com mais que o dobro da área do
      pólipo; no Kvasir, 15% saem com menos de 30% (10% com a GMM). Mas consertar a estrela
      só melhora o Dice@1: o NoC@85 depende dos cliques seguintes (ver diagnóstico abaixo).

### Diagnóstico com o método atual

| Base | Cenário | Dice@1 | Dice@3 | Dice@10 | NoC@85 | NoC@90 | Erro a ≤ 5 px da borda (1 / 3 / 10 cliques) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Kvasir | padrão | 0,689 | 0,870 | 0,939 | 4,96 | 7,02 | 38% / 53% / 74% |
| Kvasir | arestas do oráculo | 0,730 | 0,898 | 0,950 | 3,27 | 5,07 | 49% / 65% / 82% |
| CVC | padrão | 0,546 | 0,709 | 0,888 | 7,03 | 8,43 | 37% / 55% / 80% |
| CVC | arestas do oráculo | 0,597 | 0,848 | 0,904 | 5,04 | 6,98 | 48% / 76% / 93% |

Teto de superpixels (IoU pela maioria do gabarito, n = 200 / 400 / 800): Kvasir 0,859 /
0,891 / 0,916; CVC 0,787 / 0,841 / 0,877. Com 400, só 26% das imagens do CVC chegam a
IoU ≥ 0,90 (52% no Kvasir).

- Arestas perfeitas ainda tiram 1,7 (Kvasir) e 2,0 (CVC) do NoC@85: o vazamento continua
  sendo a maior alavanca, mais que a estrela. O T10 mostrou que a alavanca está nos pesos e
  não no algoritmo de propagação (ver abaixo).
- [x] T10: trocar o random walker pelo corte mínimo (mesmo grafo, prior em L1, sementes
      fixas; max-flow do SciPy) empata no Kvasir e piora o CVC, mesmo ajustando o peso da
      estrela. Elevar os pesos a uma potência p, que leva o random walker ao power
      watershed (Couprie et al., 2011), piora sempre; com pesos exponenciais isso equivale a
      dividir `graph_beta` e `boundary_gamma` por p, que já estavam ajustados. O corte fica
      desligado (`graph_solver="cut"`); a potência foi removida. O próximo passo é melhorar
      os pesos das arestas (mapa de borda, descritores), não o resolvedor.
- Depois de 10 cliques, 74% a 80% do erro está a ≤ 5 px da borda: o NoC@90 é limitado pela
  borda fina, e no CVC também pelos superpixels (teto baixo). T9 (várias tesselações) e o
  refino por pixel atacam isso.

## Pendentes

- [ ] T2 Estrela geodésica e transformada imagem-floresta (Falcão et al., 2004; Gulshan et al., 2010)
- [ ] T4 Normalização da iluminação por Retinex multiescala (Jobson et al., 1997)
- [ ] T6 Reflexos especulares como pista de cúpula (hipótese sem apoio publicado)
- [ ] T9 Várias tesselações ou hierarquia OWT-UCM (Arbeláez et al., 2011)
- [x] T10 Power watershed ou corte em grafo no lugar do random walker (Couprie et al., 2011)
- [ ] T11 Prior de convexidade ou descritores de Fourier elípticos (Gorelick et al., 2017; Kuhl e Giardina, 1982)
- [x] Repetir `scripts/diagnose.py` nas duas bases com o método atual, para separar vazamento de erro de borda
- [ ] Fatorial e teste único só com o que passar no corte
