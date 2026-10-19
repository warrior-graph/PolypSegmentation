# PolypSegmentation

Protótipo de **segmentação assistida de pólipos** em imagens de colonoscopia com
superpixels e descritores de cor e textura. O usuário clica em alguns pontos do
pólipo (e, se quiser, do fundo); o sistema rotula os superpixels clicados e
propaga a decisão para o resto da imagem. É a primeira etapa de um trabalho que
depois evolui para a segmentação automática.

![Interface](docs/interface.png)

*3 cliques de pólipo (verde) e 1 de fundo (vermelho): Dice 0,95 contra o gabarito (contorno magenta). Imagem do [Kvasir-SEG](https://arxiv.org/abs/1911.07069) (Jha et al., 2020).*

## Por que PySide6 (Qt 6)

- É o binding **oficial** do Qt para Python, mantido pela Qt Company, com licença LGPL.
- É nativo e roda offline em Linux, Windows e macOS, sem servidor web nem GPU.
- O `QGraphicsView` dá zoom, arraste e sobreposições sobre imagens grandes com
  baixa latência, que é o que importa numa ferramenta de anotação por cliques.
- O estilo Fusion do Qt 6 segue automaticamente o tema claro ou escuro do sistema.

Alternativas consideradas: napari (bom para microscopia, pesado para um fluxo
simples de cliques), Gradio e NiceGUI (web, mas com latência maior a cada clique
e menos controle de desenho).

## Instalação

Requer Python 3.11 ou mais novo. Com [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e ".[dev]"
```

Ou com pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

No Linux sem ambiente gráfico completo, o Qt pode precisar de
`libegl1 libgl1 libxkbcommon0 libfontconfig1 libdbus-1-3`.

## Uso da interface

```bash
polypseg-gui                          # abre vazio
polypseg-gui caminho/imagem.jpg       # uma imagem
polypseg-gui data/Kvasir-SEG          # pasta com images/ e masks/
```

| Ação | Comando |
| --- | --- |
| Marcar pólipo | botão esquerdo |
| Marcar fundo | botão direito |
| Zoom / arrastar | roda do mouse / botão do meio |
| Desfazer / limpar | Ctrl+Z / Esc |
| Imagem anterior / próxima | PgUp / PgDn |
| Salvar máscara | Ctrl+S |

Ao abrir uma pasta no formato do Kvasir-SEG (`images/` e `masks/`), o gabarito é
carregado sozinho e o painel mostra o Dice e o IoU a cada clique. Ao salvar, além
do PNG da máscara é gravado um `.json` com os cliques, a configuração e o
**tempo de anotação**, útil para medir custo de anotação no artigo.

No painel dá para trocar o algoritmo de superpixels (SLIC, Felzenszwalb,
watershed), a quantidade, os grupos de descritores e o modelo.

## Método

1. **Pré-processamento:** reflexos especulares (pixels claros e pouco saturados)
   são preenchidos com a cor vizinha.
2. **Superpixels:** SLIC no espaço Lab (padrão: 200 regiões).
3. **Descritores por superpixel (68 no total):**
   - cor: média e desvio em RGB, Lab e HSV, com a matiz como cos/sen;
   - LBP uniforme com raios 1 e 2 (histogramas);
   - GLCM de Haralick restrita aos pares de pixels dentro do superpixel, em 4
     direções (contraste, dissimilaridade, homogeneidade, energia, correlação, entropia);
   - banco de Gabor com 2 frequências e 4 orientações (média e desvio da magnitude).
4. **Grafo de superpixels:** arestas entre vizinhos, com peso
   `exp(-‖fᵢ − fⱼ‖² / (β · mediana))` sobre os descritores padronizados.
5. **Propagação:** os superpixels clicados são sementes. Os superpixels da borda
   da imagem entram como fundo, a menos que tenham sido clicados como pólipo.
   Um *random walker* no grafo calcula a probabilidade de pólipo de cada
   superpixel.
6. **Pós-processamento:** mantém só as regiões conexas que contêm cliques de
   pólipo e preenche buracos.

Os modelos alternativos são: `hybrid` (um SVM linear dá a probabilidade inicial
e o random walker suaviza), `svm` (SVM linear) e `rf` (Random Forest). Todos
foram piores que o grafo puro, porque treinar classificadores com poucos cliques
gera muita variância.

## Avaliação com cliques simulados

```bash
polypseg-eval --data data/Kvasir-SEG --max-clicks 10 --jobs 4 --out results/kvasir.csv
polypseg-eval --data data/Kvasir-SEG --features color --jobs 4   # ablação: só cor
```

Os cliques seguem o protocolo usual de segmentação interativa: o 1º clique cai no
centro do pólipo e cada novo clique cai no ponto mais interno da maior região de
erro. Um falso negativo vira clique de pólipo e um falso positivo vira clique de
fundo. O script relata Dice e IoU médios por número de cliques e NoC@85/90
(cliques até atingir IoU de 85% ou 90%).

### Resultados preliminares: Kvasir-SEG (1000 imagens)

Configuração: SLIC com 200 superpixels, grafo, imagens reduzidas para 512 px no
maior lado, 10 cliques.

| Cliques | mDice (todos os descritores) | mDice (só cor) |
| --- | --- | --- |
| 1 | 0,507 | 0,499 |
| 2 | 0,702 | 0,685 |
| 3 | 0,772 | 0,761 |
| 5 | 0,832 | 0,825 |
| 10 | 0,878 | 0,875 |
| NoC@85 / NoC@90 | 8,32 / 9,61 | 8,27 / 9,53 |

Leitura dos resultados:

- A textura (LBP, GLCM e Gabor) ajuda pouco sobre a cor: cerca de +0,01 a +0,02
  de Dice nos primeiros cliques e quase nada com 10 cliques.
- O NoC alto mostra que atingir IoU de 90% esbarra na precisão das bordas dos
  superpixels. Há espaço para refinar a borda no nível de pixel.
- Os parâmetros (β = 0,5 e 200 superpixels) foram escolhidos em 60 imagens do
  próprio Kvasir-SEG, então estes números ainda não são uma avaliação independente.
  Para o artigo, avaliar em CVC-ClinicDB, ETIS, ColonDB e CVC-300 com o protocolo
  do PraNet.

## Estrutura

```
src/polypseg/
  preprocessing.py   leitura, reflexos especulares, moldura preta
  superpixels.py     SLIC / Felzenszwalb / watershed e grafo de adjacência
  features.py        cor, LBP, GLCM restrita, Gabor (vetorizado com bincount)
  segmenter.py       segmentador interativo (grafo, híbrido, SVM, RF)
  simulation.py      simulação de cliques
  metrics.py         Dice, IoU, NoC
  evaluation.py      CLI de avaliação (polypseg-eval)
  gui/               interface PySide6 (polypseg-gui)
tests/               pytest, incluindo teste de fumaça da interface em modo offscreen
```

Para rodar os testes:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

## Próximos passos

- Refinar a borda no nível de pixel, por exemplo com GrabCut inicializado pela
  máscara de superpixels.
- Comparar com SAM, SAM 2 e MedSAM usando os mesmos cliques simulados.
- Fase automática: classificador de superpixels treinado em várias imagens e
  U-Net treinada com as máscaras assistidas como pseudo-rótulos.
- Estudo com anotadores reais, usando o tempo gravado no `.json`.
