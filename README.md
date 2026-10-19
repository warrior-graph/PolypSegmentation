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

### Bases de dados

```bash
pip install -e ".[data]"
python scripts/download_data.py
```

O script baixa para `data/` o Kvasir-SEG completo (1000 imagens) e os cinco
conjuntos de teste do protocolo PraNet: CVC-300 (60), CVC-ClinicDB (62),
CVC-ColonDB (380), ETIS-LaribPolypDB (196) e Kvasir (100). Os arquivos vêm de
espelhos no Hugging Face, porque o site oficial do Kvasir-SEG não passa na
verificação de certificado. Ao publicar, cite os artigos originais de cada base.

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
2. **Superpixels:** SLIC no espaço Lab (padrão: 400 regiões).
3. **Descritores por superpixel (68 no total):**
   - cor: média e desvio em RGB, Lab e HSV, com a matiz como cos/sen;
   - LBP uniforme com raios 1 e 2 (histogramas);
   - GLCM de Haralick restrita aos pares de pixels dentro do superpixel, em 4
     direções (contraste, dissimilaridade, homogeneidade, energia, correlação, entropia);
   - banco de Gabor com 2 frequências e 4 orientações (média e desvio da magnitude).
4. **Grafo de superpixels:** arestas entre vizinhos, com peso
   `exp(-‖fᵢ − fⱼ‖² / (β · mediana))` sobre os descritores padronizados.
5. **Propagação:** os superpixels clicados são sementes. Os superpixels da borda
   da imagem e os encostados na moldura preta do endoscópio entram como fundo,
   a menos que tenham sido clicados como pólipo.
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
polypseg-eval --data data/TestDataset/ETIS-LaribPolypDB --jobs 4
polypseg-eval --data data/Kvasir-SEG --features color --jobs 4   # ablação: só cor
```

Os cliques seguem o protocolo usual de segmentação interativa: o 1º clique cai no
centro do pólipo e cada novo clique cai no ponto mais interno da maior região de
erro. Um falso negativo vira clique de pólipo e um falso positivo vira clique de
fundo. O script relata Dice e IoU médios por número de cliques e NoC@85/90
(cliques até atingir IoU de 85% ou 90%).

### Resultados preliminares

Configuração padrão: SLIC com 400 superpixels, todos os descritores, grafo,
imagens reduzidas para 512 px no maior lado, 10 cliques simulados.

| Base | Imagens | Dice@1 | Dice@3 | Dice@5 | Dice@10 | NoC@85 |
| --- | --- | --- | --- | --- | --- | --- |
| Kvasir-SEG (completo) | 1000 | 0,434 | 0,771 | 0,847 | 0,899 | 8,14 |
| Kvasir (teste PraNet) | 100 | 0,419 | 0,757 | 0,828 | 0,891 | 8,17 |
| CVC-300 | 60 | 0,690 | 0,764 | 0,815 | 0,858 | 9,27 |
| CVC-ClinicDB | 62 | 0,426 | 0,727 | 0,786 | 0,842 | 9,02 |
| CVC-ColonDB | 380 | 0,549 | 0,734 | 0,797 | 0,844 | 9,47 |
| ETIS-LaribPolypDB | 196 | 0,541 | 0,690 | 0,759 | 0,800 | 9,16 |

NoC@90 fica entre 9,6 e 10 em todas as bases, ou seja, quase nunca se chega a IoU de 90%.

**Ablação: só cor vs. cor + textura** (diferença de Dice, só cor menos todos)

| Base | ΔDice@1 | ΔDice@5 | ΔDice@10 | NoC@85 só cor |
| --- | --- | --- | --- | --- |
| Kvasir-SEG (completo) | +0,048 | −0,004 | −0,003 | 7,80 |
| CVC-300 | −0,152 | −0,009 | −0,001 | 9,05 |
| CVC-ClinicDB | +0,053 | +0,007 | +0,014 | 8,58 |
| CVC-ColonDB | −0,046 | −0,011 | −0,001 | 9,23 |
| ETIS-LaribPolypDB | +0,037 | −0,011 | −0,004 | 9,03 |

Leitura dos resultados:

- **Bases menores são mais difíceis.** Com 10 cliques o Dice cai de 0,90 no
  Kvasir-SEG para 0,84 nos CVC e 0,80 no ETIS, que tem pólipos pequenos e
  imagens de 1225×966 reduzidas para 512 px.
- **A moldura preta importava muito.** Nos CVC a borda da imagem é quase só
  moldura. Antes de incluir os superpixels encostados nela como fundo, o 1º
  clique marcava cerca de 59% da imagem e o Dice@1 do CVC-300 era 0,10. Com a
  correção, subiu para 0,60 (com 200 superpixels).
- **400 superpixels foi melhor que 200** a partir do 3º clique em todas as bases:
  +0,04 a +0,08 de Dice@5 e Dice@10 nos CVC e no ETIS. O custo é um 1º clique um
  pouco pior.
- **A textura não traz ganho consistente.** Com 5 ou 10 cliques a diferença fica
  dentro de ±0,015 de Dice, e o NoC@85 é melhor só com cor em todas as bases. A
  exceção é o 1º clique no CVC-300 (+0,15 com textura).
- **A borda é o gargalo.** O NoC@90 perto de 10 mostra que a precisão fica
  limitada às bordas dos superpixels.
- **Os parâmetros não foram escolhidos às cegas.** β = 0,5 foi ajustado em 60
  imagens do Kvasir-SEG. Os 400 superpixels e a regra da moldura foram decididos
  olhando estes resultados. Para o artigo, fixe a configuração numa base de
  desenvolvimento e relate as demais como teste.

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
scripts/             download das bases de dados
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
