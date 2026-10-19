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
watershed), a quantidade (10 a 20000) e a compacidade, os grupos de descritores, o
modelo e os módulos do método (arestas de borda, estrela, refino por pixel, relevo
local e suavização da borda). Em "Visualização" escolhem-se as cores da máscara,
das fronteiras e do gabarito e a espessura das fronteiras; essas preferências ficam
salvas entre sessões.

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
4. **Grafo de superpixels com arestas cortadas pela borda** (`boundary_edges`):
   arestas entre vizinhos com peso `exp(-‖fᵢ − fⱼ‖² / (β · med)) · exp(-b̄ᵢⱼ / (γ · med))`,
   em que `b̄ᵢⱼ` é a média de um mapa de borda **ao longo da fronteira** entre os dois
   superpixels (princípio dos *intervening contours*). O mapa combina o gradiente em
   Lab com o vale escuro no canal azul, que marca a borda do pólipo
   (Bernal et al., 2015, WM-DOVA), e é zerado na moldura preta.
5. **Prior em estrela centrado no clique** (`radial_prior`): em volta do 1º clique
   de pólipo, a evidência de borda e o escurecimento ao se afastar do clique (a luz
   sai junto da câmera, então a cúpula do pólipo é mais clara que o entorno) são
   amostrados em coordenadas polares. Uma programação dinâmica circular escolhe o
   contorno fechado r(θ) de menor custo, com |Δr| limitado, sem atravessar a moldura.
   É a busca radial do WM-DOVA, condicionada ao clique, com a restrição de forma em
   estrela de Veksler (2008) e Gulshan et al. (2010). **Os cliques seguintes viram
   restrições da própria DP:** o contorno passa por fora dos cliques de pólipo e por
   dentro dos de fundo. O contorno vira um prior suave no random walker, com peso
   relativo ao grau médio do grafo e dividido pelo número de cliques.
6. **Propagação:** os superpixels clicados são sementes. Os superpixels da borda
   da imagem e os encostados na moldura preta entram como fundo, a menos que tenham
   sido clicados como pólipo. Um *random walker* com o prior acima calcula a
   probabilidade de pólipo de cada superpixel.
7. **Refino por pixel** (`pixel_refine`): um random walker por pixel (Grady, 2006)
   decide só a faixa de ±14 px em volta da borda da máscara, com o interior e o
   exterior da faixa como sementes. Só os pixels da faixa entram no sistema linear.
8. **Pós-processamento:** suavização opcional da borda (`smooth_sigma`, desligada
   por padrão; suaviza a distância assinada e corta no nível zero, um passo de fluxo
   por curvatura), depois mantém só as regiões conexas que contêm cliques de pólipo
   e preenche buracos.

Os passos 4, 5 e 7 vêm ligados por padrão. Também existem, desligados: o relevo
local no mapa de borda (`relief_map`, pseudo-elevação por sombreamento coaxial no
canal R mais o *Local Relief Model* da arqueologia com LiDAR), um grupo de
descritores de relevo por Hessiana (`FeatureConfig(relief=True)`) e o balanceamento
dos grupos de descritores na distância do grafo (`balance_groups`). Nenhum deles
melhorou a avaliação.

Os modelos alternativos são: `hybrid` (um SVM linear dá a probabilidade inicial
e o random walker suaviza), `svm` (SVM linear) e `rf` (Random Forest). Todos
foram piores que o grafo puro, porque treinar classificadores com poucos cliques
gera muita variância.

**Limitação conhecida:** a estrela supõe uma cúpula mais clara que a mucosa em volta.
Num pólipo mais escuro que o entorno o contorno sai pequeno e o 1º clique piora; na
interface, basta desmarcar "Estrela em volta do clique". Escolher a polaridade por
imagem foi testado e piorou a base de desenvolvimento.

## Avaliação com cliques simulados

```bash
polypseg-eval --data data/Kvasir-SEG --max-clicks 10 --jobs 4 --out results/kvasir.csv
polypseg-eval --data data/TestDataset/ETIS-LaribPolypDB --jobs 4
polypseg-eval --data data/Kvasir-SEG --modules none --jobs 4       # método anterior
polypseg-eval --data data/Kvasir-SEG --list data/dev.txt --jobs 4  # só a base de desenvolvimento
polypseg-eval --data data/Kvasir-SEG --smooth 4 --jobs 4           # com suavização da borda
```

Os cliques seguem o protocolo usual de segmentação interativa: o 1º clique cai no
centro do pólipo e cada novo clique cai no ponto mais interno da maior região de
erro. Um falso negativo vira clique de pólipo e um falso positivo vira clique de
fundo. O script relata Dice e IoU médios por número de cliques e NoC@85/90
(cliques até atingir IoU de 85% ou 90%).

Scripts de apoio:

```bash
python scripts/split_dev.py                    # gera data/dev.txt (200 imagens, semente 0)
python scripts/diagnose.py --data data/Kvasir-SEG --list data/dev.txt --jobs 12   # tetos
python scripts/factorial.py --data data/Kvasir-SEG --list data/dev.txt --jobs 12  # 2⁴ módulos
```

### Protocolo

- **Desenvolvimento:** 200 imagens sorteadas de `data/train.txt` do Kvasir-SEG,
  sem as 100 que também estão no teste Kvasir do PraNet. Todos os parâmetros e
  escolhas de módulos foram feitos só nessa base.
- **Teste:** `val.txt` do Kvasir-SEG (120) e os cinco conjuntos do PraNet.
- Imagens reduzidas para 512 px no maior lado, 10 cliques simulados.

### Diagnóstico com oráculos (desenvolvimento, método anterior)

| Cenário | Dice@1 | Dice@3 | Dice@10 | NoC@85 | NoC@90 |
| --- | --- | --- | --- | --- | --- |
| Método anterior | 0,433 | 0,762 | 0,899 | 8,10 | 9,56 |
| Mesmo grafo, arestas cortadas exatamente na borda do gabarito | 0,772 | 0,883 | 0,926 | 4,58 | 7,17 |

Rotulando cada superpixel pela maioria do gabarito, o IoU máximo é 0,858, 0,890 e
0,917 com 200, 400 e 800 superpixels; com 400, só 51% das imagens conseguem IoU ≥ 0,90.
Depois do 1º clique, 76% dos pixels errados estão a mais de 5 px da borda do gabarito.
Conclusão: o maior espaço estava nas arestas (vazamento), e o NoC@90 era limitado
pelos superpixels.

### Fatorial dos módulos (desenvolvimento)

| Módulos | Dice@1 | Dice@3 | Dice@5 | Dice@10 | NoC@85 | NoC@90 |
| --- | --- | --- | --- | --- | --- | --- |
| nenhum (método anterior) | 0,433 | 0,762 | 0,846 | 0,899 | 8,10 | 9,56 |
| arestas de borda (M2) | 0,546 | 0,789 | 0,864 | 0,911 | 7,01 | 8,78 |
| estrela (M3) | 0,647 | 0,809 | 0,860 | 0,905 | 7,50 | 9,14 |
| refino por pixel (M4) | 0,412 | 0,776 | 0,860 | 0,924 | 6,68 | 8,44 |
| M2 + M3 | 0,690 | 0,832 | 0,876 | 0,912 | 6,53 | 8,72 |
| M2 + M3 + M4 | 0,687 | 0,842 | 0,898 | 0,934 | 5,05 | 7,09 |
| M2 + M3 + M4, com a correção da moldura (padrão atual) | 0,685 | 0,860 | 0,899 | 0,936 | 4,92 | 6,99 |

Efeitos principais sobre NoC@85: M2 −0,96, M3 −0,63, M4 −1,50; o relevo local (M1)
teve efeito nulo (−0,01) e ficou desligado. Os ganhos se somam quase sem interação.
Antes desta etapa, descritores novos (relevo por Hessiana, balanceamento dos grupos)
mudaram o Dice em no máximo 0,02, com intervalos de confiança cruzando zero.

### Teste

**v1: rodada única com a configuração congelada no desenvolvimento.**
A estrela ainda podia atravessar a moldura preta.

**v2: versão atual.** Depois da v1, um diagnóstico mostrou que nas bases CVC a estrela
do 1º clique ia até a moldura preta (área mediana 2,5 a 8,8 vezes a do pólipo). A
correção (a estrela não atravessa a moldura) foi conferida só na base de
desenvolvimento, mas **foi motivada pelo teste, então a v2 não é uma avaliação
independente**. Uma base de desenvolvimento só com imagens CVC resolveria isso.

| Base | Imagens | Método | Dice@1 | Dice@3 | Dice@5 | Dice@10 | NoC@85 | NoC@90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Kvasir-SEG (val) | 120 | anterior | 0,428 | 0,749 | 0,836 | 0,891 | 8,58 | 9,78 |
| | | só cor | 0,470 | 0,763 | 0,830 | 0,890 | 8,11 | 9,53 |
| | | v1 | 0,666 | 0,822 | 0,880 | 0,924 | 5,92 | 7,76 |
| | | **v2** | 0,668 | 0,850 | 0,894 | 0,925 | **5,84** | 7,77 |
| Kvasir (PraNet) | 100 | anterior | 0,419 | 0,757 | 0,828 | 0,891 | 8,18 | 9,65 |
| | | só cor | 0,454 | 0,771 | 0,833 | 0,880 | 8,00 | 9,45 |
| | | v1 | 0,669 | 0,832 | 0,861 | 0,925 | 5,21 | 7,25 |
| | | **v2** | 0,673 | 0,844 | 0,880 | 0,923 | **5,25** | 7,12 |
| CVC-300 | 60 | anterior | 0,690 | 0,764 | 0,815 | 0,858 | 9,27 | 10,00 |
| | | só cor | 0,540 | 0,767 | 0,806 | 0,854 | 9,03 | 10,00 |
| | | v1 | 0,419 | 0,818 | 0,859 | 0,915 | 6,45 | 9,08 |
| | | **v2** | 0,810 | 0,864 | 0,892 | 0,915 | **5,78** | 8,80 |
| CVC-ClinicDB | 62 | anterior | 0,426 | 0,727 | 0,786 | 0,842 | 9,02 | 9,94 |
| | | só cor | 0,479 | 0,695 | 0,793 | 0,852 | 8,53 | 9,65 |
| | | v1 | 0,456 | 0,650 | 0,778 | 0,874 | 7,21 | 8,63 |
| | | **v2** | 0,535 | 0,728 | 0,819 | 0,879 | **6,94** | 8,31 |
| CVC-ColonDB | 380 | anterior | 0,549 | 0,734 | 0,797 | 0,844 | 9,47 | 9,92 |
| | | só cor | 0,503 | 0,719 | 0,787 | 0,843 | 9,23 | 9,88 |
| | | v1 | 0,429 | 0,680 | 0,792 | 0,893 | 7,87 | 9,26 |
| | | **v2** | 0,579 | 0,758 | 0,835 | 0,897 | **7,39** | 9,12 |
| ETIS-LaribPolypDB | 196 | anterior | 0,541 | 0,690 | 0,759 | 0,800 | 9,16 | 9,85 |
| | | só cor | 0,577 | 0,701 | 0,748 | 0,796 | 8,98 | 9,84 |
| | | v1 | 0,582 | 0,760 | 0,825 | 0,886 | 6,33 | 8,43 |
| | | **v2** | 0,584 | 0,778 | 0,841 | 0,885 | **6,16** | 8,35 |

Diferença pareada de NoC@85 contra o método anterior (IC 95% por bootstrap; teste de
Wilcoxon com correção de Holm nas 6 bases):

| Base | v1 | v2 |
| --- | --- | --- |
| Kvasir-SEG (val) | −2,67 [−3,25; −2,10] | −2,74 [−3,33; −2,18] |
| Kvasir (PraNet) | −2,97 [−3,56; −2,38] | −2,93 [−3,49; −2,37] |
| CVC-300 | −2,82 [−3,68; −1,98] | −3,48 [−4,32; −2,67] |
| CVC-ClinicDB | −1,81 [−2,57; −1,07] | −2,08 [−2,87; −1,29] |
| CVC-ColonDB | −1,60 [−1,86; −1,35] | −2,08 [−2,36; −1,80] |
| ETIS-LaribPolypDB | −2,84 [−3,32; −2,38] | −3,01 [−3,48; −2,55] |

Todos com p (Holm) ≤ 6×10⁻⁵, e o ganho se mantém contra o método só com cor.

Leitura dos resultados:

- **O método novo economiza de 1,6 a 3,5 cliques para chegar a IoU 85% em todas as
  bases**, e o NoC@90 deixou de ficar preso perto de 10.
- **Na v1, o 1º clique piorou no CVC-300 (0,69 → 0,42) e no CVC-ColonDB
  (0,55 → 0,43)**, porque a estrela vazava pela moldura; na v2, os dois sobem acima
  do método anterior (0,81 e 0,58).
- **CVC-ClinicDB e CVC-ColonDB continuam as bases mais difíceis** (NoC@85 de 6,9 e
  7,4), com pólipos pequenos e muito contato com a moldura.
- **Tempo:** preparo de cerca de 0,6 s por imagem de 512 px e cerca de 75 ms por
  clique (máximo de cerca de 140 ms), ainda interativo.
- **Suavização da borda:** no desenvolvimento, σ = 4 px deixa as métricas iguais
  (Dice@3 0,870, NoC@85 4,93); sem o refino por pixel ela ajuda (NoC@85 6,46 → 6,09
  com σ = 8). Fica desligada por padrão.

## Estrutura

```
src/polypseg/
  preprocessing.py   leitura, reflexos especulares, moldura preta
  superpixels.py     SLIC / Felzenszwalb / watershed e grafo de adjacência
  features.py        cor, LBP, GLCM restrita, Gabor, relevo (vetorizado com bincount)
  boundary.py        mapa de borda e força de borda entre superpixels vizinhos
  radial.py          contorno em estrela por transformada polar e DP circular
  refine.py          random walker por pixel na faixa da borda e suavização
  relief.py          pseudo-elevação coaxial e Local Relief Model
  segmenter.py       segmentador interativo (grafo, híbrido, SVM, RF)
  simulation.py      simulação de cliques
  metrics.py         Dice, IoU, NoC
  evaluation.py      CLI de avaliação (polypseg-eval)
  gui/               interface PySide6 (polypseg-gui)
scripts/             download das bases, base de desenvolvimento, oráculos, fatorial
tests/               pytest, incluindo teste de fumaça da interface em modo offscreen
```

Para rodar os testes:

```bash
QT_QPA_PLATFORM=offscreen pytest
```

## Próximos passos

- Validar a correção da moldura numa base de desenvolvimento só com imagens CVC
  (por exemplo, as 550 do CVC-ClinicDB no treino do PraNet) e rodar um teste limpo.
- Tratar pólipos mais escuros que o entorno na estrela sem piorar os claros.
- Comparar com SAM, SAM 2 e MedSAM usando os mesmos cliques simulados.
- Fase automática: classificador de superpixels treinado em várias imagens e
  U-Net treinada com as máscaras assistidas como pseudo-rótulos.
- Estudo com anotadores reais, usando o tempo gravado no `.json`.
