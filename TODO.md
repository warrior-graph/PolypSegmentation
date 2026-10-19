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

Observações:

- [ ] T3: o mapa de fundo separa bem pólipo de mucosa (AUC 0,84 no CVC), mas não muda a
      segmentação, porque a estrela e a regra de borda já cobrem esse papel. Testar como
      limite de tamanho da estrela em vez de prior.
- [ ] T1: é a única com ganho nas duas bases, mas pequeno. Escolher a polaridade sozinha
      ajuda o CVC (Dice@1 0,546 → 0,584) e piora o Kvasir (0,685 → 0,637); escolher só a
      expansão faz o contrário. Falta um critério melhor que a separação de Fisher em Lab,
      por exemplo a energia de Chan-Vese ou um modelo de cor (GMM) por imagem.
- [ ] No CVC, 31% das estrelas do 1º clique ainda saem com mais que o dobro da área do
      pólipo; no Kvasir, 16% saem com menos de 30%. É o maior erro restante.

## Pendentes

- [ ] T2 Estrela geodésica e transformada imagem-floresta (Falcão et al., 2004; Gulshan et al., 2010)
- [ ] T4 Normalização da iluminação por Retinex multiescala (Jobson et al., 1997)
- [ ] T6 Reflexos especulares como pista de cúpula (hipótese sem apoio publicado)
- [ ] T9 Várias tesselações ou hierarquia OWT-UCM (Arbeláez et al., 2011)
- [ ] T10 Power watershed ou corte em grafo no lugar do random walker (Couprie et al., 2011)
- [ ] T11 Prior de convexidade ou descritores de Fourier elípticos (Gorelick et al., 2017; Kuhl e Giardina, 1982)
- [ ] Repetir `scripts/diagnose.py` nas duas bases com o método atual, para separar vazamento de erro de borda
- [ ] Fatorial e teste único só com o que passar no corte
