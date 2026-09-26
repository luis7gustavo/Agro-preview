# Avaliação do modelo de produtividade de soja

Data da execução: 25/09/2026, após complemento climático ERA5-Land ARCO  
Dataset: `ml-soy-662c0726d3b7`  
Target: `produtividade_kg_ha`  
Linhas: 14.997; municípios: 2.716; anos: 2019–2024; features: 29.

Proveniência: 14.743 linhas com INMET e 254 com ERA5-Land ARCO; nenhuma lacuna
nos valores climáticos do dataset. A distância à estação permanece nula nas
254 linhas de reanálise, pois não existe estação associada. Grade/distância à
grade são preservadas nos dados climáticos, sem fingir observação instrumental.

## Contrato contra leakage

- histórico municipal deslocado antes do ano alvo;
- mediana regional da UF em `Y-1`;
- clima da safra concluída `Y-1`;
- área e produção do próprio ano alvo excluídas das features.

## Splits

- temporal: treino 2019–2022 (9.754 linhas), teste 2023–2024 (5.243);
- geográfico: 2.174 municípios no treino e 542 nunca vistos no teste
  (12.024/2.973 linhas), selecionados por hash determinístico.

## MAE em kg/ha

| Modelo | Temporal | Geográfico | Vence o melhor baseline nos dois? |
|---|---:|---:|---|
| Média móvel 5 anos (melhor baseline) | 653,72 | 690,27 | referência |
| CatBoost | 508,20 | 319,73 | sim |
| LightGBM | **482,84** | **306,27** | **sim** |
| XGBoost | 492,68 | 303,20 | sim |

O LightGBM foi selecionado pela menor média das razões de MAE contra o melhor
baseline em cada split. No teste temporal obteve RMSE 662,30 kg/ha, sMAPE 16,99%
e R² 0,1665; no geográfico, RMSE 431,46 kg/ha, sMAPE 11,35% e R² 0,6716.

O treino anterior, de 17/09, tinha MAE temporal/geográfico de 488,07/305,95 kg/ha.
O novo resultado melhora o temporal e piora ligeiramente o MAE geográfico;
não representa melhora universal nem um experimento causal sobre ERA5.

## Avaliação por fonte climática

As comparações abaixo usam as mesmas linhas finitas para modelo e três baselines.
São diagnósticas; não alteram o gate global de promoção.

| Fonte / split | Linhas avaliadas | MAE LightGBM | MAE melhor baseline |
|---|---:|---:|---:|
| INMET temporal | 5.054 | 489,95 | 648,30 |
| ERA5 temporal | 187 | 292,19 | 685,27 |
| INMET geográfico | 2.920 | 308,69 | 688,87 |
| ERA5 geográfico | 53 | 173,01 | 576,58 |

Duas linhas INMET temporais foram excluídas desse diagnóstico por baseline
municipal ausente; continuam no teste global do modelo. O melhor baseline ERA5
é a mediana regional defasada, não a média móvel. Apesar de vencê-lo em MAE,
o R² temporal ERA5 é negativo (-0,2340). As amostras ERA5 são pequenas e não
validam desempenho em todos os municípios ou regimes climáticos brasileiros.

## Gate e intervalo

O gate exige MAE inferior ao melhor baseline tanto no tempo quanto em municípios
não vistos. O gate passou e promoveu
`soja-lightgbm-20260925T235302Z-662c0726`.

O intervalo usa split conformal com resíduos absolutos. Treinando até 2022,
calibrando em 2023 e avaliando em 2024, a cobertura empírica foi 75,08% para uma
cobertura nominal de 80%, com largura média de 1.437,49 kg/ha. O artefato de
produção foi treinado até 2023 e calibrado em 2024, com raio de 837,03 kg/ha.

Este treino inclui INMET e ERA5-Land ARCO. O dataset preserva fonte, método e
natureza dos dados; a inferência sinaliza fontes ausentes do treino. A cobertura
do intervalo ficou abaixo de 80% e exige monitoramento/recalibração com novas
safras, sem ajustar parâmetros ao teste já observado. O horizonte atual ainda
pode diferir de custos, preços e ZARC, o que
é exposto em `temporal_context`, não tratado como previsão alinhada.

## Rastreabilidade

O relatório completo está em `reports/generated/model_evaluation_soja.json`.
Dataset, features, target, params, métricas, timestamp, commit e artefatos foram
registrados no experimento local `soy-yield` do MLflow. Como o repositório ainda
não possui commit, o campo de commit registra `uncommitted` honestamente.
