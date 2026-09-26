# Relatório de validação V1 — vertical soja

Data: 20/08/2026

## Dados materializados

- IBGE Localidades: 5.571 municípios; 5.570 geometrias.
- IBGE/PAM soja: 283.713 linhas, 1974–2024.
- Conab custos: 265 linhas, 13 polos, 10 UFs, 2019–2026.
- Conab preços: 753 linhas municipais/estaduais, 17 UFs, 2025–2026.
- MAPA/ZARC 2026/2027: 4.332 municípios `eligible`, 265 `not_eligible` e
  974 `not_available`.
- INMET 2017–2025: 1.872.835 registros estação-dia, 631 estações, 6.795
  estações-safra e 44.560 município-safra.
- Cobertura INMET dentro de 250 km: 40.296 município-safra; fora: 4.264.
- Dataset ML: 14.997 linhas, 2.716 municípios, 26 UFs, 2019–2024 e 29 features.
- Bronze oficial preservado com SHA-256 e manifestos.

## Modelo

- Splits: temporal 2019–2022/2023–2024 e holdout de 542 municípios.
- Melhor baseline: média móvel 5 anos, MAE 653,72 temporal e 690,27 geográfico.
- Vencedor: LightGBM, MAE 488,17 temporal e 306,42 geográfico.
- Gate: aprovado nos dois splits.
- Versão: `soja-lightgbm-20260820T135716Z-a8600927`.
- Intervalo conformal nominal 80%: cobertura empírica 75,91% em 2024.
- MLflow: três runs com dataset, features, target, params, métricas e artefato.

## Canary ponta a ponta

Sorriso/MT (`5107925`), área de 500 ha, perfil `balanced`:

- previsão de produtividade para 2025: P10 2.768,14; P50 3.592,05; P90
  4.415,96 kg/ha;
- clima de referência 2024: estação INMET A904 a 20,74 km, cobertura 99,62%;
- custo total observado Conab: R$ 6.329,44/ha;
- preço municipal P50: R$ 1,99/kg (5 meses, alerta amostral);
- lucro base: R$ 818,74/ha; ROI: 12,94%;
- probabilidade de lucro: 86,35% em 10.000 simulações;
- ZARC: `eligible`, melhor risco 20%;
- score: 68,30309; confiança: 85,79/100;
- alerta restante: `LOW_PRICE_SAMPLE`.

A auditoria persiste checksums/versões dos datasets, modelo, configuração,
parâmetros, fontes e hipóteses. O resultado é apoio à decisão e não substitui
assistência agronômica local.

## ERA5-Land

Atualização em 17/09/2026: credencial e termos configurados. Uma requisição
autenticada real foi aceita e entregou um NetCDF ERA5-Land de 25.116 bytes para
o canário de Sorriso/MT (`2m_temperature`, 01/01/2025 00:00). O arquivo foi
preservado no Bronze com SHA-256 e manifesto. O backfill das 4.264 lacunas INMET
ainda não foi materializado, portanto nenhuma fonte foi misturada silenciosamente.

## Software

- suíte completa: 52 testes aprovados;
- Ruff: sem ocorrências;
- MyPy: modo estrito, sem ocorrências em 57 módulos;
- FastAPI: `health`, cobertura e POST real validados; Streamlit respondeu HTTP 200.
