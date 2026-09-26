# Backlog

## P0 - necessário para o MVP

- [x] Fundação: pacote, configurações, proveniência, logging e CLI.
- [x] Taxonomia centralizada e contratos de `dim_crop`/`dim_location`.
- [x] Ingerir lista/malha municipal oficial do IBGE e materializar `dim_location`.
- [x] Ingerir PAM/IBGE com Bronze imutável, manifesto e schema Silver.
- [x] Materializar `production_history.parquet` e `crop_location_year.parquet`.
- [x] Implementar features históricas sem vazamento temporal.
- [x] Ingerir custos e preços Conab preservando unidade e linha original.
- [x] Implementar calculadora econômica pura e testes determinísticos.
- [x] Ingerir ZARC com `eligible`, `not_eligible` e `not_available`.
- [x] Implementar baselines de produtividade e validações temporal/geográfica.
- [x] Implementar intervalos P10/P50/P90 e avaliar cobertura.
- [x] Implementar Monte Carlo, confiança e ranking configurável.
- [x] Implementar FastAPI, Streamlit e auditoria ponta a ponta para dados disponíveis.

## P1 - importante

- [x] Ingerir INMET e gerar features climáticas por ciclo.
- [x] Validar autenticação e download mínimo real do ERA5-Land.
- [x] Corrigir fragmentação sazonal por revisões de metadados INMET até 1 km.
- [x] Implementar parser ERA5 NetCDF/ZIP, unidades, acumulados UTC e proveniência.
- [x] Implementar fila local retomável, limites, exclusão mútua e testes do fallback.
- [x] Exibir fonte climática e períodos reais em API, auditoria e Streamlit.
- [x] Complementar cobertura com ERA5-Land sem misturar fontes silenciosamente
  (645 lacunas preenchidas; 9 indisponibilidades explícitas na grade consultada).
- [x] Retreinar e avaliar separadamente INMET/ERA5 nos splits temporal/geográfico.
- [ ] Alinhar horizonte de previsão, custos/preços e safra ZARC; alerta não resolve o desalinhamento.
- [x] Avaliar CatBoost, LightGBM e XGBoost contra baselines.
- [x] Integrar MLflow quando o primeiro treino real existir.
- [ ] Adicionar PostgreSQL/PostGIS e migrations para estado transacional/espacial.
- [ ] Generalizar o vertical validado para milho, feijão, arroz, trigo e algodão.

## P2 - evolução

- [ ] Adapter de solos Embrapa/GeoInfo/PronaSolos.
- [ ] Adapter de uso do solo MapBiomas.
- [ ] Logística e armazenagem.
- [ ] Dependência empírica entre preço, produtividade e custo no Monte Carlo.
- [ ] Fluxo de caixa plurianual para culturas permanentes.
