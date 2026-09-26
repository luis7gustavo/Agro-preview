# Status

Atualizado em 25/09/2026. Os relatórios em `reports/generated/` registram o estado executado.

## Concluído

- Fundação modular, configurações versionadas, proveniência, logging e CLI.
- Taxonomia com soja ativa e cinco culturas explicitamente `planned`.
- IBGE Localidades/Malhas e PAM 1974–2024, com features históricas sem leakage.
- Custos e preços Conab, ZARC MAPA 2026/2027 e estados de cobertura explícitos.
- INMET 2017–2025 em Bronze/Silver/Gold: 1.872.835 estações-dia e 44.560
  município-safra, com distância, qualidade e fonte preservadas.
- Dataset de ML de soja com 14.997 linhas, 29 features e contrato `Y-1`.
- Baselines municipal/móvel/regional e avaliação temporal/geográfica.
- CatBoost, LightGBM e XGBoost registrados no MLflow; LightGBM promovido pelo gate.
- Intervalos P10/P50/P90 por split conformal e cobertura avaliada.
- Calculadora econômica, Monte Carlo, confiança, ranking, FastAPI, Streamlit e
  auditoria ponta a ponta usando produtividade prevista quando disponível.
- Credencial CDS configurada e download autenticado ERA5-Land validado com um
  recorte NetCDF de Sorriso/MT, preservado no Bronze com checksum e manifesto.
- Revisões de coordenadas INMET até 1 km e mesma UF agora são canonicalizadas
  por estação-safra; deslocamentos maiores continuam separados e auditáveis.
  Referência INMET reconstruída: 43.906 de 44.560 registros adequados; 654 lacunas
  antes do complemento ERA5.
- Backfill ERA5 implementado e testado: pedidos mensais por bloco espacial,
  IDs persistidos, limite global de três pedidos ativos, lock entre processos,
  checksums, parser NetCDF/ZIP e substituição somente das lacunas INMET.
- Complemento histórico ARCO oficial concluído: 654 pontos-safra processados,
  645 preenchimentos e nove indisponibilidades explícitas, sem erros pendentes.
  Os 645 preenchimentos têm cobertura sazonal completa de chuva/temperatura.
  Gold combinado: 44.551/44.560 registros adequados; nenhum INMET adequado alterado.
- Subconjuntos horários imutáveis, manifesto e SHA-256 dos 654 pontos preservados;
  soma diária de incrementos assinados, contagem de negativos e reprocessamento
  local sem novos downloads. Ausência de chunk Zarr é distinguida de falha HTTP.
- Dataset reconstruído com 14.743 linhas INMET e 254 ERA5, sem lacuna climática
  nas features (distância à estação nula na reanálise é intencional).
- Retreino dos três modelos e comparação com baselines também por fonte climática.
- API, auditoria e Streamlit distinguem estação e reanálise e mostram os períodos
  efetivos. `TEMPORAL_MISMATCH` explicita cenários ainda não alinhados.
- Validação desta etapa: 142 testes aprovados; Ruff, MyPy (63 arquivos), dependências
  e configuração sem erros. Smoke HTTP real e Streamlit AppTest com Sorriso,
  Carlinda e Guarantã do Norte passaram; este último usa ERA5 2024 de verdade.
  Respostas e auditorias persistidas foram conferidas, sem falso alerta de fonte
  climática ausente no treino. Servidor temporário encerrado após o teste.
- Auditoria de proveniência validou 654 manifestos de recortes, quatro metadados
  Zarr persistidos, 76 artefatos de coordenadas e 18 eventos de chunk ausente.
  Os 2.088 blocos remotos não retidos foram validados pelo manifesto; seus bytes
  não foram baixados outra vez nem apresentados como revalidados localmente.
  Há um aviso de depreciação de terceiro no TestClient Starlette, sem falha de teste.

## Resultado do modelo

- Melhor baseline temporal/geográfico: média móvel de 5 anos, MAE
  653,72/690,27 kg/ha.
- LightGBM retreinado: MAE temporal 482,84 e geográfico 306,27 kg/ha.
- Versão promovida: `soja-lightgbm-20260925T235302Z-662c0726`.
- No subconjunto ERA5, MAE temporal/geográfico 292,19/173,01 kg/ha contra
  685,27/576,58 do melhor baseline, em 187/53 linhas. Diagnóstico limitado pela
  amostra pequena; R² temporal ERA5 negativo, não valida todos os domínios.
- Intervalo nominal 80%: cobertura empírica 75,08% em 2024; o desvio está
  documentado e não é apresentado como cobertura perfeita.

## Próximo

- Alinhar horizonte de previsão e períodos de custo/preço/ZARC, preservando a
  distinção entre cenário retrospectivo e previsão corrente.
- Monitorar drift e recalibrar o intervalo quando uma nova PAM for incorporada.
- Calibrar a incerteza e dependência de custo/preço no Monte Carlo.

## Problemas conhecidos

- Nove lacunas permanecem: Afuá/PA 2022 (valores nulos na célula consultada) e
  Fernando de Noronha/PE 2018–2025 (chunks ausentes com preenchimento NaN).
  Nenhuma delas integra o dataset de soja atual. Não substituímos a célula por
  outra distante nem transformamos falta de dados em chuva zero.
- `era5_arco_backfill.json` é o relatório da coleta concluída. O plano CDS mensal
  de 2.529 pedidos continua como alternativa, mas não é necessário para os pontos
  já materializados por ARCO. Três resultados antigos CDS expiraram no servidor;
  seus IDs e estados foram preservados, sem novas submissões desnecessárias.
- O cenário atual de Sorriso combina produtividade 2025, clima 2024, custo de
  março/2026, preços de agosto–dezembro/2025 e ZARC 2026/2027. O aviso temporal
  não equivale a uma previsão de safra corrente nem corrige essa limitação.
- Boa Esperança do Norte/MT (`5101837`) existe em Localidades, mas não na malha
  oficial consultada; geometria/centroide ficam nulos com flag explícita.
- Custos cobrem 10 UFs e preços 17 UFs; cobertura insuficiente bloqueia a análise.
- A faixa de custo do Monte Carlo é uma sensibilidade documentada de ±10%.
- Apenas soja está ativa; a licença final do projeto ainda não foi definida.

## Como executar

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[data,api,ui,ml,dev]"
agri config validate
agri ingest all
agri ingest era5-status
agri ingest era5-canary
agri build climate
agri ingest era5-backfill
agri ingest era5-arco --max-new-points 1000 --workers 2
python scripts/verify_climate_backfill.py
agri build ml-dataset
agri train yield --crop soja
agri evaluate yield --crop soja
pytest
python scripts/smoke_soy_stack.py
uvicorn agri_decision.api.main:app --reload
streamlit run app\streamlit_app.py
```

Para execução limitada e retomada, consulte [ERA5_BACKFILL.md](ERA5_BACKFILL.md).
