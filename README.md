# Motor de Aptidão, Rentabilidade e Risco Agrícola do Brasil

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Status](https://img.shields.io/badge/status-MVP%20soja-brightgreen)
![ML](https://img.shields.io/badge/ML-LightGBM-orange)

Projeto de software para responder, de forma rastreável:

> Dado um município brasileiro, quais culturas apresentam a melhor combinação de
> viabilidade agronômica, produtividade esperada, custo, retorno, tempo e risco?

O sistema é apoio à decisão. Não substitui análise agronômica de campo, análise de
solo ou assistência técnica local.

## Estado atual

O primeiro caminho analítico de **soja** está implementado ponta a ponta. O
projeto ingere municípios/malhas e PAM do IBGE, custos/preços da Conab, ZARC do
MAPA e estações do INMET; treina e avalia modelos de produtividade, calcula
cenários econômicos e risco, persiste a auditoria e os expõe em FastAPI/Streamlit.
As demais culturas continuam apenas na taxonomia, marcadas como `planned`.

Nenhum número agrícola fictício é usado pela aplicação. Dados sintéticos são
permitidos somente em `tests/fixtures` e devem ser identificados como testes.

Consulte [docs/STATUS.md](docs/STATUS.md) para o que está concluído e o próximo passo.

## Arquitetura

O fluxo-alvo é:

```text
fontes oficiais -> Bronze imutável -> Silver validado -> Gold analítico
                                                       |
       produtividade prevista + clima + custo + preço + ZARC + risco
                                                       |
                           ranking -> FastAPI -> Streamlit -> auditoria
```

Granularidade principal:

```text
municipio x cultura x safra/ano x sistema_produtivo
```

Responsabilidades são separadas em módulos pequenos sob `src/agri_decision`.
Parquet é o formato analítico local; PostgreSQL/PostGIS será usado para estado
transacional e consultas espaciais quando necessário.

## Proveniência obrigatória

Todo valor relevante declara uma das naturezas:

- `observed`: medido/coletado diretamente na fonte declarada;
- `interpolated`: derivado espacial ou temporalmente por método explícito;
- `estimated`: imputado/extrapolado por método explícito;
- `predicted`: saída de modelo versionado.

Valores derivados exigem `method`. Bronze registra URL, instante de download,
período de referência, checksum, arquivo e versão do pipeline.

## Fontes

- IBGE Localidades/Malhas: dimensão territorial, integrado;
- IBGE/PAM/SIDRA tabela 5457: produção municipal 1974–2024, integrado;
- Conab Portal de Informações: custos e preços, integrado;
- MAPA/ZARC 2026/2027: elegibilidade e janelas de plantio, integrado;
- INMET: clima observado agregado por safra, integrado;
- ERA5-Land ARCO: complemento histórico executado, 645 lacunas preenchidas e
  nove indisponibilidades explícitas; INMET adequado permanece prioritário;
- Embrapa/GeoInfo/PronaSolos e MapBiomas: enriquecimentos posteriores.

Os recursos verificados, períodos e URLs efetivamente baixados estão versionados
em `configs/data_sources.yml`. Fontes ainda não integradas permanecem como pontos
de descoberta e não são tratadas como dados disponíveis.

## Instalação

Requer Python 3.11 ou superior; o ambiente validado usa Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,api,ui,ml,dev]"
```

Para restaurar o snapshot processado publicado no GitHub depois de clonar:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\restore_release_data.ps1
```

O pacote da Release inclui Silver, Gold, modelo promovido, MLflow e relatórios
gerados. Bronze bruto, caches e credenciais não são publicados. Consulte
[docs/REINSTALL_WINDOWS.md](docs/REINSTALL_WINDOWS.md) para a reinstalação completa.

## Comandos disponíveis

```powershell
agri version
agri config validate
agri crops list
agri crops list --all
agri crops resolve "Soja (em grão)" --source ibge
agri dimensions crop --output data\gold\dim_crop.json
agri ingest ibge-locations
agri ingest pam --start-year 1974 --end-year 2024
agri ingest conab
agri ingest zarc
agri ingest inmet --start-year 2017 --end-year 2025
agri ingest era5-status
agri ingest era5-canary
agri build climate
agri ingest era5-backfill
agri ingest era5-arco --max-new-points 1000 --workers 2
python scripts/verify_climate_backfill.py
agri build ml-dataset
agri train yield --crop soja
agri evaluate yield --crop soja
agri ingest all
```

Os comandos de ingestão usam cache Bronze por padrão; use `--refresh` somente
quando quiser consultar novamente a fonte e preservar uma nova versão caso o
conteúdo tenha mudado.

## Testes e qualidade

```powershell
pytest
ruff check .
mypy src
```

Os testes cobrem contratos, parsers IBGE/Conab/INMET/ERA5, ausência de leakage,
features de inferência, métricas de ML, economia, Monte Carlo, ranking, CLI e API.

## Dados e pipelines

Diretórios `data/bronze`, `data/silver`, `data/gold` e `data/external` não são
versionados. Os pipelines implementados:

1. preservar o arquivo bruto;
2. gerar checksum e manifesto;
3. validar schema, chaves, unidades e faixas plausíveis;
4. separar linhas rejeitadas sem removê-las silenciosamente;
5. produzir Silver/Gold idempotentes;
6. registrar versão da fonte, dataset, configuração e pipeline.

## API

```powershell
uvicorn agri_decision.api.main:app --host 127.0.0.1 --port 8000 --reload
```

OpenAPI: `http://127.0.0.1:8000/docs`. Estão disponíveis `/health`, `/version`,
pesquisa de municípios, culturas, recomendação, recuperação/auditoria da análise
e cobertura dos datasets.

Exemplo:

```powershell
$body = @{
  codigo_ibge = "5107925"
  area_ha = 500
  production_system = "rainfed"
  profile = "balanced"
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/recommendations `
  -ContentType application/json -Body $body
```

## Streamlit

Com a API ativa:

```powershell
streamlit run app\streamlit_app.py
```

A UI possui busca municipal, área, perfil, resumo econômico/risco, explicações,
alertas e auditoria. `AGRI_API_URL` permite apontar para outra URL da API.

## Treino e ML

O dataset de soja usa somente informações anteriores ao ano alvo: histórico
deslocado, mediana estadual de `Y-1` e clima da safra concluída `Y-1`. O treino
compara mediana histórica municipal, média móvel de cinco anos e mediana regional
contra CatBoost, LightGBM e XGBoost em splits temporal e geográfico.

O gate promove um modelo apenas quando seu MAE fica abaixo do melhor baseline nos
dois splits. Na execução validada em 25/09/2026, LightGBM foi promovido após
retreino com 254 linhas ERA5 e 14.743 INMET; os
intervalos P10/P50/P90 usam resíduos absolutos de split conformal. Metadados e
artefatos ficam em `models/yield/soja`, e experimentos/params/métricas são
registrados no SQLite local do MLflow em `models/mlflow.db`.

Consulte [reports/model-evaluation.md](reports/model-evaluation.md) para métricas,
gate, cobertura e limitações.

## Limitações atuais

- o backfill deixou nove município-safras sem clima válido na grade consultada:
  Afuá/PA 2022 e Fernando de Noronha/PE 2018–2025; nenhuma dessas lacunas afeta
  o dataset de soja atual. Isso não representa cobertura de campo de todo o país;
- a cobertura empírica do intervalo de 80% foi 75,08% em 2024 e deve continuar
  sendo monitorada/recalibrada;
- as respostas mostram `temporal_context` e `TEMPORAL_MISMATCH`: produtividade,
  clima, preços, custos e ZARC ainda podem representar períodos diferentes;
- cobertura Conab é limitada; ausência de custo/preço suficiente impede a análise;
- o custo do Monte Carlo usa sensibilidade provisória de ±10%;
- apenas soja está ativa e o ranking ainda não compara culturas;
- a licença final do projeto e das redistribuições de dados precisa ser decidida.

## Roadmap resumido

1. alinhar o horizonte da produtividade com os períodos de custos, preços e ZARC;
   o alerta temporal atual apenas informa o desalinhamento;
2. recalibrar intervalos com novas safras e monitorar drift;
3. calibrar dependências e incerteza de custo no Monte Carlo;
4. decidir a licença do código e dos artefatos redistribuíveis;
5. somente então generalizar para as outras culturas.

## Autor

Luis Gustavo  
<https://github.com/luis7gustavo>
