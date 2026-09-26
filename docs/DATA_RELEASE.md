# Snapshot de dados para restauração

O código permanece no histórico Git. Os dados processados e artefatos executáveis
ficam na Release `backup-2026-09-26`, evitando incorporar binários ao histórico.

## Conteúdo publicado

- `data/silver`: dados validados e transformados;
- `data/gold`: tabelas analíticas consumidas pela API, UI e treino;
- `data/bronze/era5`: 654 recortes horários, estados e metadados necessários
  para auditar a cadeia de proveniência do complemento climático;
- `models`: modelo promovido e banco local do MLflow;
- `mlruns`: artefatos locais dos experimentos;
- `reports/generated`: relatórios de qualidade e avaliação executados.

O modelo promovido no snapshot é
`soja-lightgbm-20260925T235302Z-662c0726`.

## Conteúdo deliberadamente excluído

- Bronze bruto de INMET, IBGE, MAPA e Conab, reconstruível pelas ingestões;
- `.venv`, caches e arquivos temporários;
- `.cdsapirc`, tokens, senhas e outros segredos;
- arquivos pessoais fora da pasta do projeto.

O restante do Bronze pode ser reconstruído pelos comandos de ingestão do README.
O pacote é suficiente para executar API, Streamlit, auditoria climática e o modelo
atual sem baixar novamente o histórico horário ERA5.

## Integridade

A Release contém o arquivo `.tar.gz`, seu `.sha256` e um manifesto JSON. O script
`scripts/restore_release_data.ps1` confere SHA-256 antes da extração e aborta se o
download estiver incompleto ou alterado.

Na cópia destinada à Release, `source_artifact_root` dos manifestos ERA5 é
normalizado para o caminho relativo `data/bronze/era5/arco_raw`. Essa alteração
remove a dependência do diretório do computador de origem; os NetCDFs, registros
de fonte, checksums e artefatos canônicos locais não são modificados.

Dados de terceiros mantêm suas fontes, natureza e checksums nos manifestos do
projeto. Publicar este snapshot não muda as licenças das fontes nem concede uma
licença para o código; a licença geral do repositório continua pendente.
