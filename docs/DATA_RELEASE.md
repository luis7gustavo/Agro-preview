# Snapshot de dados para restauração

O código permanece no histórico Git. Os dados processados e artefatos executáveis
ficam na Release `backup-2026-09-26`, evitando incorporar binários ao histórico.

## Conteúdo publicado

- `data/silver`: dados validados e transformados;
- `data/gold`: tabelas analíticas consumidas pela API, UI e treino;
- `models`: modelo promovido e banco local do MLflow;
- `mlruns`: artefatos locais dos experimentos;
- `reports/generated`: relatórios de qualidade e avaliação executados.

O modelo promovido no snapshot é
`soja-lightgbm-20260925T235302Z-662c0726`.

## Conteúdo deliberadamente excluído

- `data/bronze`: 1,5 GiB de arquivos brutos obtidos de fontes públicas;
- `.venv`, caches e arquivos temporários;
- `.cdsapirc`, tokens, senhas e outros segredos;
- arquivos pessoais fora da pasta do projeto.

Bronze pode ser reconstruído pelos comandos de ingestão documentados no README.
O pacote processado é suficiente para executar a API, Streamlit, auditoria e o
modelo atual sem baixar novamente todo o histórico bruto.

## Integridade

A Release contém o arquivo `.tar.gz`, seu `.sha256` e um manifesto JSON. O script
`scripts/restore_release_data.ps1` confere SHA-256 antes da extração e aborta se o
download estiver incompleto ou alterado.

Dados de terceiros mantêm suas fontes, natureza e checksums nos manifestos do
projeto. Publicar este snapshot não muda as licenças das fontes nem concede uma
licença para o código; a licença geral do repositório continua pendente.
