# Reinstalação no Windows após formatação

## Pré-requisitos

Instale Git, GitHub CLI e Python 3.12 de 64 bits. No PowerShell:

```powershell
git clone https://github.com/luis7gustavo/Agro-preview.git
cd Agro-preview
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[data,api,ui,ml,dev]"
powershell -ExecutionPolicy Bypass -File scripts\restore_release_data.ps1
agri config validate
python scripts\verify_climate_backfill.py
pytest -q
python scripts\smoke_soy_stack.py
```

Se a política do PowerShell impedir a ativação, execute uma vez na sessão:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Credencial climática

A credencial não está no GitHub. Para futuras consultas ERA5/CDS, entre novamente
em <https://cds.climate.copernicus.eu>, gere sua chave pessoal e recrie
`%USERPROFILE%\.cdsapirc` conforme as instruções da conta. Nunca cole a chave em
um commit, issue, prompt público ou arquivo dentro deste repositório.

O snapshot restaurado funciona sem essa credencial. Ela só é necessária para
novas consultas à fonte climática.

## Executar a aplicação

Em dois terminais com o ambiente ativado:

```powershell
uvicorn agri_decision.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
$env:AGRI_API_URL = "http://127.0.0.1:8000"
streamlit run app\streamlit_app.py
```

## Prompt para usar após a formatação

Copie e envie o texto abaixo ao Codex dentro de uma pasta onde deseja restaurar
o projeto:

> Clone `https://github.com/luis7gustavo/Agro-preview`, entre na pasta do projeto
> e restaure o ambiente Windows usando Python 3.12. Crie `.venv`, instale
> `.[data,api,ui,ml,dev]`, execute `scripts/restore_release_data.ps1` para baixar
> e verificar o snapshot da Release `backup-2026-09-26`, e então rode
> `agri config validate`, `scripts/verify_climate_backfill.py`, `pytest -q` e
> `scripts/smoke_soy_stack.py`. Não invente dados, não altere os artefatos Bronze,
> não imprima nem grave credenciais. Se o ERA5 precisar de nova coleta, peça que
> eu recrie `%USERPROFILE%\.cdsapirc`; o segredo nunca esteve no repositório.
> Ao final, informe a versão do modelo carregada, contagens de fontes climáticas,
> testes aprovados e qualquer divergência do snapshot publicado.
