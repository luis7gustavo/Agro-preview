# Complemento climático ERA5-Land

INMET continua sendo a fonte prioritária. A reanálise ERA5-Land é `estimated`,
nunca observação de uma estação. O preenchimento substitui a linha climática
inteira de um município-safra somente se chuva e temperatura forem finitas e
a cobertura sazonal atingir o mínimo configurado, atualmente 70%.

## Preparação e execução limitada

Use as credenciais CDS já configuradas fora do repositório. Nunca copie chaves
para scripts, logs, commits ou documentação. O titular deve aceitar os termos.

### Caminho preferencial: ARCO oficial

O acesso direto aos blocos Zarr oficiais evita a fila de pedidos mensais CDS.
Os dados continuam sendo ERA5-Land, mas precipitação e radiação chegam como
incrementos horários, exigindo um modo de transformação distinto.

```powershell
# Completa até 1.000 município-safras ainda ausentes, priorizando pontos usados no ML.
agri ingest era5-arco --max-new-points 1000 --workers 2

# Sem novas consultas de pontos: reaproveita subconjuntos locais pendentes.
agri ingest era5-arco

# Reexecuta a transformacao inclusive de pontos ja preenchidos, sem novos downloads.
agri ingest era5-arco --max-new-points 0 --reprocess-existing --workers 2

# Auditoria independente, somente leitura, após a coleta terminar.
python scripts/verify_climate_backfill.py
```

O cliente é limitado ao layout Zarr v2 documentado e rejeita mudanças incompatíveis
de tipo, unidades de tempo, packing e preenchimento. Há no máximo três GETs em
paralelo e cache em memória de 128 MiB para blocos grandes. Metadados e coordenadas
são preservados em `data/bronze/era5/arco_raw`; os recortes horários exatos ficam
em `data/bronze/era5/arco_points/<codigo>/<safra>` com SHA-256 e manifesto.

Esses NetCDFs são contêineres gerados localmente de valores selecionados sem
interpolação, não cópias byte a byte de um NetCDF distribuído pela fonte. Os
manifestos preservam URLs, hashes e tamanhos dos blocos originais; os blocos
grandes são descartados do cache em memória, não retidos integralmente no disco.

O relatório `reports/generated/era5_arco_backfill.json` distingue coleta concluída
de cobertura completa. Dados ausentes na grade permanecem nulos; não há mudança
automática para uma célula distante. `available`/`unavailable` são resultados de
processamento; erros de autenticação, rede ou parsing são `error`, não ausência
climática. No Zarr v2, um chunk de dados inexistente (HTTP 404, dentro dos limites
do array validado) representa o `fill_value` declarado, aqui NaN. Esse evento é
preservado no manifesto sem inventar bytes nem hash do chunk ausente. Metadados
ou coordenadas ausentes continuam sendo erro. Referência: [especificação Zarr v2](https://zarr-specs.readthedocs.io/en/latest/v2/v2.0.html#chunks).

### Alternativa: pedidos mensais CDS

```powershell
# Recalcula Gold a partir do Silver INMET com checksum verificado, sem baixar de novo.
agri build climate

# Planeja todo o histórico; não submete pedidos novos por padrão.
# Pedidos previamente iniciados são consultados e baixados se estiverem prontos.
agri ingest era5-backfill

# Exemplo de piloto: até 9 novos pedidos, no máximo 3 ativos, espera limitada a 5 min.
agri ingest era5-backfill --season-year 2025 --codigo-ibge 5102793 --max-new-requests 9 --wait-seconds 300

# Retoma os mesmos IDs. Resultados concluídos são reaproveitados.
agri ingest era5-backfill --season-year 2025 --codigo-ibge 5102793 --max-new-requests 9 --wait-seconds 300
```

`--max-new-requests` vale para a execução inteira, não para cada passagem do loop.
O prazo encerra novas passagens; uma operação HTTP já iniciada pode terminar depois
dele, sujeita ao timeout de rede. Não há tarefa recorrente instalada. Interromper
o comando preserva os pedidos remotos; outra execução retoma o estado local.

O plano usa blocos de 2 graus com margem de uma célula, setembro–abril e um pedido
adicional de acumulados à 00 UTC de 1º de maio. Todos os nove pedidos do bloco
precisam estar baixados antes da agregação. O filtro municipal reaproveita os
mesmos downloads do bloco: executar depois um escopo maior pode beneficiar outros
municípios sem repetir a requisição. A saída representa o centroide, não a média
de todas as propriedades ou de toda a área municipal.

## Proveniência e segurança operacional

- `data/bronze/era5/jobs/<hash>/state.json`: pedido determinístico, ID remoto e estado.
- Bronze imutável: NetCDF/ZIP original, SHA-256, dataset, licença e pedido exato.
- `data/silver/era5/`: valores diários UTC, grade, distância e checksums de origem.
- `data/gold/inmet_location_season.parquet`: referência independente INMET.
- `data/gold/era5_location_season.parquet`: registros município-safra de reanálise.
- `data/gold/climate_location_season.parquet`: união priorizando INMET adequado.
- `reports/generated/era5_backfill.json`: escopo, estados e lacunas restantes.

Há exclusão mútua entre execuções do backfill. Não execute simultaneamente
`agri build climate` ou ingestão INMET, que também escrevem o Gold combinado.
Não há repetição automática de POST: `submission_unknown` representa interrupção
ou resposta ambígua. Conferir o pedido no CDS antes de reconciliar o ID; apagar o
estado sem conferir pode duplicar trabalho. Pedidos `failed`/`rejected` também
exigem diagnóstico, não são submetidos novamente de forma silenciosa.

`downloads_complete` indica apenas a chegada dos arquivos. `complete_for_selected_scope`
exige cobertura climática efetiva: grade oceânica ou dados nulos não viram zeros.
Um checksum divergente bloqueia a materialização. Nunca editar artefatos Bronze.

## Semântica física

Temperatura Kelvin é convertida para Celsius; precipitação de metros para mm;
radiação de J/m² para kJ/m². Umidade é derivada do ponto de orvalho e vento dos
componentes u/v. No dataset **hourly reanalysis-era5-land**, precipitação e radiação
à 00 UTC são os totais do dia anterior. Não somar os acumulados de todas as horas.
Essa convenção difere do produto separado timeseries, já desacumulado.
Referência: [documentação oficial ERA5-Land](https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation).

No ARCO, o incremento corresponde à hora que termina no timestamp: o total do dia
é a soma de 01 UTC até 00 UTC do dia seguinte. Exigimos 24 incrementos válidos;
um dia incompleto não vira chuva zero nem é contado como dia de cobertura total.
O recorte inclui 1º de maio à 00 UTC para completar 30 de abril. Temperaturas,
umidade e vento continuam agregados pelo horário instantâneo do dia UTC.
Referência: [guia oficial ARCO ERA5-Land](https://confluence.ecmwf.int/plugins/viewsource/viewpagesrc.action?pageId=536218894).

Os incrementos são somados com sinal e precisão float64 antes da validação física
do total diário. Pequenos negativos podem resultar do packing GRIB: cortar cada
hora em zero introduziria viés positivo e descartá-la criaria lacunas artificiais.
`rain_negative_increments` e `radiation_negative_increments` registram a contagem
por dia e safra. Os valores Bronze nunca são alterados. Totais diários negativos
além de 0,00001 mm ou 0,000001 kJ/m² são retidos como nulos; apenas resíduos
diários dentro dessas tolerâncias numéricas são limitados a zero.
Referência: [ECMWF sobre negativos após desacumulação](https://confluence.ecmwf.int/spaces/UDOC/pages/208501579/Why%2Bare%2Bthere%2Bsometimes%2Bsmall%2Bnegative%2Bprecipitation%2Baccumulations%2B-%2BecCodes%2BGRIB%2BFAQ).

Máxima/mínima ERA5 são calculadas das amostras horárias e não são idênticas aos
extremos instrumentais INMET. Há mudança de distribuição entre fontes; treinar,
comparar baselines e avaliar temporal/geograficamente antes de validar uso amplo.
Após novos dados relevantes ao treino: `agri build ml-dataset` e `agri train yield`.
