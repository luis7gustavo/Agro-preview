# ADR-002 - Monólito modular e camadas de dados

## Status

Aceita em 2026-08-20.

## Contexto

O V1 precisa de ingestão, engenharia de features, modelos, economia, risco, API e
interface, mas ainda não há volume ou operação que justifique serviços distribuídos.

## Decisão

Adotar um pacote Python modular. Preservar dados em Bronze imutável, normalizar em
Silver e materializar tabelas analíticas Gold. Usar Parquet localmente e adicionar
PostgreSQL/PostGIS onde consultas espaciais ou estado transacional exigirem.

## Consequências

Kafka, Kubernetes e microserviços ficam fora do V1. Limites internos são mantidos
por contratos, testes e dependências direcionais entre módulos.

