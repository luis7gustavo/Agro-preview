# ADR-004 - Fontes por adapter e fallback manual

## Status

Aceita em 2026-08-20.

## Contexto

Portais públicos mudam formatos e alguns dados só são publicados como planilhas ou
arquivos que exigem intervenção humana.

## Decisão

Cada fonte terá adapter próprio. Antes de automatizar, o adapter documentará fonte
oficial, formato, colunas, periodicidade e licença. Se a automação não for estável,
será oferecida ingestão manual validada do arquivo oficial.

## Consequências

O sistema não inventará endpoints e não usará scraping frágil quando houver API ou
arquivo oficial. Mudança de schema deve falhar explicitamente e preservar o Bronze.

