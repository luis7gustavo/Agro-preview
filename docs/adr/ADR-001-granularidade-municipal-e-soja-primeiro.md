# ADR-001 - Granularidade municipal e soja primeiro

## Status

Aceita em 2026-08-20.

## Contexto

O produto pretende comparar culturas em todo o Brasil, mas dados e modelos têm
coberturas diferentes. Implementar seis culturas em paralelo aumentaria o risco de
arquitetura ampla e fluxos incompletos.

## Decisão

Usar `codigo_ibge` como chave territorial e a granularidade lógica
`municipio x cultura x safra/ano x sistema_produtivo`. Concluir um vertical de soja
antes de habilitar as demais culturas.

## Consequências

A taxonomia já conhece seis culturas, porém somente soja inicia como `active` e
`pilot`. Não serão feitos joins territoriais apenas por nome de município.

