# ADR-003 - Proveniência como contrato

## Status

Aceita em 2026-08-20.

## Contexto

Custos, clima e produtividade podem ser observados, interpolados, estimados ou
preditos. O usuário precisa saber de onde veio cada valor crítico.

## Decisão

Representar explicitamente a natureza do dado. Valores derivados exigem método;
artefatos Bronze exigem fonte, URL, período, instante de download, checksum, arquivo
e versão do pipeline. Gold deve rejeitar proveniência incompleta.

## Consequências

Ausência de dado continuará ausente ou indisponível. Nenhum adapter poderá preencher
lacunas com números inventados para manter um fluxo aparentemente completo.

