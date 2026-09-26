from __future__ import annotations

import io

import polars as pl


def _read_export(content: bytes) -> pl.DataFrame:
    # The current exports contain bytes outside strict UTF-8 and use fixed-width
    # text values. Decoding is explicit so the raw Bronze artifact remains intact.
    utf8 = content.decode("cp1252").encode("utf-8")
    return pl.read_csv(
        io.BytesIO(utf8),
        separator=";",
        infer_schema_length=20_000,
        null_values=["", "null", "NULL"],
        truncate_ragged_lines=False,
    )


def _text(column: str) -> pl.Expr:
    return pl.col(column).cast(pl.Utf8).str.strip_chars()


def _number(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.Utf8)
        .str.strip_chars()
        .str.replace_all(",", ".")
        .cast(pl.Float64, strict=False)
    )


def parse_soy_costs(content: bytes, *, source_checksum: str) -> pl.DataFrame:
    raw = _read_export(content)
    soy = raw.filter(_text("produto").str.to_uppercase() == "SOJA")
    frame = soy.select(
        _text("empreendimento").alias("sistema_produtivo"),
        pl.lit("soja").alias("cultura"),
        pl.col("ano").cast(pl.Int32),
        pl.col("mes").cast(pl.Int8),
        _text("ano_mes").alias("ano_mes"),
        _text("safra").alias("safra"),
        _text("uf").alias("uf"),
        _text("municipio").alias("municipio_fonte"),
        _text("cod_ibge").cast(pl.Int64, strict=False).alias("codigo_ibge_fonte"),
        _text("unidade_comercializacao").alias("unidade_comercializacao"),
        _number("vlr_custo_variavel_ha").alias("custo_variavel_brl_ha"),
        _number("vlr_custo_fixo_ha").alias("custo_fixo_brl_ha"),
        _number("vlr_renda_fator_ha").alias("renda_fatores_brl_ha"),
        _number("vlr_custo_variavel_unidade").alias("custo_variavel_brl_unidade"),
        _number("vlr_custo_fixo_unidade").alias("custo_fixo_brl_unidade"),
        _number("vlr_renda_fator_unidade").alias("renda_fatores_brl_unidade"),
    ).with_columns(
        (pl.col("custo_variavel_brl_ha") + pl.col("custo_fixo_brl_ha"))
        .alias("custo_operacional_brl_ha"),
        (
            pl.col("custo_variavel_brl_ha")
            + pl.col("custo_fixo_brl_ha")
            + pl.col("renda_fatores_brl_ha")
        ).alias("custo_total_brl_ha"),
        pl.lit("observed").alias("data_nature"),
        pl.lit("CONAB CustoProducao.txt").alias("source"),
        pl.lit(source_checksum).alias("source_dataset_checksum"),
        pl.lit(None, dtype=pl.Utf8).alias("estimation_method"),
        pl.lit(1.0).alias("confidence"),
    )
    return frame.filter(
        pl.col("codigo_ibge_fonte").is_not_null()
        & pl.col("custo_total_brl_ha").is_not_null()
        & (pl.col("custo_total_brl_ha") > 0)
    ).sort(["codigo_ibge_fonte", "ano", "mes", "sistema_produtivo"])


def parse_soy_prices(
    content: bytes,
    *,
    coverage_level: str,
    source_checksum: str,
) -> pl.DataFrame:
    raw = _read_export(content)
    soy = raw.filter(_text("produto").str.to_uppercase() == "SOJA")
    expressions: list[pl.Expr] = [
        pl.lit("soja").alias("cultura"),
        pl.col("ano").cast(pl.Int32),
        pl.col("mes").cast(pl.Int8),
        _text("uf").alias("uf"),
        _text("dsc_nivel_comercializacao").alias("nivel_comercializacao"),
        _number("valor_produto_kg").alias("preco_brl_kg"),
        pl.lit(coverage_level).alias("coverage_level"),
        pl.lit("observed").alias("data_nature"),
        pl.lit("CONAB precos mensais").alias("source"),
        pl.lit(source_checksum).alias("source_dataset_checksum"),
    ]
    if coverage_level == "municipality":
        expressions.extend(
            [
                _text("nom_municipio").alias("municipio_fonte"),
                _text("cod_ibge").cast(pl.Int64, strict=False).alias("codigo_ibge_fonte"),
            ]
        )
    else:
        expressions.extend(
            [
                pl.lit(None, dtype=pl.Utf8).alias("municipio_fonte"),
                pl.lit(None, dtype=pl.Int64).alias("codigo_ibge_fonte"),
            ]
        )
    frame = soy.select(expressions)
    return frame.filter(
        pl.col("preco_brl_kg").is_not_null() & (pl.col("preco_brl_kg") > 0)
    ).sort(["uf", "codigo_ibge_fonte", "ano", "mes"])
