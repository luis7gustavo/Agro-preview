from __future__ import annotations

import io

import polars as pl

DECENDIO_COLUMNS = [f"dec{number}" for number in range(1, 37)]


def parse_soy_zarc_windows(content: bytes, *, source_checksum: str) -> pl.DataFrame:
    raw = pl.read_csv(
        io.BytesIO(content),
        separator=";",
        encoding="utf8-lossy",
        infer_schema_length=20_000,
        null_values=[""],
    )
    first = raw.columns[0]
    if first != "Nome_cultura":
        raw = raw.rename({first: "Nome_cultura"})
    soy = raw.filter(pl.col("Nome_cultura").str.strip_chars().str.to_uppercase() == "SOJA")
    index = [
        "SafraIni",
        "SafraFin",
        "Cod_Cultura",
        "Cod_Ciclo",
        "Cod_Solo",
        "geocodigo",
        "UF",
        "Cod_Outros_Manejos",
        "Nome_Outros_Manejos",
        "Portaria",
    ]
    long = soy.unpivot(
        on=DECENDIO_COLUMNS,
        index=index,
        variable_name="decendio_label",
        value_name="nivel_risco",
    )
    return (
        long.filter(pl.col("nivel_risco").is_in([20, 30, 40]))
        .select(
            pl.lit("soja").alias("cultura"),
            pl.col("SafraIni").cast(pl.Int32).alias("safra_inicio"),
            pl.col("SafraFin").cast(pl.Int32).alias("safra_fim"),
            pl.col("Cod_Cultura").cast(pl.Utf8).alias("codigo_cultura_zarc"),
            pl.col("Cod_Ciclo").cast(pl.Int16).alias("ciclo_codigo"),
            pl.col("Cod_Solo").cast(pl.Int16).alias("tipo_solo_codigo"),
            pl.col("geocodigo").cast(pl.Utf8).str.zfill(7).alias("codigo_ibge"),
            pl.col("UF").cast(pl.Utf8).str.strip_chars().alias("uf"),
            pl.col("Cod_Outros_Manejos").cast(pl.Int16).alias("manejo_codigo"),
            pl.col("Nome_Outros_Manejos")
            .cast(pl.Utf8)
            .str.strip_chars()
            .alias("manejo_nome_fonte"),
            pl.col("Portaria").cast(pl.Utf8).str.strip_chars().alias("portaria"),
            pl.col("decendio_label")
            .str.strip_prefix("dec")
            .cast(pl.Int8)
            .alias("decendio"),
            pl.col("nivel_risco").cast(pl.Int8),
            pl.lit("eligible").alias("zarc_status"),
            pl.lit("observed").alias("data_nature"),
            pl.lit("MAPA ZARC open risk table 2026/2027").alias("source"),
            pl.lit(source_checksum).alias("source_dataset_checksum"),
        )
        .unique()
        .sort(["codigo_ibge", "ciclo_codigo", "tipo_solo_codigo", "decendio"])
    )


def build_zarc_municipality_coverage(
    windows: pl.DataFrame, locations: pl.DataFrame
) -> pl.DataFrame:
    available_ufs = windows.get_column("uf").unique().to_list()
    eligible = windows.group_by(["codigo_ibge", "cultura", "safra_inicio", "safra_fim"]).agg(
        pl.col("nivel_risco").min().alias("melhor_nivel_risco"),
        pl.col("decendio").min().alias("janela_inicio_decendio"),
        pl.col("decendio").max().alias("janela_fim_decendio"),
        pl.col("decendio").unique().sort().alias("decendios_elegiveis"),
        pl.col("ciclo_codigo").unique().sort().alias("ciclos_codigos"),
        pl.col("tipo_solo_codigo").unique().sort().alias("tipos_solo_codigos"),
        pl.col("portaria").unique().sort().alias("portarias"),
        pl.len().alias("combinacoes_elegiveis"),
        pl.col("source_dataset_checksum").first(),
    )
    base = locations.select("codigo_ibge", "municipio", "uf").with_columns(
        pl.lit("soja").alias("cultura"),
        pl.lit(2026).cast(pl.Int32).alias("safra_inicio"),
        pl.lit(2027).cast(pl.Int32).alias("safra_fim"),
    )
    return (
        base.join(
            eligible,
            on=["codigo_ibge", "cultura", "safra_inicio", "safra_fim"],
            how="left",
        )
        .with_columns(
            pl.when(pl.col("combinacoes_elegiveis").is_not_null())
            .then(pl.lit("eligible"))
            .when(pl.col("uf").is_in(available_ufs))
            .then(pl.lit("not_eligible"))
            .otherwise(pl.lit("not_available"))
            .alias("zarc_status"),
            pl.when(pl.col("combinacoes_elegiveis").is_not_null())
            .then(pl.lit("observed"))
            .otherwise(pl.lit("estimated"))
            .alias("data_nature"),
            pl.when(pl.col("combinacoes_elegiveis").is_not_null())
            .then(pl.lit(None, dtype=pl.Utf8))
            .when(pl.col("uf").is_in(available_ufs))
            .then(pl.lit("absence from complete eligible table for crop/UF/season"))
            .otherwise(pl.lit("crop/UF not covered by current official table"))
            .alias("status_method"),
            pl.lit("MAPA ZARC open risk table 2026/2027").alias("source"),
        )
        .sort("codigo_ibge")
    )
