from __future__ import annotations

from typing import Any

import polars as pl


def production_quality_report(frame: pl.DataFrame) -> dict[str, Any]:
    key = ["codigo_ibge", "cultura", "ano", "sistema_produtivo"]
    duplicates = frame.group_by(key).len().filter(pl.col("len") > 1).height
    missing = {
        column: frame.select(pl.col(column).is_null().sum()).item()
        for column in [
            "area_plantada_ha",
            "area_colhida_ha",
            "producao_t",
            "produtividade_kg_ha",
            "valor_producao_original",
            "valor_producao_brl",
        ]
    }
    flagged_rows = frame.filter(pl.col("quality_flags").list.len() > 0).height
    return {
        "rows": frame.height,
        "columns": frame.width,
        "duplicate_keys": duplicates,
        "year_min": frame.select(pl.col("ano").min()).item(),
        "year_max": frame.select(pl.col("ano").max()).item(),
        "municipalities": frame.select(pl.col("codigo_ibge").n_unique()).item(),
        "crops": sorted(frame.get_column("cultura").unique().to_list()),
        "missing_by_field": missing,
        "flagged_rows": flagged_rows,
    }
