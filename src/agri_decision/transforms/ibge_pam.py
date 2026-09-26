from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import polars as pl

from agri_decision.schemas.production import ProductionHistoryRecord
from agri_decision.taxonomy import CropTaxonomy

VARIABLE_FIELDS = {
    "8331": ("area_plantada_ha", "Hectares", 1.0),
    "216": ("area_colhida_ha", "Hectares", 1.0),
    "214": ("producao_t", "Toneladas", 1.0),
    "112": ("produtividade_kg_ha", "Quilogramas por Hectare", 1.0),
    "215": ("valor_producao_original", None, 1.0),
}

PAM_VALUE_UNITS = {
    "Mil Cruzeiros",
    "Mil Cruzados",
    "Mil Cruzados Novos",
    "Mil Cruzeiros Reais",
    "Mil Reais",
}

SPECIAL_VALUE_FLAGS = {
    "-": "SIDRA_ABSOLUTE_ZERO",
    "0": "SIDRA_ROUNDED_ZERO",
    "X": "SIDRA_SUPPRESSED",
    "..": "SIDRA_NOT_APPLICABLE",
    "...": "SIDRA_NOT_AVAILABLE",
}


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    value: float | None
    flag: str | None = None


def parse_sidra_number(raw: str) -> ParsedNumber:
    stripped = raw.strip()
    if stripped in {"-", "0"}:
        return ParsedNumber(0.0, SPECIAL_VALUE_FLAGS[stripped])
    if stripped in SPECIAL_VALUE_FLAGS:
        return ParsedNumber(None, SPECIAL_VALUE_FLAGS[stripped])
    if len(stripped) == 1 and stripped.isalpha():
        return ParsedNumber(None, "SIDRA_VALUE_RANGE_OR_SUPPRESSED")
    try:
        return ParsedNumber(float(Decimal(stripped)))
    except InvalidOperation as exc:
        raise ValueError(f"Unknown SIDRA numeric token: {raw!r}") from exc


def _municipality_and_uf(label: str) -> tuple[str, str]:
    match = re.fullmatch(r"(.+?)(?: \(([A-Z]{2})\)| - ([A-Z]{2}))", label.strip())
    if not match:
        raise ValueError(f"Unexpected SIDRA municipality label: {label!r}")
    return match.group(1), match.group(2) or match.group(3)


def parse_pam_rows(
    rows: list[dict[str, str]],
    *,
    taxonomy: CropTaxonomy,
    source_checksum: str,
) -> tuple[ProductionHistoryRecord, ...]:
    grouped: dict[tuple[str, int, str], dict[str, Any]] = defaultdict(lambda: {"quality_flags": []})
    seen_variables: set[tuple[str, int, str, str]] = set()

    for row in rows[1:]:
        codigo_ibge = str(row["D1C"])
        year = int(row["D3C"])
        crop = taxonomy.resolve(str(row["D4N"]), source="ibge")
        variable_id = str(row["D2C"])
        if variable_id not in VARIABLE_FIELDS:
            raise ValueError(f"Unexpected PAM variable {variable_id}: {row.get('D2N')}")
        seen_key = (codigo_ibge, year, crop, variable_id)
        if seen_key in seen_variables:
            raise ValueError(f"Duplicate PAM variable row: {seen_key}")
        seen_variables.add(seen_key)

        field, expected_unit, multiplier = VARIABLE_FIELDS[variable_id]
        unit = str(row["MN"])
        parsed = parse_sidra_number(str(row["V"]))
        if variable_id == "215":
            if unit and unit not in PAM_VALUE_UNITS:
                raise ValueError(f"Unexpected historical PAM value currency: {unit!r}")
        elif unit != expected_unit and not (parsed.value is None and not unit):
            message = (
                f"Unexpected unit for PAM variable {variable_id}: "
                f"{unit!r}, expected {expected_unit!r}"
            )
            raise ValueError(message)
        municipality, uf = _municipality_and_uf(str(row["D1N"]))
        key = (codigo_ibge, year, crop)
        record = grouped[key]
        record.update(
            {
                "codigo_ibge": codigo_ibge,
                "municipio": municipality,
                "uf": uf,
                "ano": year,
                "cultura": crop,
                "source_reference": (f"SIDRA:t5457:n6:{codigo_ibge}:p{year}:c782:{row['D4C']}"),
                "source_dataset_checksum": source_checksum,
            }
        )
        record[field] = parsed.value * multiplier if parsed.value is not None else None
        if variable_id == "215":
            record["valor_producao_unidade_original"] = unit or None
            if parsed.value is not None and unit == "Mil Reais":
                record["valor_producao_brl"] = parsed.value * 1000.0
            else:
                record["valor_producao_brl"] = None
                if parsed.value is not None:
                    record["quality_flags"].append(
                        "valor_producao_brl:HISTORICAL_CURRENCY_NOT_NORMALIZED"
                    )
        elif parsed.value is None and not unit:
            record["quality_flags"].append(f"{field}:UNIT_NOT_PROVIDED_FOR_MISSING_VALUE")
        if parsed.flag:
            record["quality_flags"].append(f"{field}:{parsed.flag}")

    parsed_records: list[ProductionHistoryRecord] = []
    expected_fields = {item[0] for item in VARIABLE_FIELDS.values()}
    for record in grouped.values():
        for field in expected_fields:
            if field not in record:
                record[field] = None
                record["quality_flags"].append(f"{field}:MISSING_VARIABLE")
        if (
            record["area_plantada_ha"] is not None
            and record["area_colhida_ha"] is not None
            and record["area_colhida_ha"] > record["area_plantada_ha"]
        ):
            record["quality_flags"].append("AREA_HARVESTED_GT_PLANTED")
        record["quality_flags"] = sorted(set(record["quality_flags"]))
        parsed_records.append(ProductionHistoryRecord.model_validate(record))

    return tuple(
        sorted(parsed_records, key=lambda item: (item.codigo_ibge, item.cultura, item.ano))
    )


def production_records_frame(records: tuple[ProductionHistoryRecord, ...]) -> pl.DataFrame:
    return pl.DataFrame([record.model_dump(mode="json") for record in records]).with_columns(
        pl.col("ano").cast(pl.Int32),
        pl.col("area_plantada_ha").cast(pl.Float64),
        pl.col("area_colhida_ha").cast(pl.Float64),
        pl.col("producao_t").cast(pl.Float64),
        pl.col("produtividade_kg_ha").cast(pl.Float64),
        pl.col("valor_producao_original").cast(pl.Float64),
        pl.col("valor_producao_brl").cast(pl.Float64),
    )
