from __future__ import annotations

import json
from typing import Any

import polars as pl
from shapely.geometry import shape

from agri_decision.schemas.location import Location


def _territorial_context(municipality: dict[str, Any]) -> tuple[str, str]:
    immediate = municipality.get("regiao-imediata") or {}
    intermediate = immediate.get("regiao-intermediaria") or {}
    uf_data = intermediate.get("UF") or {}
    region = uf_data.get("regiao") or {}
    uf = uf_data.get("sigla")
    region_name = region.get("nome")
    if not isinstance(uf, str) or not isinstance(region_name, str):
        raise ValueError(f"Missing UF/region for IBGE municipality {municipality.get('id')}")
    return uf, region_name


def build_location_records(
    municipalities: list[dict[str, Any]],
    state_meshes: dict[str, dict[str, Any]],
) -> tuple[Location, ...]:
    geometries: dict[str, dict[str, Any]] = {}
    for uf, mesh in state_meshes.items():
        for feature in mesh.get("features", []):
            properties = feature.get("properties") or {}
            code = str(properties.get("codarea", ""))
            geometry = feature.get("geometry")
            if not code or not isinstance(geometry, dict):
                raise ValueError(f"Invalid municipal mesh feature for {uf}")
            if code in geometries:
                raise ValueError(f"Duplicate geometry for codigo_ibge={code}")
            geometries[code] = geometry

    records: list[Location] = []
    seen_codes: set[str] = set()
    for municipality in municipalities:
        code = str(municipality["id"])
        if code in seen_codes:
            raise ValueError(f"Duplicate municipality codigo_ibge={code}")
        seen_codes.add(code)
        geometry = geometries.get(code)
        if geometry is None:
            uf, region = _territorial_context(municipality)
            records.append(
                Location(
                    codigo_ibge=code,
                    municipio=str(municipality["nome"]),
                    uf=uf,
                    regiao=region,
                    latitude_centroid=None,
                    longitude_centroid=None,
                    geometry=None,
                    source_reference="IBGE Localidades v1; geometry unavailable in Malhas v3",
                    quality_flags=["MISSING_OFFICIAL_GEOMETRY"],
                )
            )
            continue
        polygon = shape(geometry)
        if polygon.is_empty:
            raise ValueError(f"Empty official geometry for codigo_ibge={code}")
        centroid = polygon.centroid
        uf, region = _territorial_context(municipality)
        records.append(
            Location(
                codigo_ibge=code,
                municipio=str(municipality["nome"]),
                uf=uf,
                regiao=region,
                latitude_centroid=centroid.y,
                longitude_centroid=centroid.x,
                geometry=json.dumps(geometry, separators=(",", ":"), ensure_ascii=False),
                source_reference=(
                    "IBGE Localidades v1 + Malhas v3 (qualidade=minima, intrarregiao=municipio)"
                ),
                quality_flags=[],
            )
        )

    return tuple(sorted(records, key=lambda item: item.codigo_ibge))


def location_coverage_report(
    municipalities: list[dict[str, Any]],
    state_meshes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    municipality_codes = {str(item["id"]) for item in municipalities}
    mesh_codes = {
        str(feature.get("properties", {}).get("codarea", ""))
        for mesh in state_meshes.values()
        for feature in mesh.get("features", [])
    }
    mesh_codes.discard("")
    return {
        "municipalities": len(municipality_codes),
        "geometries": len(mesh_codes),
        "missing_geometry_codes": sorted(municipality_codes - mesh_codes),
        "mesh_only_codes": sorted(mesh_codes - municipality_codes),
    }


def location_records_frame(records: tuple[Location, ...]) -> pl.DataFrame:
    return pl.DataFrame([record.model_dump(mode="json") for record in records]).with_columns(
        pl.col("latitude_centroid").cast(pl.Float64),
        pl.col("longitude_centroid").cast(pl.Float64),
    )
