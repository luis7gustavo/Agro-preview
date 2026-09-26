from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from agri_decision.schemas.crop import Crop
from agri_decision.schemas.location import Location
from agri_decision.taxonomy import CropTaxonomy


def build_dim_crop(taxonomy: CropTaxonomy) -> tuple[Crop, ...]:
    """Materialize the validated crop dimension from versioned configuration."""

    return taxonomy.crops


def validate_dim_locations(records: Iterable[Mapping[str, Any]]) -> tuple[Location, ...]:
    """Validate real municipal records without inventing missing attributes."""

    locations = tuple(Location.model_validate(record) for record in records)
    codes = [location.codigo_ibge for location in locations]
    duplicates = sorted({code for code in codes if codes.count(code) > 1})
    if duplicates:
        raise ValueError(f"Duplicate codigo_ibge values: {', '.join(duplicates)}")
    return locations
