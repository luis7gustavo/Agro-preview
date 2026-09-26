from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field

from agri_decision.config import find_project_root, load_yaml
from agri_decision.schemas.crop import Crop

IndexKeyT = TypeVar("IndexKeyT", str, tuple[str, str])


def normalize_crop_label(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    alphanumeric = re.sub(r"[^a-z0-9]+", " ", ascii_text.casefold())
    return " ".join(alphanumeric.split())


class CropTaxonomyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    config_version: str = Field(min_length=1)
    taxonomy: list[Crop]


class CropTaxonomy:
    def __init__(self, config: CropTaxonomyConfig) -> None:
        self.config = config
        self._index: dict[tuple[str, str], str] = {}
        self._global_index: dict[str, str] = {}
        self._build_indexes()

    @classmethod
    def from_file(cls, path: Path | None = None) -> CropTaxonomy:
        source = path or find_project_root() / "configs" / "crops.yml"
        return cls(CropTaxonomyConfig.model_validate(load_yaml(source)))

    @property
    def crops(self) -> tuple[Crop, ...]:
        return tuple(self.config.taxonomy)

    def resolve(self, label: str, source: str | None = None) -> str:
        key = normalize_crop_label(label)
        if source:
            match = self._index.get((source.casefold(), key))
            if match:
                return match
        match = self._global_index.get(key)
        if match:
            return match
        raise KeyError(f"Unknown crop label: {label!r} (source={source!r})")

    def _register(
        self,
        index: dict[IndexKeyT, str],
        key: IndexKeyT,
        canonical: str,
    ) -> None:
        existing = index.get(key)
        if existing and existing != canonical:
            raise ValueError(f"Crop alias collision for {key!r}: {existing!r} vs {canonical!r}")
        index[key] = canonical

    def _build_indexes(self) -> None:
        ids: set[int] = set()
        canonical_names: set[str] = set()
        for crop in self.config.taxonomy:
            if crop.crop_id in ids:
                raise ValueError(f"Duplicate crop_id: {crop.crop_id}")
            if crop.canonical_name in canonical_names:
                raise ValueError(f"Duplicate canonical crop: {crop.canonical_name}")
            ids.add(crop.crop_id)
            canonical_names.add(crop.canonical_name)

            canonical_key = normalize_crop_label(crop.canonical_name)
            self._register(self._global_index, canonical_key, crop.canonical_name)
            for source, aliases in crop.aliases.items():
                for alias in aliases:
                    alias_key = normalize_crop_label(alias)
                    self._register(
                        self._index,
                        (source.casefold(), alias_key),
                        crop.canonical_name,
                    )
                    self._register(self._global_index, alias_key, crop.canonical_name)
