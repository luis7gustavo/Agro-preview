from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Location(BaseModel):
    """Canonical municipal dimension keyed exclusively by IBGE code."""

    model_config = ConfigDict(frozen=True)

    codigo_ibge: str = Field(pattern=r"^\d{7}$")
    municipio: str = Field(min_length=1)
    uf: str = Field(pattern=r"^[A-Z]{2}$")
    regiao: str = Field(min_length=1)
    latitude_centroid: float | None = Field(default=None, ge=-90, le=90)
    longitude_centroid: float | None = Field(default=None, ge=-180, le=180)
    geometry: str | None = None
    source: str = "IBGE/Localidades+Malhas"
    source_reference: str | None = None
    quality_flags: list[str] = Field(default_factory=list)

    @field_validator("municipio", "regiao")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text value cannot be blank")
        return cleaned
