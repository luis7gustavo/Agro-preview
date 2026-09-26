from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImplementationStatus(StrEnum):
    PILOT = "pilot"
    PLANNED = "planned"


class Crop(BaseModel):
    model_config = ConfigDict(frozen=True)

    crop_id: int = Field(gt=0)
    canonical_name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    nome_ptbr: str = Field(min_length=1)
    categoria: str = Field(min_length=1)
    temporaria_permanente: str = Field(pattern=r"^(temporaria|permanente)$")
    cycle_days_min: int = Field(gt=0)
    cycle_days_max: int = Field(gt=0)
    active: bool
    implementation_status: ImplementationStatus
    aliases: dict[str, list[str]]

    @model_validator(mode="after")
    def validate_cycle_and_status(self) -> Crop:
        if self.cycle_days_max < self.cycle_days_min:
            raise ValueError("cycle_days_max must be >= cycle_days_min")
        if self.implementation_status == ImplementationStatus.PILOT and not self.active:
            raise ValueError("pilot crop must be active")
        return self
