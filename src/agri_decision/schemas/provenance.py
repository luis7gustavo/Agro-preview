from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataNature(StrEnum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    ESTIMATED = "estimated"
    PREDICTED = "predicted"


class SourceMetadata(BaseModel):
    """Immutable metadata attached to every Bronze artifact."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(min_length=2)
    download_url: str = Field(min_length=4)
    download_timestamp: datetime
    reference_period: str = Field(min_length=1)
    checksum_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_name: str = Field(min_length=1)
    pipeline_version: str = Field(min_length=1)
    source_version: str | None = None
    license: str | None = None


ValueT = TypeVar("ValueT", int, float, str)


class ProvenancedValue(BaseModel, Generic[ValueT]):
    """A value that cannot hide whether it was measured or derived."""

    model_config = ConfigDict(frozen=True)

    value: ValueT | None
    unit: str | None = None
    data_nature: DataNature
    source: str = Field(min_length=2)
    source_reference: str = Field(min_length=1)
    method: str | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def derived_values_require_method(self) -> ProvenancedValue[ValueT]:
        derived = {
            DataNature.INTERPOLATED,
            DataNature.ESTIMATED,
            DataNature.PREDICTED,
        }
        if self.data_nature in derived and not self.method:
            raise ValueError(f"{self.data_nature.value} values require an explicit method")
        return self
