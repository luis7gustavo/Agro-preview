from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from agri_decision.schemas.provenance import DataNature


class ProductionHistoryRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    codigo_ibge: str = Field(pattern=r"^\d{7}$")
    municipio: str = Field(min_length=1)
    uf: str = Field(pattern=r"^[A-Z]{2}$")
    ano: int = Field(ge=1974, le=2200)
    cultura: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    sistema_produtivo: str = "not_available"
    area_plantada_ha: float | None = Field(default=None, ge=0)
    area_colhida_ha: float | None = Field(default=None, ge=0)
    producao_t: float | None = Field(default=None, ge=0)
    produtividade_kg_ha: float | None = Field(default=None, ge=0)
    valor_producao_original: float | None = Field(default=None, ge=0)
    valor_producao_unidade_original: str | None = None
    valor_producao_brl: float | None = Field(default=None, ge=0)
    source: str = "IBGE/PAM/SIDRA"
    source_table: str = "5457"
    source_type: DataNature = DataNature.OBSERVED
    source_reference: str
    source_dataset_checksum: str = Field(pattern=r"^[0-9a-f]{64}$")
    quality_flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def flag_area_inconsistency(self) -> ProductionHistoryRecord:
        if (
            self.area_plantada_ha is not None
            and self.area_colhida_ha is not None
            and self.area_colhida_ha > self.area_plantada_ha
            and "AREA_HARVESTED_GT_PLANTED" not in self.quality_flags
        ):
            raise ValueError("area inconsistency must be explicitly flagged")
        return self
