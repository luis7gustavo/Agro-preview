from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agri_decision.schemas.provenance import DataNature, ProvenancedValue, SourceMetadata


def test_observed_value_preserves_source_without_estimation_method() -> None:
    value = ProvenancedValue[float](
        value=123.4,
        unit="R$/ha",
        data_nature=DataNature.OBSERVED,
        source="CONAB",
        source_reference="arquivo.xlsx:Custos!42",
    )

    assert value.data_nature == DataNature.OBSERVED
    assert value.method is None


@pytest.mark.parametrize(
    "nature",
    [DataNature.INTERPOLATED, DataNature.ESTIMATED, DataNature.PREDICTED],
)
def test_derived_value_requires_explicit_method(nature: DataNature) -> None:
    with pytest.raises(ValidationError, match="require an explicit method"):
        ProvenancedValue[float](
            value=123.4,
            unit="kg/ha",
            data_nature=nature,
            source="pipeline",
            source_reference="dataset-version-1",
        )


def test_bronze_metadata_requires_sha256_checksum() -> None:
    metadata = SourceMetadata(
        source="IBGE/PAM",
        download_url="https://example.invalid/official-file.zip",
        download_timestamp=datetime(2026, 8, 20, tzinfo=UTC),
        reference_period="2024",
        checksum_sha256="a" * 64,
        file_name="official-file.zip",
        pipeline_version="0.1.0",
    )

    assert len(metadata.checksum_sha256) == 64
