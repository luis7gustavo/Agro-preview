import pytest
from pydantic import ValidationError

from agri_decision.dimensions import validate_dim_locations
from agri_decision.schemas.location import Location


def valid_location_payload() -> dict[str, object]:
    # Synthetic fixture used only to test validation rules.
    return {
        "codigo_ibge": "1234567",
        "municipio": "Municipio de Teste",
        "uf": "DF",
        "regiao": "Centro-Oeste",
        "latitude_centroid": -15.0,
        "longitude_centroid": -47.0,
        "geometry": None,
    }


def test_location_requires_seven_digit_ibge_code() -> None:
    payload = valid_location_payload()
    payload["codigo_ibge"] = "123456"

    with pytest.raises(ValidationError):
        Location.model_validate(payload)


def test_location_rejects_invalid_coordinates() -> None:
    payload = valid_location_payload()
    payload["latitude_centroid"] = -100.0

    with pytest.raises(ValidationError):
        Location.model_validate(payload)


def test_dimension_rejects_duplicate_ibge_keys() -> None:
    payload = valid_location_payload()

    with pytest.raises(ValueError, match="Duplicate codigo_ibge"):
        validate_dim_locations([payload, payload])
