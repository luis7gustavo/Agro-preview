import pytest

from agri_decision.ingestion.ibge.client import IbgeTerritoryClient, SidraPamClient


def test_sidra_url_uses_verified_pam_contract_for_soybean() -> None:
    url = SidraPamClient.build_year_url(2024, "soja")

    assert "/t/5457/n6/all/v/8331,216,214,112,215/p/2024/c782/40124" in url


def test_pam_rejects_crop_not_enabled_in_adapter() -> None:
    with pytest.raises(ValueError, match="not enabled"):
        SidraPamClient.build_year_url(2024, "milho")


def test_mesh_url_requests_geojson_municipal_subregions() -> None:
    url = IbgeTerritoryClient.state_mesh_url("mt")

    assert "/estados/MT?" in url
    assert "intrarregiao=municipio" in url
    assert "qualidade=minima" in url
