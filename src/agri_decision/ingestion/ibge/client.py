from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from agri_decision.ingestion.http import HttpArtifact, OfficialSourceClient

SIDRA_BASE_URL = "https://apisidra.ibge.gov.br/values"
LOCATIONS_URL = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
MESH_BASE_URL = "https://servicodados.ibge.gov.br/api/v3/malhas/estados"
PAM_TABLE_ID = 5457
PAM_VARIABLE_IDS = (8331, 216, 214, 112, 215)
PAM_CROP_CODES = {"soja": 40124}
BRAZILIAN_UFS = (
    "AC",
    "AL",
    "AP",
    "AM",
    "BA",
    "CE",
    "DF",
    "ES",
    "GO",
    "MA",
    "MT",
    "MS",
    "MG",
    "PA",
    "PB",
    "PR",
    "PE",
    "PI",
    "RJ",
    "RN",
    "RS",
    "RO",
    "RR",
    "SC",
    "SP",
    "SE",
    "TO",
)


@dataclass(frozen=True, slots=True)
class JsonArtifact:
    rows: Any
    http: HttpArtifact


def decode_json_artifact(artifact: HttpArtifact) -> JsonArtifact:
    try:
        payload = json.loads(artifact.content.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid JSON returned by {artifact.final_url}") from exc
    return JsonArtifact(rows=payload, http=artifact)


class SidraPamClient:
    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient()

    @staticmethod
    def build_year_url(year: int, crop: str = "soja") -> str:
        if year < 1974 or year > 2200:
            raise ValueError(f"Unsupported PAM year: {year}")
        try:
            crop_code = PAM_CROP_CODES[crop]
        except KeyError as exc:
            raise ValueError(f"PAM crop is not enabled yet: {crop}") from exc
        variables = ",".join(str(item) for item in PAM_VARIABLE_IDS)
        return (
            f"{SIDRA_BASE_URL}/t/{PAM_TABLE_ID}/n6/all/v/{variables}"
            f"/p/{year}/c782/{crop_code}?formato=json"
        )

    def fetch_year(self, year: int, crop: str = "soja") -> JsonArtifact:
        result = decode_json_artifact(self.http.get(self.build_year_url(year, crop)))
        if not isinstance(result.rows, list) or not result.rows:
            raise ValueError(f"Unexpected empty SIDRA response for PAM {year}/{crop}")
        header = result.rows[0]
        expected = {"D1C", "D1N", "D2C", "D2N", "D3C", "D3N", "D4C", "D4N", "V"}
        if not isinstance(header, dict) or not expected.issubset(header):
            raise ValueError(f"SIDRA schema changed for PAM {year}/{crop}")
        return result


class IbgeTerritoryClient:
    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient()

    def fetch_municipalities(self) -> JsonArtifact:
        result = decode_json_artifact(self.http.get(LOCATIONS_URL))
        if not isinstance(result.rows, list) or not result.rows:
            raise ValueError("Unexpected empty IBGE municipality response")
        required = {"id", "nome", "regiao-imediata"}
        if not isinstance(result.rows[0], dict) or not required.issubset(result.rows[0]):
            raise ValueError("IBGE Localidades schema changed")
        return result

    @staticmethod
    def state_mesh_url(uf: str) -> str:
        normalized = uf.upper()
        if normalized not in BRAZILIAN_UFS:
            raise ValueError(f"Invalid Brazilian UF: {uf}")
        geojson = quote("application/vnd.geo+json", safe="")
        return (
            f"{MESH_BASE_URL}/{normalized}?formato={geojson}"
            "&qualidade=minima&intrarregiao=municipio"
        )

    def fetch_state_mesh(self, uf: str) -> JsonArtifact:
        artifact = self.http.get(
            self.state_mesh_url(uf),
            accept="application/vnd.geo+json",
        )
        result = decode_json_artifact(artifact)
        if not isinstance(result.rows, dict) or result.rows.get("type") != "FeatureCollection":
            raise ValueError(f"Unexpected IBGE mesh response for {uf}")
        features = result.rows.get("features")
        if not isinstance(features, list) or not features:
            raise ValueError(f"Empty IBGE municipal mesh for {uf}")
        return result
