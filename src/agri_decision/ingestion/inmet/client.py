from __future__ import annotations

from agri_decision.ingestion.http import HttpArtifact, OfficialSourceClient

ANNUAL_URL_TEMPLATE = "https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip"


class InmetClient:
    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient(timeout_seconds=300, retries=3)

    def fetch_year(self, year: int) -> HttpArtifact:
        if year < 2000:
            raise ValueError("INMET automatic annual archives begin in 2000")
        return self.http.get(
            ANNUAL_URL_TEMPLATE.format(year=year),
            accept="application/zip",
            extra_headers={"User-Agent": "Mozilla/5.0 agri-decision-engine/0.1"},
        )

