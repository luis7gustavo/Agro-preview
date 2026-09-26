from __future__ import annotations

from agri_decision.ingestion.http import HttpArtifact, OfficialSourceClient

ZARC_2026_2027_URL = (
    "https://dados.agricultura.gov.br/dataset/"
    "6d3d141c-885e-41a4-ab7f-dc8ff323b96f/resource/"
    "139e5a60-1f43-4cc8-aeab-a35dbbf816c0/download/"
    "dados-abertos-tabua-de-risco-safra-2026-2027.csv"
)


class ZarcClient:
    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient(timeout_seconds=240, retries=3)

    def fetch_current_table(self) -> HttpArtifact:
        # The CKAN file host currently rejects non-browser user agents even though
        # the resource is an official public download.
        return self.http.get(
            ZARC_2026_2027_URL,
            accept="text/csv",
            extra_headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/140 Safari/537.36"
                ),
                "Referer": "https://dados.agricultura.gov.br/",
            },
        )

