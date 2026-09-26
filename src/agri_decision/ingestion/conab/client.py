from __future__ import annotations

from dataclasses import dataclass

from agri_decision.ingestion.http import HttpArtifact, OfficialSourceClient

COST_URL = (
    "https://portaldeinformacoes.conab.gov.br/downloads/arquivos/CustoProducao.txt"
)
MUNICIPAL_PRICE_URL = (
    "https://portaldeinformacoes.conab.gov.br/downloads/arquivos/PrecosMensalMunicipio.txt"
)
STATE_PRICE_URL = (
    "https://portaldeinformacoes.conab.gov.br/downloads/arquivos/PrecosMensalUF.txt"
)


@dataclass(frozen=True, slots=True)
class ConabDownloads:
    costs: HttpArtifact
    municipal_prices: HttpArtifact
    state_prices: HttpArtifact


class ConabClient:
    """Read-only client for the official Conab Portal de Informacoes exports."""

    def __init__(self, http: OfficialSourceClient | None = None) -> None:
        self.http = http or OfficialSourceClient(timeout_seconds=90)

    def fetch_costs(self) -> HttpArtifact:
        return self.http.get(COST_URL, accept="text/plain")

    def fetch_municipal_prices(self) -> HttpArtifact:
        return self.http.get(MUNICIPAL_PRICE_URL, accept="text/plain")

    def fetch_state_prices(self) -> HttpArtifact:
        return self.http.get(STATE_PRICE_URL, accept="text/plain")

