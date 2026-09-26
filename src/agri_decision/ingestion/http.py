from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx


@dataclass(frozen=True, slots=True)
class HttpArtifact:
    content: bytes
    requested_url: str
    final_url: str
    retrieved_at: datetime
    content_type: str | None


class OfficialSourceClient:
    """HTTP client with bounded retries for public, read-only data sources."""

    def __init__(self, *, timeout_seconds: float = 60.0, retries: int = 3) -> None:
        self.timeout_seconds = timeout_seconds
        self.retries = retries

    def get(
        self,
        url: str,
        *,
        accept: str = "application/json",
        extra_headers: dict[str, str] | None = None,
    ) -> HttpArtifact:
        last_error: Exception | None = None
        headers = {
            "Accept": accept,
            "User-Agent": "agri-decision-engine/0.1 (+public-data-ingestion)",
            **(extra_headers or {}),
        }
        for attempt in range(1, self.retries + 1):
            try:
                with httpx.Client(
                    timeout=self.timeout_seconds,
                    follow_redirects=True,
                    headers=headers,
                ) as client:
                    response = client.get(url)
                    response.raise_for_status()
                return HttpArtifact(
                    content=response.content,
                    requested_url=url,
                    final_url=str(response.url),
                    retrieved_at=datetime.now(UTC),
                    content_type=response.headers.get("content-type"),
                )
            except (httpx.HTTPError, httpx.TimeoutException) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(0.5 * (2 ** (attempt - 1)))
        raise RuntimeError(
            f"Official source request failed after {self.retries} attempts: {url}"
        ) from last_error
