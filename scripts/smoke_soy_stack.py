"""Exercise real local data/model through HTTP and Streamlit, without deployment.

Usage: .venv\\Scripts\\python.exe scripts/smoke_soy_stack.py
Requires the API, UI and ML extras and a promoted model trained on the current Gold.
Starts its own loopback-only server on an available port and always stops that child.
Recommendation/audit JSON files are intentionally persisted by the normal API flow.
Run after climate ingestion and model training have finished, not concurrently.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx

SCENARIOS = (
    ("5107925", "station"),  # Sorriso: INMET in the model's lagged climate season.
    ("5102793", "station"),  # Carlinda: ERA5 in 2025, but lagged 2024 is still INMET.
    ("5104104", "reanalysis"),  # Guaranta do Norte: actual ARCO fallback in lagged 2024.
)


@contextmanager
def local_api(root: Path) -> Iterator[str]:
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    api_url = f"http://127.0.0.1:{port}"
    with tempfile.TemporaryFile(mode="w+b") as server_log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "agri_decision.api.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "error",
            ],
            cwd=root,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        try:
            deadline = time.monotonic() + 120
            ready = False
            with httpx.Client(timeout=2, trust_env=False) as client:
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        break
                    try:
                        response = client.get(f"{api_url}/health")
                        if response.status_code == 200 and response.json() == {"status": "ok"}:
                            ready = True
                            break
                    except httpx.HTTPError:
                        pass
                    time.sleep(0.25)
            if ready:
                yield api_url
                return
            server_log.seek(0)
            detail = server_log.read().decode("utf-8", errors="replace")[-4000:]
            raise RuntimeError(f"The local smoke API failed to become ready: {detail}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)


def check_analysis(analysis: dict[str, Any], code: str, origin: str) -> dict[str, Any]:
    assert analysis["location"]["codigo_ibge"] == code, "Wrong municipality in analysis"
    assert analysis["recommendations"], f"No recommendation available for municipality {code}"
    recommendation = analysis["recommendations"][0]
    climate = recommendation["climate"]
    prediction = recommendation["yield"]
    assert climate["status"] == "available", f"Climate unavailable for {code}"
    assert climate["origin"] == origin, f"Unexpected climate source for {code}: {climate['source']}"
    assert prediction["type"] == "predicted", f"Promoted ML prediction unavailable for {code}"
    assert climate["reference_season"] == prediction["features_reference"]["climate_season"]
    assert climate["source"] == prediction["features_reference"]["climate_source"]
    assert climate["source"] in prediction["training_climate_sources"], "Unseen training source"
    assert "CLIMATE_SOURCE_NOT_IN_MODEL_TRAINING" not in recommendation["alerts"]
    assert "MODEL_PREDICTION_UNAVAILABLE" not in recommendation["alerts"]
    assert analysis["audit"]["analysis_id"] == analysis["analysis_id"]
    if origin == "reanalysis":
        assert "ARCO" in climate["source"], "Expected the real ARCO fallback fixture"
        assert climate["type"] == "estimated"
        assert climate["distance_to_station_km"] is None
        assert "ERA5_LAND_FALLBACK" in recommendation["alerts"]
    return {
        "codigo_ibge": code,
        "municipio": analysis["location"]["municipio"],
        "analysis_id": analysis["analysis_id"],
        "climate_source": climate["source"],
        "climate_reference_season": climate["reference_season"],
        "model_version": prediction["model_version"],
        "dataset_version": prediction["dataset_version"],
        "yield_target_year": prediction["target_year"],
        "yield_p50_kg_ha": prediction["p50"],
        "alerts": recommendation["alerts"],
    }


def api_smoke(api_url: str) -> list[dict[str, Any]]:
    results = []
    with httpx.Client(base_url=api_url, timeout=120, trust_env=False) as client:
        assert client.get("/health").json() == {"status": "ok"}
        for code, origin in SCENARIOS:
            found = client.get("/v1/locations/search", params={"q": code})
            found.raise_for_status()
            assert any(row["codigo_ibge"] == code for row in found.json())
            response = client.post(
                "/v1/recommendations",
                json={"codigo_ibge": code, "area_ha": 100.0, "profile": "balanced"},
            )
            assert response.status_code == 201, response.text
            analysis = response.json()
            summary = check_analysis(analysis, code, origin)
            analysis_id = analysis["analysis_id"]
            stored = client.get(f"/v1/recommendations/{analysis_id}")
            stored.raise_for_status()
            assert stored.json() == analysis, "Persisted recommendation differs from HTTP response"
            audit = client.get(f"/v1/recommendations/{analysis_id}/audit")
            audit.raise_for_status()
            assert audit.json() == analysis["audit"], "Persisted audit differs from HTTP response"
            results.append(summary)
    return results


def streamlit_smoke(root: Path, api_url: str) -> list[dict[str, Any]]:
    from streamlit.testing.v1 import AppTest

    results = []
    with patch.dict(os.environ, {"AGRI_API_URL": api_url}):
        for code, origin in SCENARIOS:
            app = AppTest.from_file(str(root / "app" / "streamlit_app.py"))
            app.run(timeout=120)
            assert not app.exception, str(app.exception)
            app.text_input[0].set_value(code).run(timeout=120)
            assert not app.exception, str(app.exception)
            assert not app.error, str(app.error)
            app.button[0].click().run(timeout=120)
            assert not app.exception, str(app.exception)
            assert not app.error, str(app.error)
            summary = check_analysis(app.session_state["analysis"], code, origin)
            assert len(app.metric) == 4, "Expected score/economics/risk cards"
            if origin == "reanalysis":
                assert any("ERA5-Land" in item.value for item in app.info)
            results.append(summary)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-ui", action="store_true", help="Run only the HTTP/API checks")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with local_api(root) as api_url:
        result = {
            "status": "passed",
            "api": api_smoke(api_url),
            "streamlit": "skipped" if args.skip_ui else streamlit_smoke(root, api_url),
            "persistence": "Normal recommendation/audit records remain in data/gold/analyses",
        }
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
