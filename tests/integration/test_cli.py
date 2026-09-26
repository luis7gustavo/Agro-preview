import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from agri_decision.cli import app

runner = CliRunner()


def test_cli_validates_versioned_configuration() -> None:
    result = runner.invoke(app, ["config", "validate"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "ok"


def test_cli_resolves_ibge_crop_alias() -> None:
    result = runner.invoke(app, ["crops", "resolve", "Soja (em grão)", "--source", "ibge"])

    assert result.exit_code == 0, result.output
    assert result.output.strip() == "soja"


def test_cli_lists_only_active_crops_by_default() -> None:
    result = runner.invoke(app, ["crops", "list"])

    assert result.exit_code == 0, result.output
    crops = json.loads(result.output)
    assert [crop["canonical_name"] for crop in crops] == ["soja"]


def test_arco_cli_passes_bounded_cached_reprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agri_decision.ingestion.era5 import arco_pipeline

    calls: list[dict[str, Any]] = []
    report = tmp_path / "arco-report.json"
    report.write_text('{"coverage_status": "complete"}', encoding="utf-8")

    def fake_run(**kwargs: Any) -> Path:
        calls.append(kwargs)
        return report

    monkeypatch.setattr(arco_pipeline, "run_era5_arco_backfill", fake_run)
    result = runner.invoke(
        app,
        ["ingest", "era5-arco", "--reprocess-existing", "--max-new-points", "0", "--workers", "2"],
    )
    assert result.exit_code == 0, result.output
    assert calls[0]["reprocess_existing"] is True
    assert calls[0]["max_new_points"] == 0
    assert calls[0]["workers"] == 2


def test_era5_wait_keeps_total_submission_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import agri_decision.cli as module

    budgets: list[int] = []
    report = tmp_path / "report.json"

    def fake_run(**kwargs: Any) -> Path:
        budget = kwargs["max_new_requests"]
        budgets.append(budget)
        submitted = min(3, budget)
        report.write_text(
            json.dumps(
                {
                    "new_requests_submitted": submitted,
                    "job_status_counts": {"accepted": submitted},
                    "downloads_complete": False,
                    "jobs_requiring_attention": [],
                }
            )
        )
        return report

    monkeypatch.setattr(module, "run_era5_backfill", fake_run)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    result = runner.invoke(
        app, ["ingest", "era5-backfill", "--max-new-requests", "6", "--wait-seconds", "60"]
    )
    assert result.exit_code == 0, result.output
    assert budgets == [6, 3, 0]
