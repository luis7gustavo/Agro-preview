import json

from agri_decision.provenance.artifacts import preserve_content_addressed_artifact


def test_bronze_artifact_is_content_addressed_and_idempotent(tmp_path) -> None:
    content = b'{"official": true}'
    manifest = {
        "source": "test-only",
        "download_url": "https://example.invalid/data",
        "download_timestamp": "2026-08-20T00:00:00+00:00",
        "reference_period": "test",
        "pipeline_version": "test",
    }

    first, first_manifest = preserve_content_addressed_artifact(
        directory=tmp_path,
        stem="artifact",
        suffix=".json",
        content=content,
        manifest=manifest,
    )
    second, second_manifest = preserve_content_addressed_artifact(
        directory=tmp_path,
        stem="artifact",
        suffix=".json",
        content=content,
        manifest=manifest,
    )

    assert first == second
    assert first_manifest == second_manifest
    assert json.loads(first_manifest.read_text(encoding="utf-8"))["checksum_sha256"]
