from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp_path = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp_path, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    write_bytes_atomic(path, (serialized + "\n").encode("utf-8"))


def preserve_content_addressed_artifact(
    *,
    directory: Path,
    stem: str,
    suffix: str,
    content: bytes,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    checksum = sha256_bytes(content)
    artifact_path = directory / f"{stem}-{checksum[:16]}{suffix}"
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")

    if artifact_path.exists():
        if sha256_file(artifact_path) != checksum:
            raise RuntimeError(f"Immutable Bronze collision at {artifact_path}")
    else:
        write_bytes_atomic(artifact_path, content)

    if not manifest_path.exists():
        write_json_atomic(
            manifest_path,
            {
                **manifest,
                "checksum_sha256": checksum,
                "file_name": artifact_path.name,
            },
        )
    return artifact_path, manifest_path
