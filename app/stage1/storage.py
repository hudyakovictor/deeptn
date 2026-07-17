from __future__ import annotations

import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from .utils import atomic_json


@contextmanager
def atomic_photo_directory(output_root: Path, photo_id: str, overwrite: bool) -> Iterator[Path]:
    """Write to a sibling temp directory and atomically publish after validation."""
    output_root.mkdir(parents=True, exist_ok=True)
    final = output_root / photo_id
    temp = output_root / f".{photo_id}.incomplete-{uuid.uuid4().hex}"
    temp.mkdir(parents=False)
    try:
        yield temp
        if final.exists():
            if not overwrite:
                raise FileExistsError(f"destination already exists: {final}")
            backup = output_root / f".{photo_id}.old-{uuid.uuid4().hex}"
            os.replace(final, backup)
            try:
                os.replace(temp, final)
            except Exception:
                os.replace(backup, final)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.replace(temp, final)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def clean_incomplete(output_root: Path) -> int:
    count = 0
    if not output_root.exists():
        return 0
    for path in output_root.iterdir():
        if path.is_dir() and ".incomplete-" in path.name:
            shutil.rmtree(path, ignore_errors=True); count += 1
    return count


def write_failure(output_root: Path, photo_id: str, payload: dict) -> None:
    failures = output_root / "_failures"
    failures.mkdir(parents=True, exist_ok=True)
    atomic_json(failures / f"{photo_id}.json", payload)
