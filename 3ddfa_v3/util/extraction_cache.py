"""ITER4 content-hash cache for reconstruction / extraction artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

__all__ = [
    "content_hash_bytes",
    "content_hash_file",
    "content_hash_image_array",
    "make_cache_key",
    "ExtractionCache",
]

PathLike = Union[str, Path]


def content_hash_bytes(data: bytes, *, digest_size: int = 16) -> str:
    return hashlib.blake2b(data, digest_size=digest_size).hexdigest()


def content_hash_file(path: PathLike, *, chunk_size: int = 1 << 20) -> str:
    h = hashlib.blake2b(digest_size=16)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def content_hash_image_array(image: np.ndarray) -> str:
    arr = np.ascontiguousarray(image)
    h = hashlib.blake2b(digest_size=16)
    h.update(str(arr.dtype).encode())
    h.update(str(arr.shape).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def make_cache_key(
    *,
    image_hash: str,
    backbone: str = "resnet50",
    expression_mode: str = "full",
    identity_only: bool = False,
    schema_version: str = "3ddfa_v3_iter1_v1",
    extra: Optional[str] = None,
) -> str:
    parts = [
        image_hash,
        backbone,
        expression_mode,
        "id1" if identity_only else "id0",
        schema_version,
    ]
    if extra:
        parts.append(str(extra))
    raw = "|".join(parts).encode()
    return content_hash_bytes(raw)


class ExtractionCache:
    """Filesystem cache keyed by content hash (not mtime)."""

    def __init__(self, root: PathLike, *, max_entries: int = 256):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.max_entries = int(max_entries)
        self._index_path = self.root / "index.json"
        self._index: Dict[str, str] = {}
        if self._index_path.exists():
            try:
                self._index = json.loads(self._index_path.read_text(encoding="utf-8"))
            except Exception:
                self._index = {}

    def _path_for(self, key: str) -> Path:
        return self.root / f"{key}.npz"

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        path = self._path_for(key)
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=True)
        out: Dict[str, Any] = {}
        for k in data.files:
            v = data[k]
            if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ():
                out[k] = v.item()
            else:
                out[k] = v
        return out

    def set(self, key: str, payload: Dict[str, Any]) -> Path:
        path = self._path_for(key)
        clean = {k: v for k, v in payload.items() if v is not None}
        np.savez_compressed(path, **clean)
        self._index[key] = path.name
        self._evict_if_needed()
        self._index_path.write_text(json.dumps(self._index, indent=2), encoding="utf-8")
        return path

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def _evict_if_needed(self) -> None:
        if len(self._index) <= self.max_entries:
            return
        # drop oldest keys (insertion order in py3.7+)
        overflow = len(self._index) - self.max_entries
        keys = list(self._index.keys())[:overflow]
        for k in keys:
            p = self._path_for(k)
            if p.exists():
                p.unlink()
            self._index.pop(k, None)
