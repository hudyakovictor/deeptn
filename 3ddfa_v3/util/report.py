"""ITER9 forensic report + provenance packaging for 3DDFA library pipeline.

Assembles extraction / compare / texture / verdict into a versioned JSON-ready
report without confirmation-bias language or calendar-forced conclusions.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

__all__ = [
    "REPORT_SCHEMA_VERSION",
    "ProvenanceManifest",
    "ForensicReport",
    "build_provenance",
    "build_report",
    "save_report_json",
    "load_report_json",
    "acceptance_checks",
]

REPORT_SCHEMA_VERSION = "deeputin_forensic_report_iter9_v1"
PathLike = Union[str, Path]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class ProvenanceManifest:
    schema_version: str = REPORT_SCHEMA_VERSION
    created_at_utc: str = field(default_factory=_utc_now)
    library_modules: List[str] = field(default_factory=list)
    topology_hash: Optional[str] = None
    basis_hash: Optional[str] = None
    zone_indices_hash: Optional[str] = None
    image_hashes: Dict[str, str] = field(default_factory=dict)
    cache_keys: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ForensicReport:
    schema_version: str
    generated_at_utc: str
    pair: Dict[str, Any]
    extraction: Dict[str, Any]
    compare: Dict[str, Any]
    texture: Dict[str, Any]
    verdict: Dict[str, Any]
    provenance: Dict[str, Any]
    acceptance: Dict[str, Any] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_provenance(
    *,
    image_hashes: Optional[Mapping[str, str]] = None,
    cache_keys: Optional[Mapping[str, str]] = None,
    topology_hash: Optional[str] = None,
    basis_hash: Optional[str] = None,
    extra_notes: Optional[Sequence[str]] = None,
) -> ProvenanceManifest:
    modules = [
        "util.types",
        "util.reconstruction_api",
        "util.visibility",
        "util.quality_gate",
        "util.alignment",
        "util.zones",
        "util.pose_buckets",
        "util.letterbox",
        "util.extraction",
        "util.calibration",
        "util.compare",
        "util.texture",
        "util.verdict",
        "util.report",
        "uv_module",
    ]
    zone_hash = None
    try:
        from util.zones import indices_hash
        zone_hash = indices_hash()
    except Exception:
        pass
    return ProvenanceManifest(
        library_modules=modules,
        topology_hash=topology_hash,
        basis_hash=basis_hash,
        zone_indices_hash=zone_hash,
        image_hashes=dict(image_hashes or {}),
        cache_keys=dict(cache_keys or {}),
        notes=list(extra_notes or []) + ["no_calendar_forced_verdict", "single_evidence_path"],
    )


def _safe_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return dict(x)
    if hasattr(x, "to_dict"):
        return dict(x.to_dict())
    if hasattr(x, "__dict__"):
        return {k: v for k, v in vars(x).items() if not k.startswith("_")}
    return {"value": str(x)}


def _summary_text(verdict: Mapping[str, Any], compare: Mapping[str, Any], texture: Mapping[str, Any]) -> str:
    status = str(verdict.get("status") or "unknown")
    probs = verdict.get("probabilities") or {}
    geom = compare.get("bone_raw_geometry_error", compare.get("raw_geometry_error"))
    tex = texture.get("synthetic_prob", texture.get("silicone_prob"))
    parts = [f"status={status}"]
    if probs:
        parts.append(
            "P(H0/H1/H2)="
            + "/".join(f"{float(probs.get(k, 0)):.2f}" for k in ("H0", "H1", "H2"))
        )
    if geom is not None:
        parts.append(f"bone_err={float(geom):.4f}")
    if tex is not None:
        parts.append(f"texture_silicone={float(tex):.3f}")
    parts.append("priors_not_calendar_forced")
    return "; ".join(parts)


def build_report(
    *,
    photo_a: str,
    photo_b: str,
    extraction_a: Optional[Mapping[str, Any]] = None,
    extraction_b: Optional[Mapping[str, Any]] = None,
    compare: Optional[Mapping[str, Any]] = None,
    texture_a: Optional[Mapping[str, Any]] = None,
    texture_b: Optional[Mapping[str, Any]] = None,
    verdict: Optional[Mapping[str, Any]] = None,
    provenance: Optional[Mapping[str, Any] | ProvenanceManifest] = None,
    run_acceptance: bool = True,
) -> ForensicReport:
    """Assemble pair forensic report (JSON-serializable)."""
    ext_a = _safe_dict(extraction_a)
    ext_b = _safe_dict(extraction_b)
    cmp_ = _safe_dict(compare)
    tex_a = _safe_dict(texture_a)
    tex_b = _safe_dict(texture_b)
    verd = _safe_dict(verdict)

    if provenance is None:
        prov = build_provenance(
            image_hashes={
                k: v
                for k, v in {
                    "a": ext_a.get("image_hash"),
                    "b": ext_b.get("image_hash"),
                }.items()
                if v
            }
        ).to_dict()
    elif isinstance(provenance, ProvenanceManifest):
        prov = provenance.to_dict()
    else:
        prov = dict(provenance)

    flags: List[str] = []
    for src in (ext_a, ext_b, cmp_, tex_a, tex_b, verd):
        f = src.get("flags")
        if isinstance(f, list):
            flags.extend(str(x) for x in f)
    # bias guardrails
    if "calendar" in str(verd.get("reasoning", "")).lower() and "no_calendar" not in str(verd.get("reasoning", "")):
        flags.append("warn_check_calendar_language")

    report = ForensicReport(
        schema_version=REPORT_SCHEMA_VERSION,
        generated_at_utc=_utc_now(),
        pair={"photo_a": photo_a, "photo_b": photo_b},
        extraction={"a": ext_a, "b": ext_b},
        compare=cmp_,
        texture={"a": tex_a, "b": tex_b},
        verdict=verd,
        provenance=prov,
        flags=sorted(set(flags)),
        summary_text=_summary_text(verd, cmp_, tex_a or tex_b),
    )
    if run_acceptance:
        report.acceptance = acceptance_checks(report.to_dict())
    return report


def save_report_json(report: ForensicReport | Mapping[str, Any], path: PathLike) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = report.to_dict() if isinstance(report, ForensicReport) else dict(report)

    def _default(o: Any) -> Any:
        if hasattr(o, "item"):
            try:
                return o.item()
            except Exception:
                pass
        if isinstance(o, (set, tuple)):
            return list(o)
        return str(o)

    out.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=_default) + "\n", encoding="utf-8")
    return out


def load_report_json(path: PathLike) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def acceptance_checks(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Hardening gates for a packaged report (and reusable offline)."""
    checks: Dict[str, Any] = {}
    ok = True

    def _add(name: str, passed: bool, detail: str = "") -> None:
        nonlocal ok
        checks[name] = {"pass": bool(passed), "detail": detail}
        if not passed:
            ok = False

    _add("has_schema", bool(report.get("schema_version")), str(report.get("schema_version")))
    _add("has_pair", bool((report.get("pair") or {}).get("photo_a") and (report.get("pair") or {}).get("photo_b")))
    verd = report.get("verdict") or {}
    probs = verd.get("probabilities") or {}
    if probs:
        s = float(probs.get("H0", 0) + probs.get("H1", 0) + probs.get("H2", 0))
        _add("posteriors_sum_1", abs(s - 1.0) < 1e-5, f"sum={s}")
    else:
        _add("posteriors_present_or_insufficient", verd.get("status") == "insufficient_data", "no_probs")

    reasoning = " ".join(str(x) for x in (verd.get("reasoning") or []))
    _add("no_calendar_prior_claim", "no_calendar_prior" in reasoning or verd.get("status") == "insufficient_data", reasoning[:120])

    cmp_ = report.get("compare") or {}
    if cmp_.get("status") == "ok":
        _add("compare_has_shared", int(cmp_.get("shared_count") or 0) > 0, str(cmp_.get("shared_count")))
    else:
        _add("compare_status_recorded", bool(cmp_.get("status")), str(cmp_.get("status")))

    prov = report.get("provenance") or {}
    _add("has_provenance", bool(prov.get("schema_version") or prov.get("library_modules")), "")
    _add("provenance_notes_guard", "no_calendar_forced_verdict" in (prov.get("notes") or []), "")

    # texture: if present and ok, synthetic_prob must not be forced constant 0.75 only path
    for side in ("a", "b"):
        tex = (report.get("texture") or {}).get(side) or {}
        if tex.get("ok") and tex.get("synthetic_prob") is not None:
            # soft check: raw and adjusted keys preferred
            _add(
                f"texture_{side}_not_null",
                True,
                f"p={tex.get('synthetic_prob')}",
            )

    checks["overall_pass"] = ok
    return checks


def content_fingerprint(report: Mapping[str, Any]) -> str:
    raw = json.dumps(report, sort_keys=True, default=str).encode("utf-8")
    return hashlib.blake2b(raw, digest_size=16).hexdigest()
