from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class ChangeClass(str, Enum):
    UI_ONLY = "A"
    RECLASSIFY = "B"
    RECOMPUTE = "C"
    STAGE1_INVALIDATION = "D"


STEPS = (
    "input_freeze", "quality_review", "calibration", "geometry", "texture",
    "fusion", "cross_pose", "chronology", "private_retest", "release",
)

DEPENDENCIES: dict[str, set[str]] = {
    "input_freeze": set(),
    "quality_review": {"input_freeze"},
    "calibration": {"quality_review"},
    "geometry": {"quality_review", "calibration"},
    "texture": {"quality_review", "calibration"},
    "fusion": {"geometry", "texture"},
    "cross_pose": {"geometry", "texture", "fusion"},
    "chronology": {"geometry", "texture", "fusion", "cross_pose"},
    "private_retest": {"chronology"},
    "release": {"quality_review", "calibration", "geometry", "texture", "fusion", "cross_pose", "chronology"},
}


@dataclass(frozen=True)
class ChangeImpact:
    change_class: ChangeClass
    changed_paths: tuple[str, ...]
    affected_steps: tuple[str, ...]
    approvals_to_mark_stale: tuple[str, ...]
    approvals_to_revoke: tuple[str, ...]
    reason_codes: tuple[str, ...]


def _downstream(seed: Iterable[str]) -> set[str]:
    result = set(seed)
    changed = True
    while changed:
        changed = False
        for step, deps in DEPENDENCIES.items():
            if step not in result and deps.intersection(result):
                result.add(step)
                changed = True
    return result


def classify_parameter(path: str) -> tuple[ChangeClass, set[str], str]:
    if path.startswith("ui.") or path.startswith("quality.display."):
        return ChangeClass.UI_ONLY, set(), "display_only"
    if path.startswith("stage1."):
        return ChangeClass.STAGE1_INVALIDATION, set(STEPS), "stage1_contract_changed"
    if path.startswith("quality.admission.geometry"):
        return ChangeClass.RECOMPUTE, _downstream({"geometry"}), "geometry_admission_changed"
    if path.startswith("quality.admission.texture"):
        return ChangeClass.RECOMPUTE, _downstream({"texture"}), "texture_admission_changed"
    if path.startswith("quality.admission."):
        return ChangeClass.RECOMPUTE, _downstream({"quality_review"}), "quality_admission_changed"
    if path.startswith("calibration."):
        return ChangeClass.RECOMPUTE, _downstream({"calibration"}), "calibration_changed"
    if path.startswith("geometry.compute."):
        return ChangeClass.RECOMPUTE, _downstream({"geometry"}), "geometry_compute_changed"
    if path.startswith("geometry.decision."):
        return ChangeClass.RECLASSIFY, _downstream({"geometry"}), "geometry_decision_changed"
    if path.startswith("texture.compute."):
        return ChangeClass.RECOMPUTE, _downstream({"texture"}), "texture_compute_changed"
    if path.startswith("texture.decision."):
        return ChangeClass.RECLASSIFY, _downstream({"texture"}), "texture_decision_changed"
    if path.startswith("fusion."):
        return ChangeClass.RECLASSIFY, _downstream({"fusion"}), "fusion_rule_changed"
    if path.startswith("cross_pose."):
        return ChangeClass.RECLASSIFY, _downstream({"cross_pose"}), "cross_pose_rule_changed"
    if path.startswith("chronology.pairing."):
        return ChangeClass.RECOMPUTE, _downstream({"chronology"}), "chronology_pairing_changed"
    if path.startswith("chronology.decision."):
        return ChangeClass.RECLASSIFY, _downstream({"chronology"}), "chronology_decision_changed"
    if path.startswith("private."):
        return ChangeClass.RECLASSIFY, {"private_retest"}, "private_rule_changed"
    if path.startswith("release."):
        return ChangeClass.RECLASSIFY, {"release"}, "release_rule_changed"
    raise ValueError(f"unregistered parameter namespace: {path}")


def analyze_paths(paths: Iterable[str], approved_steps: Iterable[str] = ()) -> ChangeImpact:
    paths = tuple(paths)
    classifications = [classify_parameter(path) for path in paths]
    if not classifications:
        return ChangeImpact(ChangeClass.UI_ONLY, (), (), (), (), ())
    rank = {ChangeClass.UI_ONLY: 0, ChangeClass.RECLASSIFY: 1, ChangeClass.RECOMPUTE: 2, ChangeClass.STAGE1_INVALIDATION: 3}
    overall = max((item[0] for item in classifications), key=rank.get)
    affected = set().union(*(item[1] for item in classifications))
    reasons = tuple(sorted({item[2] for item in classifications}))
    approved = set(approved_steps)
    stale: set[str] = set()
    revoked: set[str] = set()
    for step in approved.intersection(affected):
        step_classes = [cls for cls, steps, _ in classifications if step in steps]
        if any(cls in {ChangeClass.RECOMPUTE, ChangeClass.STAGE1_INVALIDATION} for cls in step_classes):
            revoked.add(step)
        else:
            stale.add(step)
    return ChangeImpact(
        change_class=overall,
        changed_paths=paths,
        affected_steps=tuple(step for step in STEPS if step in affected),
        approvals_to_mark_stale=tuple(step for step in STEPS if step in stale),
        approvals_to_revoke=tuple(step for step in STEPS if step in revoked),
        reason_codes=reasons,
    )
