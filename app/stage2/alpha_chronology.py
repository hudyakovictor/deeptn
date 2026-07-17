from __future__ import annotations

from typing import Any

from .core import calibrated_score

ALPHA_SCHEMA = "deeputin-stage2-alpha-chronology-v1.0"


def apply_alpha_chronology(rows: list[dict[str, Any]], model: Any) -> dict[str, Any]:
    """Annotate pair rows with calibrated alpha_id / alpha_exp chronology signals.

    alpha_id is treated as an additional identity-shape channel, not as an identity
    verdict. alpha_exp is used as an expression-leakage explanation channel.
    """
    events: list[dict[str, Any]] = []
    for r in rows:
        pose = str(r.get("pose_bin") or "")
        aid = r.get("alpha_id_l2")
        aexp = r.get("alpha_exp_l2")
        if aid is None:
            continue
        alpha_id_score = calibrated_score(float(aid), model.reference(pose, "alpha_id_l2"), [])
        alpha_exp_score = calibrated_score(float(aexp or 0.0), model.reference(pose, "alpha_exp_l2"), [])
        r["alpha_id_status"] = alpha_id_score["status"]
        r["alpha_id_robust_z"] = alpha_id_score["robust_z"]
        r["alpha_id_calibration_p95"] = alpha_id_score["calibration_p95"]
        r["alpha_exp_status"] = alpha_exp_score["status"]
        r["alpha_exp_robust_z"] = alpha_exp_score["robust_z"]
        r["alpha_exp_calibration_p95"] = alpha_exp_score["calibration_p95"]

        alpha_id_jump = alpha_id_score["status"] == "elevated" and float(alpha_id_score["robust_z"]) >= 3.5
        alpha_exp_jump = alpha_exp_score["status"] == "elevated" and float(alpha_exp_score["robust_z"]) >= 3.5
        if alpha_id_jump and str(r.get("status")) in {"within_reconstruction_noise", "scattered_or_uncertain", "elevated_but_uncertain"}:
            r["status"] = "alpha_id_jump_candidate"
        if alpha_exp_jump and not alpha_id_jump and str(r.get("status")) in {"coherent_jump_candidate", "alpha_id_jump_candidate"}:
            r["status"] = "expression_dominated"
        if r.get("pair_type") == "adjacent" and (alpha_id_jump or alpha_exp_jump):
            events.append({
                "pair_id": r.get("pair_id"),
                "pose_bin": pose,
                "photo_a": r.get("photo_a"),
                "photo_b": r.get("photo_b"),
                "date_a": r.get("date_a"),
                "date_b": r.get("date_b"),
                "alpha_id_l2": float(aid),
                "alpha_id_status": alpha_id_score["status"],
                "alpha_id_robust_z": alpha_id_score["robust_z"],
                "alpha_exp_l2": float(aexp or 0.0),
                "alpha_exp_status": alpha_exp_score["status"],
                "alpha_exp_robust_z": alpha_exp_score["robust_z"],
                "interpretation": "alpha_id/alpha_exp calibrated jump candidate; not an identity verdict",
            })
    return {"schema": ALPHA_SCHEMA, "event_count": len(events), "events": events}
