"""Final iteration: 50 deep integration/semantic analyses of the new version.

Complements tests/audit_50_new_version.py (structural/unit) with end-to-end and
cross-module invariants. Exit 0 only if all 50 pass.
"""
from __future__ import annotations

import json, math, sys, tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RESULTS = []

def check(name, fn):
    try:
        v = fn()
        if v is False:
            raise AssertionError("returned False")
        RESULTS.append((name, "PASS", ""))
    except Exception as e:
        RESULTS.append((name, "FAIL", f"{type(e).__name__}: {e}"))

def req(x, msg="condition failed"):
    if not x:
        raise AssertionError(msg)

rng = np.random.default_rng(42)
CLOUD = rng.normal(0, 1, (300, 3)) * np.array([8.0, 10.0, 4.0])

# ---------- alignment (1-7) ----------

def a01_umeyama_identity():
    from util.alignment import rigid_umeyama
    r = rigid_umeyama(CLOUD, CLOUD)
    req(r.residual_after < 1e-9, f"residual {r.residual_after}")
    req(np.allclose(r.rotation, np.eye(3), atol=1e-6))

def a02_umeyama_rotation_recovery():
    from util.alignment import rigid_umeyama, euler_to_rotation_matrix
    R = euler_to_rotation_matrix(np.deg2rad(np.array([10.0, 25.0, -15.0])))
    src = CLOUD @ R.T + np.array([3.0, -2.0, 5.0])
    r = rigid_umeyama(src, CLOUD)
    req(r.residual_after < 1e-6, f"residual {r.residual_after}")

def a03_umeyama_no_scale_locked():
    from util.alignment import rigid_umeyama
    r = rigid_umeyama(CLOUD * 2.0, CLOUD, allow_scale=False)
    req(abs(r.scale - 1.0) < 1e-9, f"scale {r.scale}")

def a04_umeyama_scale_recovery():
    from util.alignment import rigid_umeyama
    r = rigid_umeyama(CLOUD * 2.0, CLOUD, allow_scale=True)
    req(abs(r.scale - 0.5) < 1e-3, f"scale {r.scale}")
    req(r.residual_after < 1e-6)

def a05_euler_orthonormal():
    from util.alignment import euler_to_rotation_matrix
    R = euler_to_rotation_matrix(np.deg2rad(np.array([31.0, -47.0, 12.0])))
    req(np.allclose(R @ R.T, np.eye(3), atol=1e-8))
    req(abs(float(np.linalg.det(R)) - 1.0) < 1e-8)

def a06_canonical_angles_shape():
    from util.alignment import canonical_angles_deg_for_bucket
    from util.pose_buckets import ALL_BUCKETS, CANONICAL_YAW_BY_VIEW_GROUP
    for b in ALL_BUCKETS:
        a = np.asarray(canonical_angles_deg_for_bucket(b), dtype=float).reshape(-1)
        req(a.size == 3 and np.isfinite(a).all(), b)
        req(abs(float(a[1]) - CANONICAL_YAW_BY_VIEW_GROUP[b]) < 1e-6, f"{b} yaw {a[1]}")

def a07_align_meshes_shared_identity():
    from util.alignment import align_meshes_shared
    r = align_meshes_shared(CLOUD, CLOUD)
    req(r.residual_after < 1e-9)

# ---------- compare primitives (8-15) ----------

def c08_shared_vertex_indices():
    from util.compare import shared_vertex_indices
    a = np.array([True, True, False, True])
    b = np.array([True, False, False, True])
    req(shared_vertex_indices(a, b).tolist() == [0, 3])

def c09_geodesic_zero():
    from util.compare import geodesic_pose_distance
    req(abs(geodesic_pose_distance(np.eye(3), np.eye(3))) < 1e-9)

def c10_geodesic_ninety():
    from util.compare import geodesic_pose_distance
    from util.alignment import euler_to_rotation_matrix
    R = euler_to_rotation_matrix(np.deg2rad(np.array([0.0, 90.0, 0.0])))
    req(abs(geodesic_pose_distance(np.eye(3), R) - 90.0) < 1e-6)

def c11_pose_delta_none():
    from util.compare import pose_delta_deg
    req(pose_delta_deg(None, np.array([0, 10, 0])) == 0.0)

def c12_id_cosine_identical():
    from util.compare import id_params_cosine_distance
    v = rng.normal(0, 1, 80)
    req(abs(id_params_cosine_distance(v, v)) < 1e-12)

def c13_id_cosine_orthogonal():
    from util.compare import id_params_cosine_distance
    a = np.zeros(80); a[:40] = 1.0
    b = np.zeros(80); b[40:] = 1.0
    req(abs(id_params_cosine_distance(a, b) - 1.0) < 1e-12)

def c14_score_identical_pair():
    from util.compare import score_aligned_pair
    e1, s1, e2, s2 = score_aligned_pair(CLOUD, CLOUD, np.ones(len(CLOUD)))
    req(e1 < 1e-12 and e2 < 1e-12 and abs(s1 - 1) < 1e-9 and abs(s2 - 1) < 1e-9)

def c15_score_monotonic_deform():
    from util.compare import score_aligned_pair
    w = np.ones(len(CLOUD))
    small = CLOUD + rng.normal(0, 0.05, CLOUD.shape)
    big = CLOUD + rng.normal(0, 0.50, CLOUD.shape)
    e_small = score_aligned_pair(small, CLOUD, w)[0]
    e_big = score_aligned_pair(big, CLOUD, w)[0]
    req(e_big > e_small > 0, f"{e_small} vs {e_big}")

# ---------- compare e2e on BFM-size mesh (16-18) ----------

N_BFM = 35709
BASE = rng.normal(0, 1, (N_BFM, 3)) * np.array([8.0, 10.0, 4.0])
NORMALS = np.zeros((N_BFM, 3)); NORMALS[:, 2] = 1.0

def _mesh(verts, person, photo):
    from util.compare import PairCompareInput
    return PairCompareInput(
        vertices=verts, normals=NORMALS.copy(),
        angles_deg=np.array([0.0, 0.0, 0.0]), pose_bucket="frontal",
        alpha_id=rng.normal(0, 1, 80), person_id=person, photo_id=photo,
    )

def e16_compare_pair_same_mesh():
    from util.compare import compare_pair
    r = compare_pair(_mesh(BASE, "p1", "a"), _mesh(BASE.copy(), "p1", "b"))
    req(r.status == "ok", r.status)
    req(r.shared_count > 1000, r.shared_count)
    req(r.raw_geometry_error is not None and r.raw_geometry_error < 1e-9, f"{r.raw_geometry_error}")
    req(r.bounded_similarity_score is not None and r.bounded_similarity_score > 0.999)

def e17_compare_pair_detects_deform():
    from util.compare import compare_pair
    r_same = compare_pair(_mesh(BASE, "p1", "a"), _mesh(BASE.copy(), "p1", "b"))
    deformed = BASE + rng.normal(0, 0.4, BASE.shape)
    r_diff = compare_pair(_mesh(BASE, "p1", "a"), _mesh(deformed, "p2", "c"))
    req(r_diff.status == "ok", r_diff.status)
    req(r_diff.raw_geometry_error > max(r_same.raw_geometry_error, 1e-9) * 100,
        f"{r_same.raw_geometry_error} vs {r_diff.raw_geometry_error}")

def e18_compare_pair_shape_mismatch():
    from util.compare import compare_pair
    r = compare_pair(_mesh(BASE, "p1", "a"), _mesh(BASE[:100], "p1", "b"))
    req(r.status != "ok", r.status)

# ---------- calibration (19-27) ----------

def k19_mad():
    from util.calibration import mad
    req(mad([1.0, 1.0, 1.0]) == 0.0)
    req(abs(mad([1.0, 2.0, 3.0]) - 1.0) < 1e-12)

def k20_effective_sample_size():
    from util.calibration import effective_sample_size
    req(effective_sample_size(100, 0.0) == 100.0)
    req(effective_sample_size(100, 0.9) < 10.0)
    req(effective_sample_size(0) == 0.0)

def k21_bootstrap_ci():
    from util.calibration import block_bootstrap_ci
    vals = list(rng.normal(5, 0.5, 60))
    ci = block_bootstrap_ci(vals)
    req(ci["lo"] <= ci["mean"] <= ci["hi"], str(ci))
    req(ci["n"] == 60 and ci["n_eff"] > 0)

def k22_linear_snr_floor():
    from util.calibration import linear_snr
    req(linear_snr(0.01, 0.05) == 0.0)
    req(linear_snr(0.15, 0.05) > 1.0)

def k23_quality_bands():
    from util.calibration import quality_band_from_score
    req(quality_band_from_score(0.8) == "high")
    req(quality_band_from_score(0.5) == "mid")
    req(quality_band_from_score(0.4) == "low")
    req(quality_band_from_score(0.1) == "reject")
    req(quality_band_from_score(None) == "unknown")

def k24_health_status():
    from util.calibration import metric_health_status
    req(metric_health_status(0.1, 0.01, 2) == "insufficient")
    req(metric_health_status(0.1, 0.01, 20) == "stable")
    req(metric_health_status(0.9, 0.5, 20) == "replace")

def k25_person_baseline():
    from util.calibration import build_person_baselines, apply_person_baseline
    bl = build_person_baselines([("p1", "frontal", 0.10), ("p1", "frontal", 0.20), ("p2", "frontal", 0.30)])
    req(abs(bl["p1"]["frontal"] - 0.15) < 1e-12, str(bl))
    req(abs(apply_person_baseline(0.5, "p1", "p1", "frontal", bl) - 0.35) < 1e-12)
    req(apply_person_baseline(0.5, "p1", "p2", "frontal", bl) == 0.5)

def k26_cell_key_normalized():
    from util.calibration import pose_quality_cell_key
    req(pose_quality_cell_key("profile-left", "high") == "left_profile|high")

def k27_noise_model():
    from util.calibration import NoiseModel, NoiseObservation
    nm = NoiseModel()
    nm.extend([
        NoiseObservation("a", "b", "p1", "frontal", 2.0, {"orbit_L": 0.02, "chin": 0.03}),
        NoiseObservation("c", "d", "p1", "frontal", 4.0, {"orbit_L": 0.03, "chin": 0.05}),
        NoiseObservation("e", "f", "p2", "frontal", 3.0, {"orbit_L": 0.025}),
    ])
    req("orbit_L" in nm.zone_profiles and "chin" in nm.zone_profiles)
    p = nm.zone_profiles["orbit_L"]
    req(p.count == 3 and p.predict_noise(5.0) > p.mean)
    req(nm.cell_profiles, "cell profiles empty")

# ---------- cache (28-30) ----------

def h28_cache_roundtrip():
    from util.extraction_cache import ExtractionCache
    with tempfile.TemporaryDirectory() as td:
        c = ExtractionCache(td)
        arr = rng.normal(0, 1, (5, 3))
        c.set("k1", {"verts": arr, "note": "x", "skipme": None})
        out = c.get("k1")
        req(out is not None and np.allclose(out["verts"], arr))
        req(str(out["note"]) == "x")
        req(c.get("missing") is None and c.has("k1"))

def h29_cache_eviction():
    from util.extraction_cache import ExtractionCache
    with tempfile.TemporaryDirectory() as td:
        c = ExtractionCache(td, max_entries=2)
        for i in range(4):
            c.set(f"k{i}", {"v": np.array([i])})
        req(not c.has("k0"), "oldest not evicted")
        req(c.has("k3"), "newest must remain")

def h30_image_hash_dtype_sensitive():
    from util.extraction_cache import content_hash_image_array
    a8 = np.zeros((4, 4), dtype=np.uint8)
    a16 = np.zeros((2, 4), dtype=np.uint16)  # same byte count, different dtype/shape
    req(content_hash_image_array(a8) != content_hash_image_array(a16))

# ---------- verdict (31-37) ----------

def v31_verdict_fail_closed_and_signal():
    from util.verdict import EvidenceBundle, GeometryEvidenceMode, render_verdict
    empty = render_verdict(EvidenceBundle())
    req(abs(sum(empty.probabilities.values()) - 1.0) < 1e-9)
    strong = render_verdict(EvidenceBundle(geometry_snr=5.0, geometry_mode=GeometryEvidenceMode.CALIBRATED, shared_vertex_count=5000))
    req(strong.probabilities["H2"] > strong.probabilities["H0"], str(strong.probabilities))

def v32_geometry_likelihood_monotonic():
    from util.verdict import geometry_likelihoods
    l_low = geometry_likelihoods(0.0)
    l_high = geometry_likelihoods(4.0)
    req(l_high["H2"] > l_low["H2"] and l_low["H0"] > l_high["H0"])

def v33_geometry_unavailable_noninformative():
    from util.verdict import geometry_likelihoods, GeometryEvidenceMode
    l = geometry_likelihoods(None)
    req(l == {"H0": 1.0, "H1": 1.0, "H2": 1.0})
    l2 = geometry_likelihoods(3.0, mode=GeometryEvidenceMode.UNAVAILABLE)
    req(l2 == {"H0": 1.0, "H1": 1.0, "H2": 1.0})

def v34_texture_likelihood_bounds():
    from util.verdict import texture_likelihoods
    for p in (None, 0.0, 0.5, 1.0):
        l = texture_likelihoods(p, reliability=0.8)
        req(all(v > 0 and math.isfinite(v) for v in l.values()), f"{p}: {l}")

def v35_posterior_noninformative_keeps_priors():
    from util.verdict import update_posteriors_log
    pri = {"H0": 0.5, "H1": 0.05, "H2": 0.45}
    post = update_posteriors_log(pri, [{"H0": 1.0, "H1": 1.0, "H2": 1.0}])
    req(all(abs(post[k] - pri[k]) < 1e-9 for k in pri), str(post))

def v36_posterior_zero_likelihood():
    from util.verdict import update_posteriors_log
    post = update_posteriors_log({"H0": 0.5, "H1": 0.25, "H2": 0.25}, [{"H0": 0.0, "H1": 1.0, "H2": 1.0}])
    req(abs(sum(post.values()) - 1.0) < 1e-9 and all(math.isfinite(v) for v in post.values()))
    req(post["H0"] < 1e-6, str(post))

def v37_fuzzy_insufficient():
    from util.verdict import fuzzy_label_from_evidence, FuzzyLabel
    lbl = fuzzy_label_from_evidence({"H0": 0.34, "H1": 0.33, "H2": 0.33}, None, None, insufficient=True)
    req(lbl == FuzzyLabel.INSUFFICIENT_DATA, str(lbl))

# ---------- report (38-40) ----------

def r38_report_roundtrip():
    from util.report import build_report, save_report_json, load_report_json, content_fingerprint
    rep = build_report(
        photo_a="a.jpg", photo_b="b.jpg",
        compare={"raw_geometry_error": 0.01, "bone_raw_geometry_error": 0.008},
        verdict={"status": "uncertain", "probabilities": {"H0": 0.4, "H1": 0.1, "H2": 0.5}},
    )
    req(rep.schema_version and rep.acceptance, "missing schema/acceptance")
    with tempfile.TemporaryDirectory() as td:
        p = save_report_json(rep, Path(td) / "r.json")
        loaded = load_report_json(p)
    req(loaded["pair"]["photo_a"] == "a.jpg")
    req(content_fingerprint(loaded) == content_fingerprint(json.loads(json.dumps(loaded))))

def r39_acceptance_rejects_bad_posteriors():
    from util.report import build_report
    rep = build_report(
        photo_a="a", photo_b="b",
        verdict={"status": "uncertain", "probabilities": {"H0": 0.5, "H1": 0.2, "H2": 0.2}},
    )
    req(rep.acceptance["posteriors_sum_1"]["pass"] is False, str(rep.acceptance.get("posteriors_sum_1")))

def r40_report_numpy_safe():
    from util.report import build_report, save_report_json
    rep = build_report(
        photo_a="a", photo_b="b",
        compare={"raw_geometry_error": np.float32(0.02)},
        verdict={"status": "uncertain", "probabilities": {"H0": np.float64(0.4), "H1": 0.1, "H2": 0.5}},
    )
    with tempfile.TemporaryDirectory() as td:
        save_report_json(rep, Path(td) / "r.json")

# ---------- catalog x policy integration (41-45) ----------

def p41_views_parse_to_known_buckets():
    from util.metrics_catalog import load_active_catalog_rows
    from util.legacy_metrics.catalog_specs import _buckets_from_views
    from util.legacy_metrics.common import ALL_BUCKETS
    known = set(ALL_BUCKETS)
    for r in load_active_catalog_rows():
        buckets = _buckets_from_views(r.get("views", ""))
        req(buckets and set(buckets) <= known, f"{r['metric_name']}: {buckets}")

def p42_bucket_specs_subset_consistent():
    from util.legacy_metrics.registry import all_specs, specs_for_bucket
    from util.legacy_metrics.policy import spec_allowed_for_bucket
    allk = {(s.name, s.implementation, s.scope) for s in all_specs()}
    for b in ("frontal", "left_profile", "right_threequarter_mid"):
        for s in specs_for_bucket(b):
            req((s.name, s.implementation, s.scope) in allk)
            req(spec_allowed_for_bucket(s, b), f"{b}:{s.name}")

def p43_profile_hides_wrong_side():
    from util.legacy_metrics.registry import specs_for_bucket
    bad = [s.name for s in specs_for_bucket("left_profile") if s.side in {"R", "B"}]
    req(not bad, f"leaked sides in left_profile: {bad[:5]}")

def p44_selected_names_exist_in_catalog():
    from util.metrics_catalog import NINE_BUCKETS, metrics_for_bucket, recovered_metric_names
    names = set(recovered_metric_names())
    for b in NINE_BUCKETS:
        missing = [m for m in metrics_for_bucket(b) if m not in names]
        req(not missing, f"{b}: {missing[:5]}")

def p45_runtime_gate_blocks_bilateral_at_yaw():
    from util.legacy_metrics.policy import apply_runtime_confidence_gates, POSE_YAW_BILATERAL_OFF_DEG
    from util.legacy_metrics.types import MetricContext, MetricSpec, MetricValue
    spec = MetricSpec(name="test_bilateral_width", family="F0", group="g", zone="face", side="B",
                      buckets=("left_threequarter_light",))
    ctx = MetricContext(photo_id="x", image_path=Path("x"), pose_bucket="left_threequarter_light",
                        yaw_deg=float(POSE_YAW_BILATERAL_OFF_DEG + 5), pitch_deg=0.0, roll_deg=0.0, recon=None,
                        vertices_raw=np.zeros((0, 3)), vertices_canon=np.zeros((0, 3)),
                        vertices_shape_neutral=None, normals_raw=None, normals_canon=None,
                        normals_shape_neutral=None, triangles=np.zeros((0, 3), int),
                        annotation_groups=[], macro_indices={}, landmarks_106=None)
    out = apply_runtime_confidence_gates(ctx, [MetricValue(spec=spec, value=1.0)])
    req(out and out[0].quality_gate == "blocked", str(out[0].quality_gate if out else None))

# ---------- legacy runner e2e + selection (46-47) ----------

def _legacy_ctx():
    from util.legacy_metrics.types import MetricContext
    from util.zones import MACRO_BONE_INDICES
    macro = {k: np.asarray(sorted(v), dtype=np.int64) for k, v in MACRO_BONE_INDICES.items()}
    tris = np.asarray([[i, i + 1, i + 2] for i in range(0, 300, 3)], dtype=np.int64)
    return MetricContext(
        photo_id="synt", image_path=Path("synt.jpg"), pose_bucket="frontal",
        yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, recon=None,
        vertices_raw=BASE.copy(), vertices_canon=BASE.copy(), vertices_shape_neutral=None,
        normals_raw=NORMALS.copy(), normals_canon=NORMALS.copy(), normals_shape_neutral=None,
        triangles=tris, annotation_groups=[], macro_indices=macro, landmarks_106=None,
    )

def l46_legacy_runner_produces_finite_values():
    from util.legacy_metrics.runner import compute_single_photo_metrics
    vals, errs = compute_single_photo_metrics(_legacy_ctx())
    req(len(vals) >= 100, f"only {len(vals)} values, errors={errs[:3]}")
    bad = [v.spec.name for v in vals if v.value is None or not math.isfinite(float(v.value))]
    req(not bad, f"non-finite: {bad[:5]}")

def l47_selection_keys_and_empty_config():
    from util.legacy_metrics.runner import compute_single_photo_metrics
    from util.legacy_metrics.selection import filter_selected, metric_keys_for_value
    vals, _ = compute_single_photo_metrics(_legacy_ctx())
    req(vals, "no values to select from")
    kept = filter_selected(vals, "frontal", selected={})
    req(len(kept) == len(vals), "empty config must keep all")
    keys = metric_keys_for_value(vals[0])
    req(vals[0].spec.name in keys)

# ---------- zones + visibility integration (48-50) ----------

def z48_zone_metrics_identity():
    from util.zones import compute_zone_metrics, summarize_bone_priority_metrics
    shared = np.arange(N_BFM, dtype=np.int64)
    zones = compute_zone_metrics(aligned_points_a=BASE, points_b=BASE, shared_indices=shared)
    req(zones, "no zones returned")
    ok = [z for z in zones if z.status == "ok"]
    req(len(ok) >= 5, f"too few ok zones: {len(ok)}")
    req(all(z.raw_error < 1e-9 for z in ok), "identity zones must have ~0 error")
    summary = summarize_bone_priority_metrics(zones)
    req(isinstance(summary, dict) and summary, "empty bone summary")

def z49_canthus_points():
    from util.zones import canthus_points_from_orbit
    orbit = rng.normal(0, 1, (40, 3))
    inner, outer = canthus_points_from_orbit(orbit, "L")
    req(np.asarray(inner).shape == (3,) and np.asarray(outer).shape == (3,))
    req(np.isfinite(inner).all() and np.isfinite(outer).all())

def z50_yaw_fade_and_uv_parity():
    from util.visibility import compute_vertex_visibility_from_normals, compute_triangle_visibility as tri_util
    from uv_module.visibility import compute_triangle_visibility as tri_uv
    verts = np.zeros((4, 3)); verts[:, 0] = [-2.0, -1.0, 1.0, 2.0]
    normals = np.tile(np.array([0.0, 0.0, 1.0]), (4, 1))
    r = compute_vertex_visibility_from_normals(verts, normals, use_zbuffer=False, angles_deg=[0.0, 70.0, 0.0])
    req(float(r.cosine_weights[0]) == 0.0 and float(r.cosine_weights[1]) == 0.0, "turning-away not zeroed at yaw 70")
    req(float(r.cosine_weights[2]) > 0.5 and float(r.cosine_weights[3]) > 0.5)
    tris = np.array([[0, 1, 2], [1, 2, 3]])
    tv1 = tri_util(BASE[:100], tris, use_zbuffer=False)
    tv2 = tri_uv(BASE[:100], tris, use_zbuffer=False)
    req(np.allclose(tv1, tv2, atol=1e-6), "util vs uv_module triangle visibility mismatch")

CHECKS = [
    ("01_umeyama_identity", a01_umeyama_identity),
    ("02_umeyama_rotation_recovery", a02_umeyama_rotation_recovery),
    ("03_umeyama_no_scale_locked", a03_umeyama_no_scale_locked),
    ("04_umeyama_scale_recovery", a04_umeyama_scale_recovery),
    ("05_euler_orthonormal", a05_euler_orthonormal),
    ("06_canonical_angles_all_buckets", a06_canonical_angles_shape),
    ("07_align_meshes_shared_identity", a07_align_meshes_shared_identity),
    ("08_shared_vertex_indices", c08_shared_vertex_indices),
    ("09_geodesic_zero", c09_geodesic_zero),
    ("10_geodesic_ninety", c10_geodesic_ninety),
    ("11_pose_delta_none", c11_pose_delta_none),
    ("12_id_cosine_identical", c12_id_cosine_identical),
    ("13_id_cosine_orthogonal", c13_id_cosine_orthogonal),
    ("14_score_identical_pair", c14_score_identical_pair),
    ("15_score_monotonic_deform", c15_score_monotonic_deform),
    ("16_compare_pair_same_mesh", e16_compare_pair_same_mesh),
    ("17_compare_pair_detects_deform", e17_compare_pair_detects_deform),
    ("18_compare_pair_shape_mismatch", e18_compare_pair_shape_mismatch),
    ("19_mad", k19_mad),
    ("20_effective_sample_size", k20_effective_sample_size),
    ("21_bootstrap_ci", k21_bootstrap_ci),
    ("22_linear_snr_floor", k22_linear_snr_floor),
    ("23_quality_bands", k23_quality_bands),
    ("24_health_status", k24_health_status),
    ("25_person_baseline", k25_person_baseline),
    ("26_cell_key_normalized", k26_cell_key_normalized),
    ("27_noise_model", k27_noise_model),
    ("28_cache_roundtrip", h28_cache_roundtrip),
    ("29_cache_eviction", h29_cache_eviction),
    ("30_image_hash_dtype_sensitive", h30_image_hash_dtype_sensitive),
    ("31_verdict_fail_closed_and_signal", v31_verdict_fail_closed_and_signal),
    ("32_geometry_likelihood_monotonic", v32_geometry_likelihood_monotonic),
    ("33_geometry_unavailable_noninformative", v33_geometry_unavailable_noninformative),
    ("34_texture_likelihood_bounds", v34_texture_likelihood_bounds),
    ("35_posterior_noninformative", v35_posterior_noninformative_keeps_priors),
    ("36_posterior_zero_likelihood", v36_posterior_zero_likelihood),
    ("37_fuzzy_insufficient", v37_fuzzy_insufficient),
    ("38_report_roundtrip", r38_report_roundtrip),
    ("39_acceptance_rejects_bad_posteriors", r39_acceptance_rejects_bad_posteriors),
    ("40_report_numpy_safe", r40_report_numpy_safe),
    ("41_views_parse_to_known_buckets", p41_views_parse_to_known_buckets),
    ("42_bucket_specs_subset_consistent", p42_bucket_specs_subset_consistent),
    ("43_profile_hides_wrong_side", p43_profile_hides_wrong_side),
    ("44_selected_names_exist_in_catalog", p44_selected_names_exist_in_catalog),
    ("45_runtime_gate_blocks_bilateral", p45_runtime_gate_blocks_bilateral_at_yaw),
    ("46_legacy_runner_finite_values", l46_legacy_runner_produces_finite_values),
    ("47_selection_keys_empty_config", l47_selection_keys_and_empty_config),
    ("48_zone_metrics_identity", z48_zone_metrics_identity),
    ("49_canthus_points", z49_canthus_points),
    ("50_yaw_fade_and_uv_parity", z50_yaw_fade_and_uv_parity),
]

for name, fn in CHECKS:
    check(name, fn)

for n, s, d in RESULTS:
    print(f"{s:4s} {n}" + (f" :: {d}" if d else ""))
p = sum(s == "PASS" for _, s, _ in RESULTS)
f = len(RESULTS) - p
print(f"SUMMARY total={len(RESULTS)} pass={p} fail={f}")
Path(ROOT / "audit_final_50.json").write_text(json.dumps([{"name": n, "status": s, "detail": d} for n, s, d in RESULTS], indent=2))
raise SystemExit(1 if f else 0)
