"""ITER4 unit tests: pose buckets, letterbox, cache, fail-closed metrics."""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from util.pose_buckets import (  # noqa: E402
    ALL_BUCKETS,
    classify_pose_bucket,
    classify_pose_bucket_from_yaw_ranges,
    load_pose_yaw_ranges,
    normalize_bucket_name,
    visible_face_side_from_yaw,
)
from util.letterbox import (  # noqa: E402
    FACE_CROP_HEIGHT,
    FACE_CROP_WIDTH,
    crop_bbox_with_margin,
    letterbox_meta,
    resize_letterbox,
    should_reextract_face_crop,
)
from util.extraction_cache import (  # noqa: E402
    ExtractionCache,
    content_hash_image_array,
    make_cache_key,
)
from util.selected_metrics import (  # noqa: E402
    assert_metrics_present,
    merge_metric_dicts_fail_closed,
    select_metrics,
)
from util.extraction import (  # noqa: E402
    build_extraction_record,
    build_face_crop_letterbox,
    cache_key_for_image,
)


def test_pose_ranges_cover_all_buckets():
    ranges = load_pose_yaw_ranges(reload=True)
    for b in ALL_BUCKETS:
        assert b in ranges
        assert ranges[b]["min"] <= ranges[b]["max"]


def test_classify_pose_buckets_boundaries():
    assert classify_pose_bucket_from_yaw_ranges(0.0) == "frontal"
    assert classify_pose_bucket_from_yaw_ranges(6.0) == "frontal"
    assert classify_pose_bucket_from_yaw_ranges(6.1).startswith("right_threequarter_light")
    assert classify_pose_bucket_from_yaw_ranges(-6.1).startswith("left_threequarter_light")
    assert classify_pose_bucket_from_yaw_ranges(30.0) == "right_threequarter_mid"
    assert classify_pose_bucket_from_yaw_ranges(-50.0) == "left_threequarter_deep"
    assert classify_pose_bucket_from_yaw_ranges(70.0) == "right_profile"
    assert classify_pose_bucket_from_yaw_ranges(-90.0) == "left_profile"


def test_false_profile_guard():
    # needs_manual_review + near-frontal pitch/roll demotes profile -> deep 3/4
    b = classify_pose_bucket(80.0, pitch_deg=5.0, roll_deg=3.0, needs_manual_review=True)
    assert b == "right_threequarter_deep"


def test_normalize_aliases():
    assert normalize_bucket_name("front") == "frontal"
    assert normalize_bucket_name("profile_left") == "left_profile"
    assert normalize_bucket_name("weird") == "unclassified"
    assert visible_face_side_from_yaw(-20.0) == "left"
    assert visible_face_side_from_yaw(20.0) == "right"


def test_letterbox_preserves_aspect():
    try:
        import cv2  # noqa: F401
    except Exception:
        print("SKIP letterbox: no cv2")
        return
    # wide crop 200x100 -> letterbox 424x500 should not stretch to full height content
    img = np.zeros((100, 200, 3), dtype=np.uint8)
    img[:, :] = (0, 255, 0)
    out, meta = resize_letterbox(img, FACE_CROP_WIDTH, FACE_CROP_HEIGHT)
    assert out.shape == (FACE_CROP_HEIGHT, FACE_CROP_WIDTH, 3)
    assert meta.content_w == FACE_CROP_WIDTH or meta.content_h < FACE_CROP_HEIGHT
    # content aspect ~= source aspect
    src_aspect = 200 / 100
    out_aspect = meta.content_w / max(meta.content_h, 1)
    assert abs(src_aspect - out_aspect) < 0.05
    # padded regions remain black
    if meta.offset_y > 0:
        assert out[0, FACE_CROP_WIDTH // 2].sum() == 0


def test_should_reextract_policy():
    assert should_reextract_face_crop(None) is True
    bad = np.zeros((100, 100, 3), dtype=np.uint8)
    assert should_reextract_face_crop(bad) is True
    good = np.zeros((FACE_CROP_HEIGHT, FACE_CROP_WIDTH, 3), dtype=np.uint8)
    assert should_reextract_face_crop(good, meta=None) is True
    assert should_reextract_face_crop(good, meta={"method": "stretch"}) is True
    assert should_reextract_face_crop(good, meta={"method": "letterbox"}) is False


def test_crop_bbox_and_build_face_crop():
    try:
        import cv2  # noqa: F401
    except Exception:
        print("SKIP crop: no cv2")
        return
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    img[100:300, 150:350] = 128
    crop, meta, xyxy = build_face_crop_letterbox(img, {"x": 150, "y": 100, "w": 200, "h": 200})
    assert crop.shape == (FACE_CROP_HEIGHT, FACE_CROP_WIDTH, 3)
    assert meta["method"] == "letterbox"
    assert xyxy[2] > xyxy[0]


def test_content_hash_cache_roundtrip(tmp_path=None):
    img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    h1 = content_hash_image_array(img)
    h2 = content_hash_image_array(img.copy())
    assert h1 == h2
    img2 = img.copy(); img2[0, 0, 0] = np.uint8((int(img2[0, 0, 0]) + 1) % 256)
    assert content_hash_image_array(img2) != h1
    key = make_cache_key(image_hash=h1, backbone="resnet50", identity_only=True)
    cache = ExtractionCache("/tmp/iter4_cache_test", max_entries=5)
    cache.set(key, {"alpha_id": np.zeros(80, dtype=np.float32), "note": "x"})
    loaded = cache.get(key)
    assert loaded is not None
    assert np.allclose(loaded["alpha_id"], 0)


def test_selected_metrics_fail_closed():
    src = {"a": 1.0, "b": None, "c": 3.0}
    res = select_metrics(src, ["a", "b", "c"], allow_fail_open=False)
    assert res.ok is False
    assert "b" in res.nulls
    res2 = select_metrics(src, ["a", "b"], allow_fail_open=True, fill_value=0.0)
    assert res2.ok is False  # never silent success
    assert res2.fail_open_attempted is True
    assert res2.values["b"] == 0.0
    ok = select_metrics({"a": 1.0, "b": 2.0}, ["a", "b"])
    assert ok.ok is True
    vals = assert_metrics_present({"x": 1}, ["x"])
    assert vals["x"] == 1
    try:
        assert_metrics_present({}, ["x"])
        raise AssertionError("expected raise")
    except ValueError:
        pass
    m = merge_metric_dicts_fail_closed({"a": 1}, {"a": 2, "b": 3}, required_keys=("a", "b"))
    assert m.ok and m.values["a"] == 2


def test_build_extraction_record():
    try:
        import cv2  # noqa: F401
        has_cv2 = True
    except Exception:
        has_cv2 = False
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    if has_cv2:
        img[50:200, 50:200] = 200
        rec = build_extraction_record(
            yaw_deg=-30.0,
            pitch_deg=2.0,
            roll_deg=1.0,
            image=img,
            bbox={"x": 50, "y": 50, "w": 150, "h": 150},
            existing_face_crop=None,
            metrics={"bone_err": 0.1, "missing": None},
            required_metric_keys=("bone_err",),
        )
        assert rec.pose_bucket == "left_threequarter_mid"
        assert rec.face_crop is not None
        assert rec.letterbox_meta["method"] == "letterbox"
        assert rec.metrics_ok is True
        assert rec.image_hash is not None
    # fail-closed metrics without image
    rec2 = build_extraction_record(
        yaw_deg=0.0,
        metrics={"a": None},
        required_metric_keys=("a", "b"),
    )
    assert rec2.pose_bucket == "frontal"
    assert rec2.metrics_ok is False


def test_modules_parse():
    for rel in (
        "util/pose_buckets.py",
        "util/letterbox.py",
        "util/extraction_cache.py",
        "util/selected_metrics.py",
        "util/extraction.py",
    ):
        ast.parse((ROOT / rel).read_text())


if __name__ == "__main__":
    test_pose_ranges_cover_all_buckets()
    test_classify_pose_buckets_boundaries()
    test_false_profile_guard()
    test_normalize_aliases()
    test_letterbox_preserves_aspect()
    test_should_reextract_policy()
    test_crop_bbox_and_build_face_crop()
    test_content_hash_cache_roundtrip()
    test_selected_metrics_fail_closed()
    test_build_extraction_record()
    test_modules_parse()
    print("ALL ITER4 UNIT TESTS PASSED")
