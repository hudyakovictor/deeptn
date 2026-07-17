"""Full metrics catalog restore checks."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from util.metrics_catalog import (
    NINE_BUCKETS, catalog_summary, metrics_for_bucket,
    recovered_metric_names, target_catalog_count, view_distribution,
)

def test_target_is_thousands_not_234():
    assert target_catalog_count() >= 2000
    s = catalog_summary()
    assert s["target_catalog_count"] == 2199
    assert s["stability_allowed"] >= 900

def test_nine_buckets_have_coverage_counts():
    d = view_distribution()
    single = d["single_by_bucket"]
    assert len(single) >= 9
    assert single["frontal"] >= 1800
    assert single["left_profile"] >= 1000

def test_selected_metrics_per_view_nonempty():
    for b in NINE_BUCKETS:
        sel = metrics_for_bucket(b, tier="selected")
        assert len(sel) >= 20, b

def test_recovered_names_gt_1000():
    names = recovered_metric_names()
    assert len(names) >= 1000

if __name__ == "__main__":
    test_target_is_thousands_not_234()
    test_nine_buckets_have_coverage_counts()
    test_selected_metrics_per_view_nonempty()
    test_recovered_names_gt_1000()
    print("ALL METRICS CATALOG RESTORE TESTS PASSED")
    print(catalog_summary())
