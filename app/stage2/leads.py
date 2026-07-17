from __future__ import annotations
import json,re
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any
REGIONS=("orbit","brow","eyebrow","temporal","zygoma","cheekbone","cheek_soft","nose_bridge","nose_wing","nose","chin","jaw_angle","jaw","forehead","ligament_orbital","ligament_zygomatic","palpebral","lid","malar","submalar")

def _date(v: str | None) -> str | None:
    m=re.search(r"(19|20)\d{2}[-_]\d{2}[-_]\d{2}",str(v or ""))
    return m.group(0).replace("_","-") if m else None

def _load(root: Path, name: str) -> dict[str, Any]:
    try:
        return json.loads((root/name).read_text(encoding="utf-8"))
    except Exception:
        return {}

def load_leads(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {"status":"not_provided","dates":{},"metrics":[],"regions":[],"coverage":[]}
    root=path/"final_inference" if (path/"final_inference").is_dir() else path
    dates=defaultdict(lambda:{"photos":set(),"sources":set(),"events":set(),"metrics":set(),"regions":set(),"priority":0})
    metrics=Counter(); regions=Counter(); sources=Counter()
    def add(d, src, photo=None, event=None, ms=(), rs=(), weight=1):
        d=_date(d or photo)
        if not d: return
        x=dates[d]; x["sources"].add(src); x["priority"]+=weight; sources[src]+=1
        if photo: x["photos"].add(str(photo))
        if event: x["events"].add(str(event))
        for m in ms:
            m=str(m); x["metrics"].add(m); metrics[m]+=1
            low=m.lower()
            for token in REGIONS:
                if token in low:
                    x["regions"].add(token); regions[token]+=1
        for r in rs:
            if r:
                x["regions"].add(str(r)); regions[str(r)]+=1
    for e in _load(root,"top_identity_breaks.json").get("entries",[]):
        zs=[z.get("name") for z in e.get("calibration_features",{}).get("mesh_zones",[]) if z.get("raw_error") is not None]
        add(e.get("date_str"),"identity",e.get("photo_id"),ms=e.get("calibration_features",{}).get("exceeded_metrics",[]),rs=zs,weight=4)
    for e in _load(root,"top_chrono_breaks.json").get("entries",[]):
        for side in ("a","b"):
            add(e.get(f"photo_id_{side}"),"chronology",e.get(f"photo_id_{side}"),ms=e.get("exceeded_metrics",[]),weight=3)
    for e in _load(root,"top_evidence_packets.json").get("entries",[]):
        add(e.get("date_str"),"evidence_packet",e.get("photo_id"),e.get("primary_hypothesis"),weight=4)
    for e in _load(root,"chronology_events.json").get("events",[]):
        for event in e.get("event_types",[]): add(e.get("date_str"),"event",e.get("photo_id"),event,weight=1)
    for model in _load(root,"bucket_reference_models.json").get("models",{}).values():
        for m in model.get("envelopes",{}): metrics[m]+=1
    coverage=[]
    for m,n in metrics.most_common():
        low=m.lower(); family="point_motion_vector"; level="direct_analogue"
        if any(x in low for x in ("texture","synthetic","silicone","mask")): family="texture"; level="pending_texture"
        elif any(x in low for x in ("normal","flatness")): family="normal_direction_and_planarity"
        elif any(x in low for x in ("span","distance","aperture")): family="local_span_and_relative_distance"
        elif any(x in low for x in ("bbox","volume","surface_area")): family="local_bbox_area_volume"
        elif any(x in low for x in ("convexity","curvature","ellipse","dispersion","plane_residual")): family="local_shape_descriptor"
        coverage.append({"legacy_metric":m,"occurrences":n,"new_family":family,"coverage":level})
    out_dates={d:{k:(sorted(v) if isinstance(v,set) else v) for k,v in x.items()} for d,x in sorted(dates.items())}
    return {
        "schema_version":"deeputin-prior-leads-v1",
        "status":"loaded",
        "source_root":str(root),
        "policy":"audit targets only; never ground truth or threshold tuning",
        "date_count":len(out_dates),
        "metric_count":len(metrics),
        "sources":dict(sources),
        "dates":out_dates,
        "regions":[{"name":k,"occurrences":v} for k,v in regions.most_common()],
        "metrics":[m for m,_ in metrics.most_common()],
        "coverage":coverage,
    }

def pair_leads(reg: dict[str, Any], date_a: str | None, date_b: str | None) -> dict[str, Any]:
    xs=[reg.get("dates",{}).get(d) for d in (date_a,date_b) if reg.get("dates",{}).get(d)]
    if not xs:
        return {"lead_overlap":False,"lead_priority":0,"lead_regions":"","lead_events":"","lead_metric_count":0}
    return {
        "lead_overlap":True,
        "lead_priority":sum(int(x["priority"]) for x in xs),
        "lead_regions":"|".join(sorted({v for x in xs for v in x["regions"]})),
        "lead_events":"|".join(sorted({v for x in xs for v in x["events"]})),
        "lead_metric_count":len({v for x in xs for v in x["metrics"]}),
    }
