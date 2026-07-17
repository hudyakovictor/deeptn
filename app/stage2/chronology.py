from __future__ import annotations
from collections import defaultdict
from datetime import date
import math
import numpy as np

def _days(a: str | None, b: str | None) -> int | None:
    if not a or not b: return None
    try:
        da=date.fromisoformat(str(a)[:10]); db=date.fromisoformat(str(b)[:10]); return abs((db-da).days)
    except Exception:
        return None

def _robust(vals: list[float]) -> tuple[float,float,float]:
    arr=np.asarray([v for v in vals if np.isfinite(v)],dtype=float)
    if arr.size==0: return 0.0,0.0,0.0
    med=float(np.median(arr)); mad=float(np.median(np.abs(arr-med))); p95=float(np.percentile(arr,95)); return med,mad,p95

def apply_chronology_rate_flags(rows: list[dict]) -> dict[str,dict[str,float]]:
    refs={}; by=defaultdict(list)
    for r in rows:
        if r.get('pair_type')=='adjacent': by[r['pose_bin']].append(r)
    for pose,group in by.items():
        rates=[]; coherent=[]
        for r in group:
            d=_days(r.get('date_a'),r.get('date_b')); eff=max(1,d or 1)
            weighted=float(r.get('p95_point_z',0.0))*max(float(r.get('coherent_motion_fraction',0.0)),0.1)/math.sqrt(eff)
            rates.append(weighted); coherent.append(float(r.get('coherent_motion_fraction',0.0)))
            r['days_delta']=d if d is not None else -1; r['time_weighted_jump_rate']=weighted
        med,mad,p95=_robust(rates); cmed,cmad,cp95=_robust(coherent)
        refs[pose]={'rate_median':med,'rate_mad':mad,'rate_p95':p95,'coherence_median':cmed,'coherence_mad':cmad,'coherence_p95':cp95,'count':len(rates)}
        floor=max(1.4826*mad,0.05)
        seq=sorted(group,key=lambda x:(x.get('date_b') or '',x.get('pair_index',0)))
        for i,r in enumerate(seq):
            d=r.get('days_delta',-1); weighted=float(r.get('time_weighted_jump_rate',0.0)); pz=float(r.get('p95_point_z',0.0)); coh=float(r.get('coherent_motion_fraction',0.0)); sig=float(r.get('significant_point_fraction',0.0))
            rate_z=(weighted-med)/floor if floor>0 else 0.0
            r['chronology_rate_z']=rate_z; r['chronology_rate_status']='within_expected_rate'; r['chronology_rate_reason']=''
            same_day=(d==0 and pz>=4.5 and coh>=0.35)
            fast=(d is not None and 0<d<=60 and pz>=4.5 and sig>=0.15 and coh>=0.45 and rate_z>=3.0)
            medium=(d is not None and 60<d<=180 and pz>=5.5 and sig>=0.18 and coh>=0.5 and rate_z>=3.5)
            if r.get('status')=='expression_dominated':
                r['chronology_rate_status']='expression_excluded'; r['chronology_rate_reason']='expression_dominated'; continue
            if same_day:
                r['chronology_rate_status']='same_day_structural_conflict'; r['chronology_rate_reason']='same day but coherent structural shift above calibrated noise'
            elif fast or medium:
                r['chronology_rate_status']='rapid_change_candidate'; r['chronology_rate_reason']='coherent calibrated shift is large relative to the elapsed days'
            if r['chronology_rate_status'] in ('same_day_structural_conflict','rapid_change_candidate'):
                nxt=seq[i+1] if i+1 < len(seq) else None
                if nxt and float(nxt.get('p95_point_z',0.0))>=3.5:
                    r['chronology_rate_status']='persistent_rapid_change_candidate'; r['chronology_rate_reason']+='; post-change state remains elevated'; r['status']='persistent_rapid_change_candidate'
                elif r['status'] in ('coherent_jump_candidate','persistent_geometric_change'):
                    r['status']=r['chronology_rate_status']
            # Compatibility aliases for older consumers. Canonical public fields are chronology_rate_*.
            r['biological_rate_z']=r['chronology_rate_z']
            r['biological_rate_status']=r['chronology_rate_status']
            r['biological_reason']=r['chronology_rate_reason']
    return refs


def apply_biological_rate_flags(rows: list[dict]) -> dict[str,dict[str,float]]:
    """Deprecated compatibility alias; use apply_chronology_rate_flags."""
    return apply_chronology_rate_flags(rows)
