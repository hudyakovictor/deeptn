from __future__ import annotations
import warnings
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from .core import Record,robust_rigid_align
NAMES=("centroid_dx","centroid_dy","centroid_dz","span_lateral","span_vertical","span_depth","bbox_area","bbox_volume","radial_dispersion","plane_residual","normal_angle","curvature","planarity")

def _neighbors(template: np.ndarray, k: int = 8) -> np.ndarray:
    d=np.sum((template[:,None]-template[None,:])**2,axis=2)
    return np.argsort(d,axis=1)[:,1:k+1]

def _one(points: np.ndarray, ids: np.ndarray):
    p=points[ids]; c=p.mean(0); q=p-c; span=np.ptp(p,axis=0)
    area=2*(span[0]*span[1]+span[0]*span[2]+span[1]*span[2]); volume=float(np.prod(span))
    rad=float(np.mean(np.linalg.norm(q,axis=1))); cov=q.T@q/max(len(q)-1,1); ev,vec=np.linalg.eigh(cov); s=max(float(ev.sum()),1e-9)
    normal=vec[:,0]; plane=float(np.std(q@normal)); curv=float(ev[0]/s); plan=float((ev[1]-ev[0])/max(ev[2],1e-9))
    return c,span,float(area),volume,rad,plane,normal,curv,plan

def local_pair_descriptors(a: Record, b: Record, template: np.ndarray) -> dict[str, np.ndarray | str]:
    vis=np.asarray(a.visible134,bool)&np.asarray(b.visible134,bool); out=np.full((134,len(NAMES)),np.nan,np.float32)
    if vis.sum()<30: return {"status":"insufficient_visibility","values":out}
    _,r,t,_=robust_rigid_align(b.ldm134[vis],a.ldm134[vis]); pb=b.ldm134@r+t; neigh=_neighbors(template)
    for i,ns in enumerate(neigh):
        ids=np.array([i,*ns]); ids=ids[vis[ids]]
        if len(ids)<5: continue
        A=_one(a.ldm134,ids); B=_one(pb,ids); eps=1e-7
        out[i,:3]=np.abs(B[0]-A[0]); out[i,3:6]=np.abs(B[1]-A[1])/(np.abs(A[1])+eps)
        out[i,6]=abs(B[2]-A[2])/(abs(A[2])+eps); out[i,7]=abs(B[3]-A[3])/(abs(A[3])+eps); out[i,8]=abs(B[4]-A[4])/(abs(A[4])+eps); out[i,9]=abs(B[5]-A[5])/(abs(A[5])+eps)
        out[i,10]=np.degrees(np.arccos(np.clip(abs(float(np.dot(A[6],B[6]))),0,1))); out[i,11]=abs(B[7]-A[7]); out[i,12]=abs(B[8]-A[8])
    return {"status":"measured","values":out}

@dataclass
class Ref:
    median: np.ndarray; mad: np.ndarray; p95: np.ndarray; count: np.ndarray; template: np.ndarray

class DescriptorNoiseModel:
    def __init__(self, records: list[Record]): self.refs={}; self._build(records)
    @staticmethod
    def _pd(a: Record, b: Record) -> float: return float(np.linalg.norm((a.angles-b.angles)/np.array([15.,20.,15.])))
    def _build(self, records: list[Record]):
        groups=defaultdict(list); templates=defaultdict(list)
        for r in records: groups[(r.dataset_id,r.pose_bin)].append(r); templates[r.pose_bin].append(r.ldm134)
        vals=defaultdict(list)
        for (_,pose),rs in groups.items():
            rs=sorted(rs,key=lambda r:(float(r.angles[1]),float(r.angles[0]),r.sequence)); tpl=np.median(np.stack(templates[pose][:200]),axis=0)
            for off in (1,2):
                for a,b in zip(rs,rs[off:]):
                    if self._pd(a,b)<=2.5:
                        x=local_pair_descriptors(a,b,tpl)
                        if x["status"]=="measured": vals[pose].append(x["values"])
        for pose,arr in vals.items():
            st=np.stack(arr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore",RuntimeWarning)
                med=np.nanmedian(st,0); mad=np.nanmedian(np.abs(st-med),0); p95=np.nanpercentile(st,95,0)
            self.refs[pose]=Ref(med.astype('f4'),mad.astype('f4'),p95.astype('f4'),np.sum(np.isfinite(st),0).astype('i4'),np.median(np.stack(templates[pose][:200]),0).astype('f4'))
    def score(self, pose: str, a: Record, b: Record) -> dict[str, object]:
        ref=self.refs.get(pose)
        if ref is None:
            base=np.full((134,len(NAMES)),np.nan,np.float32)
            return {"status":"insufficient_calibration","values":base,'z':base.copy(),'significant':np.zeros((134,len(NAMES)),bool),'summary':{}}
        raw=local_pair_descriptors(a,b,ref.template); v=raw['values']; floor=np.maximum(np.nanmedian(ref.mad,axis=0)*.25,1e-6); den=np.maximum(1.4826*ref.mad,floor)
        valid=np.isfinite(v)&(ref.count>=7)&np.isfinite(ref.p95); z=np.full_like(v,np.nan); z[valid]=(v[valid]-ref.median[valid])/den[valid]; sig=valid&(v>ref.p95)&(z>=3)
        zv=z[np.isfinite(z)]; lm=np.any(sig,axis=1); top=[]
        for j,name in enumerate(NAMES):
            q=z[:,j]; top.append((name,float(np.nanpercentile(q,95)) if np.isfinite(q).any() else 0.,int(sig[:,j].sum())))
        top.sort(key=lambda x:(x[2],x[1]),reverse=True)
        summary={"significant_cell_fraction":float(sig.sum()/max(valid.sum(),1)),"significant_landmark_fraction":float(lm.sum()/max(np.any(valid,axis=1).sum(),1)),"p95_descriptor_z":float(np.percentile(zv,95)) if zv.size else 0.,"top_descriptor_families":"|".join(x[0] for x in top[:5] if x[2]>0),"top_descriptor_counts":"|".join(f'{x[0]}:{x[2]}' for x in top[:5] if x[2]>0)}
        status='descriptor_jump_candidate' if summary['significant_landmark_fraction']>=.15 and summary['p95_descriptor_z']>=3.5 else ('within_descriptor_noise' if summary['significant_landmark_fraction']<.08 else 'descriptor_uncertain')
        return {"status":status,"values":v,"z":z,"significant":sig,"summary":summary}
