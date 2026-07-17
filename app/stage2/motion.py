from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Any
import numpy as np
import warnings
from .anchor_policy import stable_anchor_mask
from .core import Record,robust_rigid_align


def aligned_point_motion(a:Record,b:Record,count:int,identity_only:bool=False)->dict[str,np.ndarray|int|str]:
    if count==106:
        pa,pb=a.ldm106,b.ldm106;vis=np.asarray(a.visible106,bool)&np.asarray(b.visible106,bool)
        if identity_only: pa,pb=a.identity_only106,b.identity_only106
        minimum=24
    else:
        pa,pb=a.ldm134,b.ldm134;vis=np.asarray(a.visible134,bool)&np.asarray(b.visible134,bool)
        if identity_only: pa,pb=a.identity_only134,b.identity_only134
        minimum=30
    vectors=np.full((count,3),np.nan,np.float32);magnitude=np.full(count,np.nan,np.float32)
    if pa is None or pb is None:return {'status':'unavailable','vectors':vectors,'magnitude':magnitude,'visible':vis,'point_count':int(vis.sum()),'anchor_count':0,'anchor_policy':'unavailable'}
    if int(vis.sum())<minimum:return {'status':'insufficient_visibility','vectors':vectors,'magnitude':magnitude,'visible':vis,'point_count':int(vis.sum()),'anchor_count':0,'anchor_policy':'insufficient_visibility'}
    anchors,ameta=stable_anchor_mask(pa,vis,min_count=minimum)
    _,r,t,align_meta=robust_rigid_align(pb[anchors],pa[anchors]);aligned=pb@r+t;vectors[vis]=aligned[vis]-pa[vis];magnitude[vis]=np.linalg.norm(vectors[vis],axis=1)
    return {'status':'measured','vectors':vectors,'magnitude':magnitude,'visible':vis,'point_count':int(vis.sum()),'anchor_count':int(ameta.get('anchor_count',0)),'anchor_policy':str(ameta.get('anchor_policy','unknown')),'alignment_policy':str(align_meta.get('alignment_policy','unknown')),'alignment_trimmed_count':int(align_meta.get('trimmed_point_count',0)),'alignment_residual_before_median':float(align_meta.get('residual_before_median',0.0) or 0.0),'alignment_residual_after_median':float(align_meta.get('residual_after_median',0.0) or 0.0)}


@dataclass
class PointNoiseReference:
    median:np.ndarray;mad:np.ndarray;p95:np.ndarray;count:np.ndarray;template:np.ndarray


class PointNoiseModel:
    """Per-landmark same-person reconstruction noise by pose bin."""
    def __init__(self,records:list[Record]):
        self.records=records;self.references:dict[tuple[str,int],PointNoiseReference]={};self._build()
    @staticmethod
    def _pose_distance(a,b):return float(np.linalg.norm((a.angles-b.angles)/np.array([15.,20.,15.])))
    def _build(self):
        groups=defaultdict(list)
        for r in self.records:groups[(r.dataset_id,r.pose_bin)].append(r)
        values=defaultdict(list);templates=defaultdict(list)
        for (_,pose),rs in groups.items():
            rs=sorted(rs,key=lambda r:(float(r.angles[1]),float(r.angles[0]),r.sequence))
            for r in rs:templates[(pose,106)].append(r.ldm106);templates[(pose,134)].append(r.ldm134)
            for off in (1,2):
                for a,b in zip(rs,rs[off:]):
                    if self._pose_distance(a,b)>2.5:continue
                    for count in (106,134):
                        m=aligned_point_motion(a,b,count)
                        if m['status']=='measured':values[(pose,count)].append(m['magnitude'])
        for key,arrs in values.items():
            stack=np.stack(arrs)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',RuntimeWarning)
                median=np.nanmedian(stack,axis=0);mad=np.nanmedian(np.abs(stack-median),axis=0);p95=np.nanpercentile(stack,95,axis=0)
            cnt=np.sum(np.isfinite(stack),axis=0);template=np.nanmedian(np.stack(templates[key][:200]),axis=0)
            self.references[key]=PointNoiseReference(median.astype(np.float32),mad.astype(np.float32),p95.astype(np.float32),cnt.astype(np.int32),template.astype(np.float32))
    def score(self,pose:str,count:int,motion:dict[str,Any])->dict[str,Any]:
        ref=self.references.get((pose,count));mag=np.asarray(motion['magnitude'],np.float32);z=np.full(count,np.nan,np.float32);sig=np.zeros(count,bool)
        if ref is None:return {'status':'insufficient_calibration','z':z,'significant':sig,'summary':{'calibrated_point_count':0}}
        floor=max(float(np.nanmedian(ref.mad))*0.25,1e-6);den=np.maximum(1.4826*ref.mad,floor);valid=np.isfinite(mag)&(ref.count>=7)&np.isfinite(ref.p95);z[valid]=(mag[valid]-ref.median[valid])/den[valid];sig[valid]=(mag[valid]>ref.p95[valid])&(z[valid]>=3.0)
        coh=self._coherence(ref.template,np.asarray(motion['vectors']),valid,sig)
        zv=z[np.isfinite(z)];summary={'calibrated_point_count':int(valid.sum()),'significant_point_count':int(sig.sum()),'significant_fraction':float(sig.sum()/max(valid.sum(),1)),'coherent_fraction':float(coh),'median_point_z':float(np.median(zv)) if zv.size else 0.,'p95_point_z':float(np.percentile(zv,95)) if zv.size else 0.}
        if valid.sum()<max(30,int(count*.4)):status='insufficient_calibration'
        elif summary['significant_fraction']<.08:status='within_reconstruction_noise'
        elif summary['significant_fraction']>=.15 and coh>=.45 and summary['p95_point_z']>=3.5:status='coherent_jump_candidate'
        else:status='scattered_or_uncertain'
        return {'status':status,'z':z,'significant':sig,'summary':summary}
    @staticmethod
    def _coherence(template,vectors,valid,significant,k=6):
        ids=np.flatnonzero(valid);sids=np.flatnonzero(significant)
        if len(sids)<3 or len(ids)<k+1:return 0.
        dist=np.sum((template[:,None,:]-template[None,:,:])**2,axis=2);scores=[]
        for i in sids:
            neigh=[j for j in np.argsort(dist[i]) if j!=i and valid[j]][:k]
            if not neigh:continue
            vi=vectors[i];ni=np.linalg.norm(vi)
            agree=[]
            for j in neigh:
                nj=np.linalg.norm(vectors[j]);cos=float(np.dot(vi,vectors[j])/(ni*nj)) if ni>0 and nj>0 else -1.
                agree.append(bool(significant[j] and cos>.25))
            scores.append(sum(agree)/len(agree))
        return float(np.mean(scores)) if scores else 0.
