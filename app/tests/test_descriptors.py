from __future__ import annotations
import unittest,numpy as np
from app.stage2.core import Record
from app.stage2.descriptors import local_pair_descriptors,NAMES

def rec(points):
    return Record('r','d','2020-01-01',0,'frontal',np.zeros(3,np.float32),points[:106].copy(),points.copy(),np.ones(106,bool),np.ones(134,bool),np.zeros(80,np.float32),np.zeros(64,np.float32),points[:106].copy(),points.copy())
class DescriptorTests(unittest.TestCase):
    def test_all_metric_families_are_finite(self):
        rng=np.random.default_rng(1);p=rng.normal(size=(134,3)).astype('f4');q=p.copy();q[20:40,0]*=1.2;q[20:40,2]+=.3
        out=local_pair_descriptors(rec(p),rec(q),p)
        self.assertEqual(out['values'].shape,(134,len(NAMES)))
        self.assertGreater(np.isfinite(out['values']).sum(),1000)
        self.assertGreater(float(np.nanmax(out['values'][:,3:])),0)
if __name__=='__main__':unittest.main()
