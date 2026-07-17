from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from app.stage2.leads import load_leads,pair_leads
class LeadTests(unittest.TestCase):
    def test_archive_is_coverage_not_ground_truth(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            (p/'top_identity_breaks.json').write_text(json.dumps({'entries':[{'date_str':'2024-10-04','photo_id':'2024_10_04','calibration_features':{'exceeded_metrics':['zone_orbit_R_normal_mean_x'],'mesh_zones':[{'name':'orbit_R','raw_error':1}]}}]}))
            r=load_leads(p)
            self.assertEqual(r['status'],'loaded')
            self.assertIn('never ground truth',r['policy'])
            x=pair_leads(r,'2024-01-01','2024-10-04')
            self.assertTrue(x['lead_overlap'])
            self.assertIn('orbit',x['lead_regions'])
if __name__=='__main__':unittest.main()
