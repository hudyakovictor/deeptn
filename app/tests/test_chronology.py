from __future__ import annotations
import unittest
from app.stage2.chronology import apply_chronology_rate_flags
class ChronologyTests(unittest.TestCase):
    def test_short_interval_jump_gets_flagged(self):
        rows=[
            {'pair_type':'adjacent','pose_bin':'frontal','date_a':'2024-01-01','date_b':'2024-01-10','pair_index':1,'status':'coherent_jump_candidate','p95_point_z':7.0,'coherent_motion_fraction':0.8,'significant_point_fraction':0.4},
            {'pair_type':'adjacent','pose_bin':'frontal','date_a':'2024-01-10','date_b':'2024-03-10','pair_index':2,'status':'coherent_jump_candidate','p95_point_z':4.0,'coherent_motion_fraction':0.5,'significant_point_fraction':0.2},
            {'pair_type':'adjacent','pose_bin':'frontal','date_a':'2024-03-10','date_b':'2024-06-10','pair_index':3,'status':'within_reconstruction_noise','p95_point_z':1.0,'coherent_motion_fraction':0.1,'significant_point_fraction':0.01},
        ]
        refs=apply_chronology_rate_flags(rows)
        self.assertIn('frontal',refs)
        self.assertIn(rows[0]['chronology_rate_status'],('rapid_change_candidate','persistent_rapid_change_candidate'))
        self.assertEqual(rows[0]['biological_rate_status'], rows[0]['chronology_rate_status'])
if __name__=='__main__':unittest.main()
