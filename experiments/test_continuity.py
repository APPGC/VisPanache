import unittest
import numpy as np
from continuity import ContinuityConfig,repair_sequence,attach_continuity_qc,CONTINUITY_INFERRED
class Tests(unittest.TestCase):
 def setUp(self):
  self.w=np.zeros((96,128),np.float32);self.w[:48]=1
  self.m=np.zeros(self.w.shape,bool);self.m[20:40,20:100]=True;self.m[55:80,20:100]=True
  self.c=ContinuityConfig(max_gap_px=20,interface_band_px=30,min_column_run_px=8,max_corridor_width_fraction=.9,max_added_area_fraction=.5,min_component_pixels=10)
 def test_local_additive_bridge_and_raw_input_preserved(self):
  old=self.m.copy();r,_=repair_sequence([self.m]*3,self.w,self.c);a=r[1]['continuity_added_mask']
  np.testing.assert_array_equal(self.m,old);self.assertTrue(a[40:55,20:100].all());self.assertFalse(a[:,:20].any());self.assertFalse(a[:,100:].any());self.assertTrue(r[1]['continuity_mask'][20:80,20:100].all())
 def test_large_gap_or_missing_support_abstains(self):
  c=ContinuityConfig(**{**self.c.__dict__,'max_gap_px':10});r,_=repair_sequence([self.m]*3,self.w,c);self.assertFalse(r[1]['continuity_added_mask'].any())
  missing=self.m.copy();missing[:48]=False;r,_=repair_sequence([missing]*3,self.w,self.c);self.assertFalse(r[1]['continuity_added_mask'].any())
 def test_temporally_isolated_bridge_abstains(self):
  r,_=repair_sequence([np.zeros_like(self.m),self.m,np.zeros_like(self.m)],self.w,self.c);self.assertFalse(r[1]['continuity_added_mask'].any())
 def test_internal_gap_already_connected_is_still_repaired(self):
  m=self.m.copy();m[40:55,20]=True;r,_=repair_sequence([m]*3,self.w,self.c);self.assertTrue(r[1]['continuity_mask'][20:80,20:100].all())
 def test_added_area_guard_abstains_without_trimming_or_removal(self):
  c=ContinuityConfig(**{**self.c.__dict__,'max_added_area_fraction':.01});r,_=repair_sequence([self.m]*3,self.w,c);np.testing.assert_array_equal(r[1]['continuity_mask'],self.m);self.assertEqual(r[1]['metrics']['status'],'abstained_added_area_limit')
 def test_inferred_pixels_never_gain_opacity_or_confidence(self):
  raw=np.array([[1,0],[0,0]],bool);added=np.array([[0,1],[0,0]],bool)
  base=dict(plume_mask=raw,opacity=np.array([[.7,np.nan],[np.nan,np.nan]],np.float32),quality_flag=np.array([[8,0],[0,0]],np.uint32),retrieval_confidence=np.array([[.5,np.nan],[np.nan,np.nan]],np.float32),retrieval_regime=np.array([[1,0],[0,0]],np.uint8),valid_opacity_mask=raw.copy())
  r=attach_continuity_qc(base,raw|added,added,{})
  np.testing.assert_array_equal(r['opacity'],base['opacity']);self.assertTrue(np.isnan(r['opacity'][0,1]));self.assertEqual(r['retrieval_confidence'][0,1],0);self.assertEqual(r['retrieval_regime'][0,1],4);self.assertTrue(r['quality_flag'][0,1]&CONTINUITY_INFERRED);self.assertEqual(base['quality_flag'][0,1],0)
if __name__=='__main__':unittest.main(verbosity=2)
