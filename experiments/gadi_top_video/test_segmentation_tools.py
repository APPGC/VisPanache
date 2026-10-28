import unittest
import numpy as np
from segmentation_tools import luminance,sky_weight,recovered_wide_patch,validate_prompts,extract_object,mask_comparison
class Tests(unittest.TestCase):
 def test_uint8_and_normalized_float_match(self):
  a=np.full((32,32,3),200,np.uint8)
  np.testing.assert_array_equal(luminance(a),luminance(a.astype('float32')/255))
  self.assertTrue((sky_weight(a)>0.99).all())
  with self.assertRaises(ValueError):luminance(a.astype('float32'))
 def test_original_and_fixed_patch_do_not_mutate_or_remove_support(self):
  rgb=np.full((64,64,3),180,np.uint8);bg=np.full((64,64,3),.6,np.float32)
  m=np.zeros((64,64),bool);m[20:30,20:40]=True;m[33:45,20:40]=True;before=m.copy();w=np.full(m.shape,.5,np.float32)
  for fixed in [True,False]:
   result,patch,_=recovered_wide_patch(correct_units=fixed)(rgb,bg,m,w,[0,0,64,64],max_distance_from_sam3_px=5,vertical_pad_px=10,contrast_threshold=0,support_threshold=.55,smooth_sigma=.5)
   self.assertFalse((patch&m).any());self.assertTrue(np.array_equal(result,m|patch));np.testing.assert_array_equal(m,before)
 def test_prompt_identity_and_bounds(self):
  p=dict(frame_index=5,positive_points=[[10,20]],negative_points=[[30,40]])
  i,xy,labels=next(validate_prompts([p]));self.assertEqual(i,5);np.testing.assert_array_equal(labels,[1,0]);self.assertAlmostEqual(float(xy[0,0]),10/3840)
  with self.assertRaises(ValueError):list(validate_prompts([dict(p,negative_points=[[10,20]])]))
  with self.assertRaises(ValueError):list(validate_prompts([dict(p,frame_index=22)]))
 def test_object_identity_and_missing_not_zero(self):
  a=np.ones((2,3),bool);r=dict(outputs=dict(out_obj_ids=[9,1],out_binary_masks=np.array([~a,a])))
  np.testing.assert_array_equal(extract_object(r,1,(2,3)),a)
  with self.assertRaises(ValueError):extract_object(r,2,(2,3))
 def test_coverage_comparison_separates_added_removed(self):
  d=mask_comparison(np.array([[1,0],[1,0]]),np.array([[0,1],[1,0]]))
  self.assertEqual(d['added_pixels'],1);self.assertEqual(d['removed_pixels'],1);self.assertAlmostEqual(d['iou'],1/3)
class ComplianceTests(unittest.TestCase):
 def test_prompt_compliance_reports_failures_without_accepting_as_truth(self):
  from segmentation_tools import prompt_satisfaction
  m=np.zeros((10,10),bool);m[2,2]=True
  r=prompt_satisfaction(m,[[.25,.25],[.75,.75],[.25,.25]],[1,0,0])
  self.assertEqual(r['per_point_satisfied'],[True,True,False]);self.assertEqual(r['satisfied_count'],2)
class StagingTests(unittest.TestCase):
 def test_numeric_frame_mapping_preserves_source_bytes_and_refuses_overwrite(self):
  import tempfile,hashlib
  from pathlib import Path
  from video_experiment import prepare_frames
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);src=root/'source';src.mkdir()
   for i in range(355,377):(src/('frame_%05d.png'%i)).write_bytes(('original-%d'%i).encode())
   records=prepare_frames(src,root/'staged')
   self.assertEqual(records[5]['frame_id'],'frame_00360')
   self.assertEqual((root/'staged/5.png').read_bytes(),(src/'frame_00360.png').read_bytes())
   self.assertEqual((root/'staged/5.png').stat().st_ino,(src/'frame_00360.png').stat().st_ino)
   self.assertEqual(len(records),22)
   with self.assertRaises(FileExistsError):prepare_frames(src,root/'staged')
if __name__=='__main__':unittest.main(verbosity=2)
