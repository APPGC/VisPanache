import sys,unittest,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from smoke_opacity import retrieve_hybrid_opacity,retrieve_hybrid_opacity_qc
from smoke_opacity.quality import QualityFlag as F
from smoke_opacity.quality_views import retain_flagged_boundary
class PackageTests(unittest.TestCase):
 def test_cpu_api_import_does_not_load_segmentation(self):
  code='from smoke_opacity import retrieve_hybrid_opacity_qc, retrieve_hybrid_opacity; import sys; assert "torch" not in sys.modules; assert "cv2" not in sys.modules; assert "imageio" not in sys.modules'
  subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],check=True)
 def test_normal_package_retrieval_matches_raw_function(self):
  image=np.full((20,30,3),.5,'float32');background=np.full_like(image,.2);mask=np.ones((20,30),bool);w=np.zeros(mask.shape,'float32')
  expected=retrieve_hybrid_opacity(image,background,mask,w)[0]
  result=retrieve_hybrid_opacity_qc(image,background,mask,w)
  np.testing.assert_array_equal(result['opacity_baseline'],expected);np.testing.assert_array_equal(result['opacity'],expected)
 def test_boundary_view_restores_only_flagged_safe_raw_pixels(self):
  flags=np.array([[16,16|1,16,16|64,16,16|4096,16|8192,16|1024]],'uint32')
  base=dict(plume_mask=np.array([[1,1,0,1,1,1,1,1]],bool),quality_flag=flags,opacity_baseline=np.full((1,8),.2,'float32'),opacity=np.full((1,8),np.nan,'float32'),valid_opacity_mask=np.zeros((1,8),bool),retrieval_regime=np.full((1,8),3,'uint8'),retrieval_confidence=np.zeros((1,8),'float32'))
  boundary=np.ones((1,8),bool);boundary[0,4]=False
  out=retain_flagged_boundary(base,boundary)
  np.testing.assert_array_equal(out['boundary_estimate_retained_mask'],np.array([[1,0,0,0,0,0,0,0]],bool))
  assert out['opacity_with_flagged_boundary'][0,0]==np.float32(.2)
  for key in base:np.testing.assert_array_equal(base[key],out[key])
 def test_boundary_view_rejects_malformed_mask(self):
  with self.assertRaises(ValueError):retain_flagged_boundary(dict(plume_mask=np.ones((2,2),bool)),np.ones((2,2),'uint8'))
if __name__=='__main__':unittest.main(verbosity=2)
