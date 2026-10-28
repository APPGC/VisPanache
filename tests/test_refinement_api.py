import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from smoke_opacity.retrieval import retrieve,annotate_refinement
from smoke_opacity.quality import retrieve_hybrid_opacity_qc,to_quality_dataset
class Tests(unittest.TestCase):
 def test_disabled_refinement_is_exact_original_qc(self):
  rng=np.random.default_rng(42);img=rng.uniform(.2,.8,(32,40,3)).astype('float32');bg=np.full_like(img,.45);m=np.ones((32,40),bool);w=np.tile(np.linspace(0,1,40,dtype='float32'),(32,1))
  old=retrieve_hybrid_opacity_qc(img,bg,m,w);new=retrieve(img,bg,m,w)
  for key in old:
   if isinstance(old[key],np.ndarray):np.testing.assert_array_equal(old[key],new[key])
  self.assertNotIn('opacity_refined',new)
 def test_optional_refinement_preserves_screening_and_original_fields(self):
  img=np.full((20,30,3),1.,'float32');bg=np.full_like(img,.6);m=np.ones((20,30),bool);w=np.full(m.shape,.5,'float32')
  old=retrieve_hybrid_opacity_qc(img,bg,m,w);new=retrieve(img,bg,m,w,refine_seam_edges=True)
  for key in ['opacity','opacity_baseline','quality_flag','retrieval_confidence','retrieval_regime','valid_opacity_mask']:np.testing.assert_array_equal(old[key],new[key])
  self.assertTrue(np.isnan(new['opacity_refined']).all());self.assertTrue(np.all(new['retrieval_confidence_refined']==0))
  d=annotate_refinement(to_quality_dataset(new,provenance=dict(camera_id='test',frame_id='test',source_version='test',environment='test',input_paths=['synthetic'])),new)
  self.assertEqual(int(d.quality_flag_refined.attrs['flag_masks'][-1]),262144)
if __name__=='__main__':unittest.main(verbosity=2)
