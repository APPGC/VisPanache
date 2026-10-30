from pathlib import Path
import argparse,json,sys,hashlib
import numpy as np,xarray as xr
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
sys.path.insert(0,str(a.source.resolve()))
from smoke_opacity.retrieval import retrieve
from reference_diagnostics import assess_reference
params=json.loads((a.source/'configs/retrieval_paper.json').read_text());bg=np.load(a.bundle/'data/background.npy');weight=np.load(a.bundle/'data/sky_weight.npy');records=[]
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(a.source/'smoke_opacity/opacity.py')=='d565c572699f59d35690593471e43334705291447567e1c150b1daa604382293'
for anchor in (2500,2600):
 w=a.bundle/'data/sam3'/f'window_{anchor}';run=w/'attempt03';frames=json.loads((w/'frame_manifest.json').read_text());stream=json.loads((run/'stream_manifest.json').read_text());files={r['frame_index']:r['file'] for r in stream};assert set(files)==set(range(11))
 for row in frames:
  image_path=a.bundle/'data/frames'/(row['frame_id']+'.png');assert sha(image_path)==row['sha256'];img=np.asarray(Image.open(image_path).convert('RGB')).astype('float32')/255
  with np.load(run/files[row['frame_index']]) as z:mask=z['mask'].astype(bool)
  result=retrieve(img,bg,mask,weight,**params);diag=assess_reference(img,bg,mask,result['diagnostics']['sky_weight']);valid=result['valid_opacity_mask']&((diag['case_warning_flags']&np.uint16(4|8|64))==0)
  arrays={n:result[n] for n in ['opacity_baseline','opacity','quality_flag','retrieval_confidence','retrieval_regime','plume_mask','valid_opacity_mask']}
  arrays.update({n:diag[n] for n in ['case_warning_flags','case_evaluated_mask','reference_comparison_mask','archived_reference_contrast','same_row_control_contrast','control_reference_shift']});arrays['opacity_case_screened']=np.where(valid,result['opacity'],np.nan).astype('float32');arrays['valid_opacity_case_mask']=valid
  with xr.open_dataset(a.bundle/'expected/products'/(row['frame_id']+'.nc')) as expected:
   for name,arr in arrays.items():np.testing.assert_equal(arr,expected[name].values,err_msg=row['frame_id']+' '+name)
  np.savez_compressed(a.output/(row['frame_id']+'.npz'),**arrays);records.append(dict(frame=row['frame_id'],arrays_exact=len(arrays)));print('PASS',row['frame_id'],flush=True)
(a.output/'verification.json').write_text(json.dumps(dict(status='PASS_ALL22_EXACT_SAVED_MASK_CPU_REPLAYS',frames=len(records),arrays=records,source=str(a.source),interpretation='Software reproduction in recorded environment, not a SAM3 rerun or physical-opacity validation.'),indent=2)+'\n')
