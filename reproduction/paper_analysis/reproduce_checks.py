"""Reproduce paper ablations, reference perturbations and QC threshold sensitivity.
Run against the frozen v1.0.0 source and the extracted private input archive.
No segmentation, downloads, installation, source edits or output overwrites.
"""
from pathlib import Path
import argparse,json,sys,hashlib,gc
import numpy as np,xarray as xr

def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():p.error('output must be new')
 expected_digest='d565c572699f59d35690593471e43334705291447567e1c150b1daa604382293'
 assert digest(a.source/'smoke_opacity/opacity.py')==expected_digest,'unmatched numerical source'
 for f in json.loads((a.data/'input_manifest.json').read_text())['files']:assert digest(a.data/f['path'])==f['sha256'],f['path']
 sys.path.insert(0,str(a.source));from smoke_opacity.opacity import retrieve_hybrid_opacity as retrieve
 from smoke_opacity.quality import QCConfig,quality_from_diagnostics
 params=json.loads((a.source/'configs/retrieval_paper.json').read_text());a.output.mkdir(parents=True);(a.output/'arrays').mkdir();rows=[];thresholds=[]
 expected=json.loads((a.data/'expected/ablation_background_metrics.json').read_text());ee={(r['camera'],r['frame_id']):r for r in expected}
 sensitivity_expected=json.loads((Path(__file__).parent/'expected_threshold_rows.json').read_text());se={(r['camera'],r['frame_id'],r['contrast_levels'],r['near_clipping_level']):r for r in sensitivity_expected}
 for camera,frames,sensitivity_frames in [('bottom',[700,750,817],[601,700,750,817,900]),('top',[355,367,376],[355,364,366,376])]:
  bg=np.load(a.data/camera/'background.npy');w=np.load(a.data/camera/'sky_weight.npy');inpaint=np.load(a.data/camera/'background_inpaint_mask.npy') if camera=='top' else None
  with xr.open_dataset(a.data/camera/'plume_tracking.nc',engine='h5netcdf') as ds:
   for number in sorted(set(frames+sensitivity_frames)):
    frame='frame_'+str(number).zfill(6 if camera=='bottom' else 5);img=ds.rgb.sel(time=frame).values.astype('float32')/255;mask=ds.plume_mask.sel(time=frame).values.astype(bool)
    # Threshold analysis uses original masks, matching the baseline coverage experiment.
    if number in sensitivity_frames:
     alpha,_,_,diag=retrieve(img,bg,mask,w,return_diagnostics=True,**params)
     for contrast in [1,3,5]:
      for near in [248,250,252]:
       q=quality_from_diagnostics(img,bg,mask,alpha,diag,config=QCConfig(contrast_tolerance=contrast/255,near_clipping_level=near/255),smoke_lum=params['smoke_lum'],min_support_pixels=params['min_support_pixels'],scale_clip=params['scale_clip'],background_inpaint_mask=inpaint)
       row=dict(camera=camera,frame_id=frame,contrast_levels=contrast,near_clipping_level=near,plume_pixels=int(mask.sum()),valid_pixels=int(q['valid_opacity_mask'].sum()))
       old=se[(camera,frame,contrast,near)];assert row['valid_pixels']==old['valid_pixels'],(row,old);thresholds.append(row);del q
     del alpha,diag;gc.collect()
    if number not in frames:continue
    # Reference perturbations use the full-guided top masks, as in the paper.
    if camera=='top':
     folder=a.data/'top/guided_full';manifest=json.loads((folder/'stream_manifest.json').read_text());entry=next(r for r in manifest if r['frame_id']==frame)
     with np.load(folder/entry['file']) as f:mask=f['mask'].astype(bool)
    alpha,_,_,diag=retrieve(img,bg,mask,w,return_diagnostics=True,**params);sm=diag['seam_gap'].copy();del diag
    row=dict(camera=camera,frame_id=frame,plume_pixels=int(mask.sum()),baseline_seam_pixels=int(sm.sum()),background_sensitivity=[])
    if camera=='bottom':
     no_seam=retrieve(img,bg,mask,w,**dict(params,seam_repair=False))[0];no_cal=retrieve(img,bg,mask,w,**dict(params,seam_repair=False,calibration_strength=0.))[0]
     row.update(crosscal_mean_absolute_change=float(np.abs(no_seam[mask]-no_cal[mask]).mean()),seam_mean_absolute_change=float(np.abs(alpha[mask]-no_seam[mask]).mean()))
     np.savez_compressed(a.output/'arrays'/(camera+'_'+frame+'.npz'),opacity_baseline=alpha,opacity_no_seam=no_seam,opacity_no_cal_no_seam=no_cal,plume_mask=mask,seam_mask=sm);del no_seam,no_cal
    for offset in [-5,-2,2,5]:
     perturbed=np.clip(bg+offset/255,0,1).astype('float32');aa=retrieve(img,perturbed,mask,w,**params)[0]
     row['background_sensitivity'].append(dict(background_offset_encoded_levels=offset,mean_absolute_opacity_change=float(np.abs(aa[mask]-alpha[mask]).mean())));del aa,perturbed;gc.collect()
    old=ee[(camera,frame)];assert row['plume_pixels']==old['plume_pixels'];assert row['baseline_seam_pixels']==old['baseline_seam_pixels']
    for key in ['crosscal_mean_absolute_change','seam_mean_absolute_change']:
     if key in row:assert row[key]==old[key],(frame,key,row[key],old[key])
    for new,prior in zip(row['background_sensitivity'],old['background_sensitivity']):assert new['mean_absolute_opacity_change']==prior['mean_absolute_opacity_change'],(frame,new,prior)
    rows.append(row);(a.output/'ablation_background_metrics.json').write_text(json.dumps(rows,indent=2)+'\n');print(camera,frame,'exact expected metrics PASS',flush=True);del alpha,sm;gc.collect()
 (a.output/'threshold_rows.json').write_text(json.dumps(thresholds,indent=2)+'\n');(a.output/'completion.json').write_text(json.dumps(dict(status='PASS_EXACT_PAPER_ABLATION_BACKGROUND_AND_THRESHOLD_METRICS',ablation_frames=3,reference_frames=6,reference_scenarios=24,threshold_frame_configurations=len(thresholds),code_sha256=expected_digest),indent=2)+'\n')
if __name__=='__main__':main()
