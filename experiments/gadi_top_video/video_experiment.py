"""Explicit offline SAM3 point-guided experiment; call within approved GPU allocation.

Importing this module does not load torch, allocate a model, or touch the network.
"""
from pathlib import Path
import json,hashlib,os,sys,inspect,subprocess,platform,time
import numpy as np
from segmentation_tools import validate_prompts,extract_object,mask_comparison,prompt_satisfaction
PIN='f1647e8d0c1d280991f4d1b7d974c2704537fe66'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def prepare_frames(source,output):
 """Hard-link the exact 22 approved PNGs with numeric names; never rewrite originals."""
 source=Path(source);output=Path(output)
 selected=[source/('frame_%05d.png'%i) for i in range(355,377)]
 if not all(p.is_file() for p in selected):raise FileNotFoundError('22 selected top frames not present')
 output.mkdir(mode=0o700,exist_ok=False);records=[]
 for i,p in enumerate(selected):
  target=output/(str(i)+'.png');os.link(p,target)
  records.append(dict(frame_index=i,frame_id=p.stem,source=str(p),sha256=sha(p),size=p.stat().st_size,model_filename=target.name))
 (output.parent/'frame_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
 return records

def run_experiment(*,frame_dir,prompts_path,checkpoint,source_checkout,output,postprocess='native'):
 """One 22-frame both-direction propagation, with explicit per-index outputs."""
 if postprocess not in ['native','historical_identity']:raise ValueError('explicit postprocess mode required')
 if not os.environ.get('PBS_JOBID'):raise RuntimeError('GPU experiment must run inside approved PBS allocation')
 if os.environ.get('HF_HUB_OFFLINE')!='1':raise RuntimeError('require offline inference; no implicit model download')
 checkpoint=Path(checkpoint)
 if not checkpoint.is_file():raise FileNotFoundError('pre-staged model checkpoint required')
 revision=subprocess.check_output(['git','-C',str(source_checkout),'rev-parse','HEAD'],text=True).strip()
 if revision!=PIN:raise RuntimeError('unapproved SAM3 source revision')
 prompt_doc=json.loads(Path(prompts_path).read_text());prompts=list(validate_prompts(prompt_doc['prompts']))
 if prompt_doc['frame_count']!=22 or prompt_doc['obj_id']!=1:raise ValueError('scope must remain 22 frames / object1')
 frame_dir=Path(frame_dir)
 if sorted(int(p.stem) for p in frame_dir.glob('*.png'))!=list(range(22)):raise ValueError('numeric model frame mapping is incomplete')
 from PIL import Image
 for p in frame_dir.glob('*.png'):
  with Image.open(p) as img:
   if img.size!=(3840,2160):raise ValueError('unexpected image dimensions')
 import torch
 if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; do not silently run model on CPU')
 import sam3.model_builder as builder
 import sam3.model.sam3_video_inference as inference
 # Confirm that the imported package is the pinned checkout, not a stale unrelated install.
 imported=Path(inspect.getfile(builder)).resolve()
 if Path(source_checkout).resolve() not in imported.parents:raise RuntimeError('imported SAM3 differs from pinned source')
 output=Path(output);output.mkdir(mode=0o700,exist_ok=False)
 manifest_path=frame_dir.parent/'frame_manifest.json';frames=json.loads(manifest_path.read_text())
 if [f['frame_index'] for f in frames]!=list(range(22)):raise ValueError('bad frame manifest')
 for record in frames:
  if sha(frame_dir/record['model_filename'])!=record['sha256']:raise ValueError('staged input checksum mismatch')
 metadata=dict(status='STARTED',pbs_job_id=os.environ['PBS_JOBID'],source_commit=revision,checkpoint_sha256=sha(checkpoint),prompt_sha256=sha(prompts_path),frame_manifest_sha256=sha(manifest_path),postprocess=postprocess,torch_version=torch.__version__,cuda_runtime=torch.version.cuda,gpu=torch.cuda.get_device_name(0),python=sys.version,platform=platform.platform(),direction='both',start_frame_index=min(p[0] for p in prompts),model_score_interpretation='Model object score, not calibrated plume confidence',labels_review_status=prompt_doc['review_status'])
 def status(**kw):
  metadata.update(kw);(output/'status.json').write_text(json.dumps(metadata,indent=2)+'\n')
 status();original=inference.fill_holes_in_mask_scores;session=None;predictor=None;START=time.monotonic();seen={};stream_records=[]
 try:
  if postprocess=='historical_identity':inference.fill_holes_in_mask_scores=lambda mask,*args,**kwargs:mask
  torch.manual_seed(0);np.random.seed(0)
  predictor=builder.build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()],checkpoint_path=str(checkpoint),compile=False)
  response=predictor.handle_request(dict(type='start_session',resource_path=str(frame_dir)))
  session=response['session_id']
  # Use exactly the historical request fields; normalize coordinates explicitly.
  for index,points,labels in prompts:
   response=predictor.handle_request(dict(type='add_prompt',session_id=session,frame_index=index,points=torch.tensor(points,dtype=torch.float32),point_labels=torch.tensor(labels,dtype=torch.int32),obj_id=1))
   prompt_mask=extract_object(response,1,(2160,3840))
   np.save(output/('prompt_frame_%02d.npy'%index),prompt_mask.astype('uint8'))
   (output/('prompt_compliance_%02d.json'%index)).write_text(json.dumps(prompt_satisfaction(prompt_mask,points,labels),indent=2)+'\n')
  stream=predictor.handle_stream_request(dict(type='propagate_in_video',session_id=session,propagation_direction='both',start_frame_index=min(p[0] for p in prompts),max_frame_num_to_track=22))
  for sequence,response in enumerate(stream):
   i=int(response['frame_index'])
   if not 0<=i<22:raise ValueError('model returned out-of-scope frame')
   mask=extract_object(response,1,(2160,3840));name='stream_%03d_frame_%02d.npz'%(sequence,i)
   np.savez_compressed(output/name,mask=mask.astype('uint8'))
   if i in seen:
    with np.load(output/seen[i]) as prior:
     if not np.array_equal(mask,prior['mask']):raise RuntimeError('conflicting duplicate frame; preserve both for review')
   seen[i]=name;stream_records.append(dict(sequence=sequence,frame_index=i,frame_id=frames[i]['frame_id'],file=name,area_px=int(mask.sum())))
   (output/'stream_manifest.json').write_text(json.dumps(stream_records,indent=2)+'\n');status(frames_completed=len(seen))
   if sum(p.stat().st_size for p in output.iterdir() if p.is_file())>2*1024**3:raise RuntimeError('2 GiB experiment output cap exceeded')
  if set(seen)!=set(range(22)):raise RuntimeError('missing propagated frames; no zero masks fabricated')
  status(status='PASS_MODEL_EXPORT_ONLY',elapsed_seconds=time.monotonic()-START,frames_completed=22,validation='22 masks exported; accuracy/cloud labels unvalidated')
  return metadata
 except Exception as e:
  status(status='FAILED_PRESERVED',error=repr(e));raise
 finally:
  try:
   if predictor is not None and session is not None:predictor.handle_request(dict(type='close_session',session_id=session))
  finally:
   inference.fill_holes_in_mask_scores=original
   if predictor is not None:predictor.shutdown()
