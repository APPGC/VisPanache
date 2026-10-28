"""Run one explicit attempt inside an already allocated GPU shell."""
from pathlib import Path
import argparse,json,os
from video_experiment import prepare_frames,run_experiment
from compare_video import compare_video
p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--prompts-name',default='prompts.json');p.add_argument('--postprocess',choices=['native','historical_identity'],default='native');args=p.parse_args()
if '/' in args.attempt or not args.attempt.startswith('video_'):raise ValueError('invalid attempt label')
if '/' in args.prompts_name or not args.prompts_name.startswith('prompts') or not args.prompts_name.endswith('.json'):raise ValueError('prompt file must remain in approved root')
root=Path('/g/data/kl02/ag8736/FLAMEX/vispanache_top_segmentation_20261027_attempt01');data=Path('/g/data/kl02/ag8736/FLAMEX/data/vispanache_restore_20261027_attempt01')
if not os.environ.get('PBS_JOBID'):raise RuntimeError('run within approved GPU allocation')
from allocation_race import claim_allocation
won,winner=claim_allocation(root,os.environ['PBS_JOBID'])
print(json.dumps(dict(allocation_winner=winner,this_job_won=won)),flush=True)
if not won:raise SystemExit(20)
frames=root/'input_frames'
if not frames.exists():prepare_frames(data/'working/frames_20260428_hilltop',frames)
checkpoint=json.loads((root/'runtime/checkpoint.json').read_text())['path']
result=run_experiment(frame_dir=frames,prompts_path=root/args.prompts_name,checkpoint=checkpoint,source_checkout=root/'runtime/sam3',output=root/'runs'/args.attempt,postprocess=args.postprocess)
print(json.dumps(result),flush=True)
saved=data/'saved_runs/frames_20260428_hilltop_2_harmo'
rows=compare_video(root/'runs'/args.attempt,saved/'plume_tracking.nc',saved/'background_clean.npy',root/'comparisons'/args.attempt)
print('Compared',len(rows),'frames; no independent plume/cloud truth',flush=True)
