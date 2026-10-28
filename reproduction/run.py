"""Reproduce saved paper cases from the matching private data archive, CPU only."""
from pathlib import Path
import argparse,json,hashlib,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for a in iter(lambda:f.read(8*1024**2),b''):h.update(a)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=['all','bottom','top_original','top_guided','top_lower_ablation'],default='all');p.add_argument('--quick',action='store_true');p.add_argument('--refine-seam-edges',action='store_true');a=p.parse_args()
 if a.output.exists():p.error('output must be new; previous products are never overwritten')
 manifest=json.loads((a.data/'input_manifest.json').read_text())
 for r in manifest['files']:
  path=a.data/r['path']
  if path.stat().st_size!=r['bytes'] or sha(path)!=r['sha256']:raise RuntimeError('input mismatch:'+r['path'])
 cases=json.loads((ROOT/'reproduction/cases.json').read_text());names=list(cases) if a.case=='all' else [a.case];a.output.mkdir(parents=True);receipts=[]
 for name in names:
  c=cases[name];cmd=[sys.executable,str(ROOT/'examples/retrieve_saved_sequence.py'),'--tracking',str(a.data/c['tracking']),'--background',str(a.data/c['background']),'--sky-weight',str(a.data/c['weight']),'--parameters',str(ROOT/'configs/retrieval_paper.json'),'--output',str(a.output/name),'--camera',c['camera']]
  if c.get('inpaint'):cmd+=['--inpaint-mask',str(a.data/c['inpaint'])]
  if c.get('masks'):cmd+=['--mask-manifest',str(a.data/c['masks'])]
  if a.quick:cmd+=['--frames',*c['quick']]
  if a.refine_seam_edges and name=='bottom':cmd+=['--refine-seam-edges']
  subprocess.run(cmd,check=True);receipts.append(dict(case=name,result=json.loads((a.output/name/'completion.json').read_text())))
 (a.output/'reproduction_receipt.json').write_text(json.dumps(dict(version='1.0.0',cases=receipts,quick=a.quick,seam_refinement_requested=a.refine_seam_edges,scope='SavedRGBandmasks; no newSAM3 or modeldownload'),indent=2)+'\n')
if __name__=='__main__':main()
