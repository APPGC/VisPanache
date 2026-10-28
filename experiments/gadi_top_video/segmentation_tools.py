"""Pure CPU helpers; no model imports, downloads or hidden prompt edits."""
from pathlib import Path
import ast,hashlib,json
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt

POSITIVE=[[2450,1220],[2700,1080],[2920,930],[3050,760],[3180,520],[3250,280]]
NEGATIVE=[[2850,250],[3500,250],[3650,600],[3600,900],[2150,900],[2450,1400]]


def rgb01(rgb):
 a=np.asarray(rgb)
 if a.ndim!=3 or a.shape[-1]!=3:raise ValueError('expected H,W,3 RGB')
 if a.dtype==np.uint8:return a.astype('float32')/255
 if not np.issubdtype(a.dtype,np.floating):raise ValueError('only uint8 or explicitly normalized float RGB supported')
 if not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('float RGB must be finite in [0,1]')
 return a.astype('float32',copy=False)


def luminance(rgb):
 a=rgb01(rgb)
 return (.2126*a[...,0]+.7152*a[...,1]+.0722*a[...,2]).astype('float32')


def sky_weight(bg):return np.clip(gaussian_filter((luminance(bg)>.55).astype('float32'),5),0,1)


def recovered_wide_patch(*,correct_units=True):
 """Execute original notebook function definitions; replace only RGB conversion."""
 evidence=Path(__file__).resolve().parents[2]/'top_experiments_audit_20261027/notebook_evidence.json'
 entry=next(e for e in json.loads(evidence.read_text()) if e['path'].endswith('/RUN_SAM3_bridge_correction.ipynb'))
 if entry['sha256']!='1dcb277d339eb1bcbda57a2a009c0c8281b31fc221374c1111a3c8b8be339a08':raise RuntimeError('source notebook changed')
 source=''.join(entry['cells'][32]['source']);tree=ast.parse(source)
 functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
 ns=dict(np=np,gaussian_filter=gaussian_filter,distance_transform_edt=distance_transform_edt)
 exec(compile(ast.Module(body=functions,type_ignores=[]),entry['path']+'#cell32','exec'),ns)
 if correct_units:ns['luminance']=luminance
 return ns['fast_interface_patch_wide']


def validate_prompts(prompts,width=3840,height=2160,frame_count=22):
 seen=set()
 for p in prompts:
  i=p['frame_index']
  if not isinstance(i,int) or not 0<=i<frame_count or i in seen:raise ValueError('invalid/duplicate prompt frame')
  seen.add(i)
  pos=np.asarray(p['positive_points'],dtype='float32').reshape(-1,2);neg=np.asarray(p['negative_points'],dtype='float32').reshape(-1,2)
  points=np.concatenate([pos,neg]);labels=np.array([1]*len(pos)+[0]*len(neg),dtype='int32')
  if not len(pos) or not np.isfinite(points).all():raise ValueError('positive prompts required; all coordinates finite')
  if np.any(points<0) or np.any(points[:,0]>=width) or np.any(points[:,1]>=height):raise ValueError('prompt outside image')
  if any(tuple(x) in set(map(tuple,pos)) for x in neg):raise ValueError('contradictory prompt labels')
  yield i,points/np.array([width,height],dtype='float32'),labels


def extract_object(response,obj_id,shape):
 out=response['outputs'];ids=np.asarray(out['out_obj_ids']);masks=np.asarray(out['out_binary_masks'])
 if masks.shape!=(len(ids),*shape):raise ValueError('unexpected model output dimensions')
 index=np.flatnonzero(ids==obj_id)
 if len(index)!=1:raise ValueError('target object missing or duplicated; do not fabricate an empty mask')
 return masks[index[0]].astype(bool)


def mask_comparison(before,after,upper_region=None):
 a=np.asarray(before,dtype=bool);b=np.asarray(after,dtype=bool)
 if a.shape!=b.shape:raise ValueError('mask shape mismatch')
 inter=int((a&b).sum());union=int((a|b).sum())
 out=dict(before_pixels=int(a.sum()),after_pixels=int(b.sum()),added_pixels=int((b&~a).sum()),removed_pixels=int((a&~b).sum()),iou=inter/union if union else None)
 if upper_region is not None:
  u=np.asarray(upper_region,dtype=bool);out.update(upper_before_pixels=int((a&u).sum()),upper_after_pixels=int((b&u).sum()))
 return out


def prompt_satisfaction(mask,points_rel,labels):
 """Report point compliance, without interpreting it as segmentation accuracy."""
 m=np.asarray(mask,dtype=bool);points=np.asarray(points_rel,dtype='float32');labels=np.asarray(labels,dtype='int32')
 if points.shape!=(len(labels),2) or not np.isin(labels,[0,1]).all():raise ValueError('bad prompt shapes/labels')
 if not np.isfinite(points).all() or np.any(points<0) or np.any(points>=1):raise ValueError('relative points must lie in [0,1)')
 xy=np.floor(points*np.array([m.shape[1],m.shape[0]])).astype(int)
 inside=m[xy[:,1],xy[:,0]];satisfied=inside==(labels==1)
 return dict(points=len(labels),satisfied_count=int(satisfied.sum()),positive_satisfied=int((satisfied&(labels==1)).sum()),negative_satisfied=int((satisfied&(labels==0)).sum()),per_point_inside=inside.tolist(),per_point_satisfied=satisfied.tolist(),coordinate_sampling='floor(relative * image dimensions)',interpretation='Point compliance only; no segmentation accuracy claim')
