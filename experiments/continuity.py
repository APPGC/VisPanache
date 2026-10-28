"""Optional case-scoped continuity prior. Raw detections are never removed."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.ndimage import label,binary_propagation,distance_transform_edt
CONTINUITY_INFERRED=np.uint32(1<<17)

@dataclass(frozen=True)
class ContinuityConfig:
 max_gap_px:int=60
 interface_band_px:int=80
 min_column_run_px:int=12
 max_corridor_width_fraction:float=.25
 max_added_area_fraction:float=.05
 min_component_pixels:int=1000
 min_component_fraction:float=.02
 terrain_max_weight:float=.45
 sky_min_weight:float=.55
 temporal_radius_px:int=20
 temporal_coverage_fraction:float=.9
 require_temporal_support:bool=True
 def __post_init__(self):
  for name in ['max_gap_px','interface_band_px','min_column_run_px','min_component_pixels']:
   if not isinstance(getattr(self,name),int) or getattr(self,name)<1:raise ValueError(name+' must be positive integer')
  if self.temporal_radius_px<0:raise ValueError('negative temporal radius')
  for name in ['max_corridor_width_fraction','max_added_area_fraction','min_component_fraction','temporal_coverage_fraction']:
   if not 0<getattr(self,name)<=1:raise ValueError(name+' must be in (0,1]')
  if not 0<=self.terrain_max_weight<self.sky_min_weight<=1:raise ValueError('bad background weight thresholds')

def interface_geometry(sky_weight,config):
 w=np.asarray(sky_weight,dtype='float32')
 if w.ndim!=2 or not np.isfinite(w).all() or np.any((w<0)|(w>1)):raise ValueError('invalid sky weights')
 sky=w>=.5;seed=np.zeros(w.shape,bool);seed[0]=sky[0]
 top=binary_propagation(seed,structure=np.ones((3,3),bool),mask=sky)
 boundary=np.zeros(w.shape,bool);boundary[:-1]=top[:-1]&~top[1:]
 if boundary.any():band=distance_transform_edt(~boundary)<=config.interface_band_px
 else:band=np.zeros(w.shape,bool)
 return band,boundary

def coherent_columns(columns,config,width):
 good=np.zeros(len(columns),bool);starts=np.flatnonzero(np.diff(np.r_[False,columns,False].astype('int8'))==1);ends=np.flatnonzero(np.diff(np.r_[False,columns,False].astype('int8'))==-1)
 for start,end in zip(starts,ends):
  if config.min_column_run_px<=end-start<=config.max_corridor_width_fraction*width:good[start:end]=True
 return good

def candidate_bridge(mask,sky_weight,band,config):
 m=np.asarray(mask,dtype=bool);w=np.asarray(sky_weight)
 if m.shape!=w.shape or m.shape!=band.shape:raise ValueError('shape mismatch')
 labs,n=label(m,structure=np.ones((3,3),bool));area=np.bincount(labs.ravel());minimum=max(config.min_component_pixels,int(m.sum()*config.min_component_fraction));ids=np.flatnonzero(area>=minimum);ids=ids[ids>0]
 support=np.isin(labs,ids);upper=support&band&(w>=config.sky_min_weight);lower=support&band&(w<=config.terrain_max_weight)
 candidate=np.zeros(m.shape,bool);eligible=np.zeros(m.shape[1],bool)
 for x in np.flatnonzero(upper.any(axis=0)&lower.any(axis=0)):
  up=np.flatnonzero(upper[:,x]);lo=np.flatnonzero(lower[:,x]);u=int(up[-1]);l=int(lo[0])
  if u>=l:continue
  missing=~m[u:l+1,x];gap=int(missing.sum())
  if 0<gap<=config.max_gap_px and l-u-1<=config.max_gap_px and band[u:l+1,x].all():candidate[u:l+1,x]=missing;eligible[x]=True
 keep=coherent_columns(eligible,config,m.shape[1]);candidate[:,~keep]=False
 return candidate,dict(major_components=int(len(ids)),major_component_min_pixels=minimum,candidate_pixels=int(candidate.sum()),eligible_columns=int(keep.sum()))

def finalize_bridge(raw,candidate,neighbor_support,config):
 m=np.asarray(raw,dtype=bool);candidate=np.asarray(candidate,dtype=bool)
 if m.shape!=candidate.shape or np.any(candidate&m):raise ValueError('candidate must be outside same-shaped raw mask')
 cols=candidate.any(axis=0);coverage=np.ones(m.shape[1],dtype='float32')
 if config.require_temporal_support:
  if neighbor_support is None:cols[:]=False
  else:
   if neighbor_support.shape!=m.shape:raise ValueError('neighbor support shape mismatch')
   nearby=distance_transform_edt(~np.asarray(neighbor_support,dtype=bool))<=config.temporal_radius_px if np.any(neighbor_support) else np.zeros(m.shape,bool)
   counts=candidate.sum(axis=0);coverage=np.divide((candidate&nearby).sum(axis=0),counts,out=np.zeros(m.shape[1],dtype='float32'),where=counts>0)
   cols&=coverage>=config.temporal_coverage_fraction
 # Retain complete qualifying column bridges, never a pixelwise intersection that reopens the gap.
 cols=coherent_columns(cols,config,m.shape[1]);added=candidate&cols[None,:]
 status='prior_added' if added.any() else 'abstained_or_no_eligible_gap'
 if added.sum()>config.max_added_area_fraction*max(int(m.sum()),1):added[:]=False;status='abstained_added_area_limit'
 repaired=m|added
 assert not np.any(m&~repaired)
 return repaired,added,dict(status=status,added_pixels=int(added.sum()),added_fraction_of_raw=float(added.sum()/max(int(m.sum()),1)),qualifying_columns=int(cols.sum()),temporal_column_coverage_min=float(coverage[cols].min()) if cols.any() else None)

def repair_sequence(masks,sky_weight,config=None):
 c=config or ContinuityConfig();band,boundary=interface_geometry(sky_weight,c);candidates=[];details=[]
 for m in masks:
  a,d=candidate_bridge(m,sky_weight,band,c);candidates.append(a);details.append(d)
 results=[]
 for i,m in enumerate(masks):
  neighbors=[np.asarray(masks[j],bool)|candidates[j] for j in [i-1,i+1] if 0<=j<len(masks)]
  support=np.logical_or.reduce(neighbors) if neighbors else None
  repaired,added,d=finalize_bridge(m,candidates[i],support,c)
  results.append(dict(raw_mask=m,continuity_mask=repaired,continuity_added_mask=added,continuity_candidate_mask=candidates[i],metrics=dict(details[i],**d)))
 return results,dict(config=asdict(c),interface_boundary_pixels=int(boundary.sum()),temporal_interpretation='Eligibility from adjacent raw detections or independently eligible bridge candidates; not independent validation of added pixels')

def attach_continuity_qc(base,repaired,added,config):
 raw=np.asarray(base['plume_mask'],bool);repaired=np.asarray(repaired,bool);added=np.asarray(added,bool)
 if raw.shape!=repaired.shape or added.shape!=raw.shape or not np.array_equal(repaired,raw|added) or np.any(raw&added):raise ValueError('invalid additive repair')
 out=dict(base);out['plume_mask_sam3']=raw.copy();out['plume_mask']=repaired.copy();out['continuity_added_mask']=added.copy()
 out['quality_flag']=base['quality_flag'].copy();out['quality_flag'][added]|=CONTINUITY_INFERRED|np.uint32(4|8)
 # Added pixels have no quantitative retrieval; opacity stays missing and existing values stay exact.
 out['opacity']=base['opacity'].copy();out['opacity'][added]=np.nan
 out['retrieval_confidence']=base['retrieval_confidence'].copy();out['retrieval_confidence'][added]=0
 out['retrieval_regime']=base['retrieval_regime'].copy();out['retrieval_regime'][added]=4
 out['valid_opacity_mask']=base['valid_opacity_mask'].copy();out['valid_opacity_mask'][added]=False
 out['continuity_parameters']=config
 return out
