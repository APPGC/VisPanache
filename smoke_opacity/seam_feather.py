"""Experimental smoothing of the repair increment, never the full opacity field."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt
@dataclass(frozen=True)
class FeatherConfig:
 sigma_px:float=1.
 radius_sigma:float=3.
 terrain_edge_weight:float=.05
 sky_edge_weight:float=.02
 def __post_init__(self):
  if not np.isfinite(self.sigma_px) or self.sigma_px<=0:raise ValueError('sigma_px must be positive')
  if not np.isfinite(self.radius_sigma) or self.radius_sigma<=0:raise ValueError('radius_sigma must be positive')
  if not 0<self.terrain_edge_weight<.5 or not 0<self.sky_edge_weight<.5:raise ValueError('invalid edge weights')
def smoothstep(v):
 v=np.clip(v,0.,1.);return v*v*(3.-2.*v)
def feather_repair(before,original,plume,sky_weight,config=None):
 c=config or FeatherConfig();pre=np.asarray(before);old=np.asarray(original);m=np.asarray(plume);w=np.asarray(sky_weight)
 if pre.ndim!=2 or any(a.shape!=pre.shape for a in [old,m,w]) or m.dtype!=bool:raise ValueError('same-shaped 2D arrays and boolean plume required')
 if not np.isfinite(w).all() or np.any((w<0)|(w>1)):raise ValueError('invalid sky weights')
 if not np.array_equal(np.isfinite(pre),np.isfinite(old)) or np.any(m&(~np.isfinite(pre))):raise ValueError('consistent finite retrieval support required')
 delta=np.where(m,old-pre,0.).astype('float32')
 if np.any(delta[m]<-1e-6):raise ValueError('original repair must be additive')
 delta=np.maximum(delta,0);seeds=m&(delta>0);out=old.copy();change=np.zeros(pre.shape,bool)
 if not seeds.any():return out,dict(original_repair_increment=delta,feathered_repair_increment=delta.copy(),correction_modified_mask=change,feather_config=asdict(c))
 domain=m&(distance_transform_edt(~seeds)<=c.radius_sigma*c.sigma_px)
 numerator=gaussian_filter(delta,c.sigma_px,mode='constant',cval=0,truncate=c.radius_sigma)
 denominator=gaussian_filter(m.astype('float32'),c.sigma_px,mode='constant',cval=0,truncate=c.radius_sigma)
 feathered=np.divide(numerator,denominator,out=np.zeros(pre.shape,'float32'),where=denominator>1e-6)
 taper=smoothstep(w/c.terrain_edge_weight)*smoothstep((1.-w)/c.sky_edge_weight)
 feathered=np.where(domain,feathered*taper,0.).astype('float32')
 # All original repair pixels are within domain. Away from domain, old equals pre.
 out[domain]=np.clip(pre[domain]+feathered[domain],0.,1.)
 change=m&(out!=old)
 assert np.array_equal(out[~domain],old[~domain],equal_nan=True)
 return out,dict(original_repair_increment=delta,feathered_repair_increment=feathered,correction_modified_mask=change,feather_domain_mask=domain,feather_config=asdict(c))
