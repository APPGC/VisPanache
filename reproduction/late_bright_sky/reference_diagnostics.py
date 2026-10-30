"""Exploratory sidecar checks, separate from frozen production quality_flag.

Same-row sky controls are consistency diagnostics, not replacement backgrounds.
No plume identification or physical opacity inversion is performed here.
"""
from dataclasses import dataclass,asdict
import numpy as np
@dataclass(frozen=True)
class ReferenceConfig:
    row_start:int=80
    row_stop:int=220
    control_start:int=1080
    control_stop:int=1280
    minimum_control_fraction:float=.8
    contrast_tolerance:float=3/255
    reference_shift_tolerance:float=10/255
    sky_min_weight:float=.55
    clipping_level:float=254.5/255
    near_clipping_level:float=250/255
FLAG_MASKS=np.array([1,2,4,8,16,32,64],dtype='uint16')
FLAG_MEANINGS='sky_brightening low_archived_contrast reference_control_shift reference_sign_disagreement encoded_clipping near_encoded_clipping insufficient_control_support'
def assess_reference(image,background,plume,weight,config=None):
    c=config or ReferenceConfig();a=np.asarray(image,dtype='float32');b=np.asarray(background,dtype='float32');m=np.asarray(plume,dtype=bool);w=np.asarray(weight,dtype='float32')
    if a.ndim!=3 or a.shape[-1]!=3 or a.shape!=b.shape or a.shape[:2]!=m.shape or w.shape!=m.shape:raise ValueError('image/mask shape mismatch')
    h,width=m.shape
    if not 0<=c.row_start<c.row_stop<=h or not 0<=c.control_start<c.control_stop<=width:raise ValueError('diagnostic support outside image')
    if not .0<c.minimum_control_fraction<=1 or not 0<c.contrast_tolerance<1 or not 0<c.reference_shift_tolerance<1:raise ValueError('invalid thresholds')
    k=np.array([.2126,.7152,.0722],dtype='float32');li=a@k;lb=b@k;delta=li-lb
    rowdomain=np.zeros(m.shape,bool);rowdomain[c.row_start:c.row_stop]=True
    domain=m&(w>=c.sky_min_weight)&rowdomain
    cc=slice(c.control_start,c.control_stop)
    candidate=(~m[:,cc])&(w[:,cc]>=c.sky_min_weight)&np.isfinite(li[:,cc])&np.isfinite(lb[:,cc])
    counts=candidate.sum(1);supported=counts>=int(np.ceil((c.control_stop-c.control_start)*c.minimum_control_fraction))
    current=np.full(h,np.nan,dtype='float32');prior=current.copy()
    for row in np.flatnonzero(supported):
        current[row]=np.median(li[row,cc][candidate[row]])
        prior[row]=np.median(lb[row,cc][candidate[row]])
    shift=current-prior;local=li-current[:,None];valid=domain&supported[:,None]
    flags=np.zeros(m.shape,dtype='uint16')
    def add(cond,bit):flags[domain&cond]|=np.uint16(bit)
    add(delta>c.contrast_tolerance,1);add(np.abs(delta)<=c.contrast_tolerance,2)
    add(supported[:,None]&(np.abs(shift[:,None])>c.reference_shift_tolerance),4)
    add(valid&(delta<-c.contrast_tolerance)&(local>c.contrast_tolerance),8)
    maximum=a.max(-1);add(maximum>=c.clipping_level,16);add((maximum>=c.near_clipping_level)&(maximum<c.clipping_level),32)
    add(~supported[:,None],64)
    return dict(case_warning_flags=flags,case_evaluated_mask=domain,reference_comparison_mask=valid,archived_reference_contrast=np.where(domain,delta,np.nan),same_row_control_contrast=np.where(valid,local,np.nan),control_reference_shift=np.where(valid,np.broadcast_to(shift[:,None],m.shape),np.nan),control_support_count=counts,parameters=asdict(c))
