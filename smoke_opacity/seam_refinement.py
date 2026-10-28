"""Feather repair edges while preserving every existing repaired value."""
import numpy as np
from .seam_feather import feather_repair,FeatherConfig

def feather_envelope(before,original,plume,sky_weight,config=None):
 blurred,info=feather_repair(before,original,plume,sky_weight,config)
 out=np.maximum(np.asarray(original),blurred)
 info=dict(info)
 info['feathered_repair_increment']=np.where(plume,out-before,0.).astype('float32')
 info['correction_modified_mask']=np.asarray(plume)&(out!=original)
 info['preserves_existing_repaired_values']=True
 return out,info
