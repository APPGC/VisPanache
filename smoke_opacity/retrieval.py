"""Frozen baseline API with an explicitly optional flagged refinement companion."""
from dataclasses import asdict
import json
import numpy as np
from .opacity import retrieve_hybrid_opacity
from .quality import retrieve_hybrid_opacity_qc
from .seam_feather import FeatherConfig
from .seam_refinement import feather_envelope
SEAM_EDGE_REFINEMENT=np.uint32(262144)

def retrieve(img,bg,plume,sky_weight,*,refine_seam_edges=False,config=None,
             segmentation_confidence=None,background_quality=None,
             background_inpaint_mask=None,**parameters):
    result=retrieve_hybrid_opacity_qc(img,bg,plume,sky_weight,config=config,
        segmentation_confidence=segmentation_confidence,background_quality=background_quality,
        background_inpaint_mask=background_inpaint_mask,**parameters)
    if not refine_seam_edges:
        return result
    kwargs=dict(parameters);kwargs['seam_repair']=False;kwargs['return_diagnostics']=False
    before=retrieve_hybrid_opacity(img,bg,plume,sky_weight,**kwargs)[0]
    settings=FeatherConfig(sigma_px=1.)
    raw,details=feather_envelope(before,result['opacity_baseline'],result['plume_mask'],result['diagnostics']['sky_weight'],settings)
    changed=details['correction_modified_mask'];valid=result['valid_opacity_mask']
    result['opacity_refined_raw']=raw
    result['opacity_refined']=np.where(valid,raw,np.nan).astype('float32')
    result['seam_refinement_mask']=changed
    result['seam_refinement_increment']=np.where(result['plume_mask'],raw-result['opacity_baseline'],np.nan).astype('float32')
    flag=result['quality_flag'].copy();flag[changed]|=SEAM_EDGE_REFINEMENT;result['quality_flag_refined']=flag
    score=result['retrieval_confidence'].copy();score[changed]=np.minimum(score[changed],.25);result['retrieval_confidence_refined']=score
    result['seam_refinement_parameters']=asdict(settings)
    return result

def annotate_refinement(dataset,result):
    if 'opacity_refined' not in dataset:return dataset
    dataset.attrs['seam_refinement_parameters']=json.dumps(result['seam_refinement_parameters'],sort_keys=True)
    dataset.attrs['seam_refinement_interpretation']='Optional empirical additive local correction; not independent physical validation. Original baseline, screened opacity, validity, masks and quality information remain intact.'
    dataset.opacity_refined.attrs.update(units='1',comment='Optional local seam-edge refinement on unchanged screened-validity support.',ancillary_variables='quality_flag_refined retrieval_confidence_refined seam_refinement_mask valid_opacity_mask')
    dataset.opacity_refined_raw.attrs.update(units='1',comment='Raw optional refinement, including unsupported original raw estimates; finite does not establish quantitative reliability.')
    dataset.quality_flag_refined.attrs=dict(dataset.quality_flag.attrs)
    dataset.quality_flag_refined.attrs['flag_masks']=np.r_[dataset.quality_flag.attrs['flag_masks'],SEAM_EDGE_REFINEMENT].astype('uint32')
    dataset.quality_flag_refined.attrs['flag_meanings']+=' local_seam_edge_refinement'
    dataset.retrieval_confidence_refined.attrs=dict(dataset.retrieval_confidence.attrs)
    dataset.retrieval_confidence_refined.attrs['comment']='Original uncalibrated screening score, capped at0.25 on locally modified pixels; never promoted.'
    return dataset
