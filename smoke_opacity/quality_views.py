"""Optional diagnostic views; strict retrieval validity always stays explicit."""
import numpy as np
from .quality import QualityFlag

def retain_flagged_boundary(result, boundary_mask):
    """Re-display finite baseline estimates on ambiguous boundary pixels.

    This never restores clipping, unstable/nonfinite or explicitly unreliable
    measurements. It never changes strict opacity, flags, confidence, regime,
    masks or validity. Finite companion values can still be unsupported.
    """
    raw = np.asarray(result['plume_mask'], dtype=bool)
    boundary = np.asarray(boundary_mask)
    if boundary.shape != raw.shape or boundary.dtype != np.dtype(bool):
        raise ValueError('boundary_mask must be a same-shaped boolean array')
    flags = np.asarray(result['quality_flag'])
    baseline = np.asarray(result['opacity_baseline'])
    strict = np.asarray(result['opacity'])
    valid = np.asarray(result['valid_opacity_mask'], dtype=bool)
    regime = np.asarray(result['retrieval_regime'])
    for a in [flags, baseline, strict, valid, regime]:
        if a.shape != raw.shape:
            raise ValueError('inconsistent result shapes')
    hard = int(QualityFlag.RGB_CLIPPING | QualityFlag.UNSTABLE_DENOMINATOR |
               QualityFlag.NONFINITE_INPUT | QualityFlag.BACKGROUND_UNRELIABLE |
               QualityFlag.SEGMENTATION_UNCERTAIN)
    retained = (raw & boundary & ~valid & ((flags & int(QualityFlag.AMBIGUOUS_REGIME)) > 0)
                & ((flags & hard) == 0) & np.isfinite(baseline)
                & (baseline >= 0) & (baseline <= 1) & np.isin(regime, [3, 4]))
    provisional = strict.copy()
    provisional[retained] = baseline[retained]
    out = dict(result)
    out['opacity_with_flagged_boundary'] = provisional
    out['boundary_estimate_retained_mask'] = retained
    return out

def annotate_boundary_view(dataset):
    """Describe the companion explicitly after normal dataset serialization."""
    dataset.opacity_with_flagged_boundary.attrs.update(
        units='1',
        ancillary_variables='quality_flag retrieval_regime retrieval_confidence valid_opacity_mask boundary_estimate_retained_mask',
        comment='Provisional baseline estimates retained for flagged display. Finite values may remain unsupported/ambiguous with zero confidence and valid_opacity_mask=false; no interpolation or new inversion.')
    dataset.boundary_estimate_retained_mask.attrs['comment'] = 'Original baseline value shown again; strict opacity and retrieval validity are unchanged.'
    return dataset
