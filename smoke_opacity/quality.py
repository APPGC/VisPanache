"""QC for the existing hybrid retrieval; no new opacity inversion.

Confidence is an uncalibrated screening score, not a probability or uncertainty.
The baseline remains available verbatim, including its original missing values.
"""
from dataclasses import asdict, dataclass
from enum import IntFlag
import hashlib
import inspect
import json

import numpy as np


class QualityFlag(IntFlag):
    RGB_CLIPPING = 1
    NEAR_CLIPPING = 2
    SEGMENTATION_UNASSESSED = 4
    BACKGROUND_UNASSESSED = 8
    AMBIGUOUS_REGIME = 16
    TERRAIN_SKY_TRANSITION = 32
    UNSTABLE_DENOMINATOR = 64
    INSUFFICIENT_SUPPORT = 128
    SEAM_REPAIRED = 256
    SIGN_FALLBACK = 512
    NONFINITE_INPUT = 1024
    BACKGROUND_INPAINTED = 2048
    BACKGROUND_UNRELIABLE = 4096
    SEGMENTATION_UNCERTAIN = 8192
    CALIBRATION_AT_LIMIT = 16384
    OPACITY_AT_BOUND = 32768


REGIME_MEANINGS = (
    "outside_plume terrain_brightening sky_attenuation "
    "bright_sky_plume_unsupported invalid_or_unreliable mixed_transition"
)


@dataclass(frozen=True)
class QCConfig:
    # Encoded RGB endpoints, not a claim about calibrated sensor saturation.
    clipping_level: float = 254.5 / 255
    near_clipping_level: float = 250 / 255
    contrast_tolerance: float = 3 / 255
    terrain_max_weight: float = 0.45
    sky_min_weight: float = 0.55
    transition_low_weight: float = 0.05
    transition_high_weight: float = 0.98
    min_terrain_headroom: float = 0.02
    min_sky_luminance: float = 0.02
    minimum_background_quality: float = 0.5
    minimum_segmentation_confidence: float = 0.5
    support_confidence_cap: float = 0.5
    unassessed_confidence_cap: float = 0.5
    transition_confidence_cap: float = 0.5
    near_clipping_confidence_cap: float = 0.25
    modified_confidence_cap: float = 0.25
    radiometric_representation: str = "encoded_rgb_uncalibrated"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, (float, int)) and (not np.isfinite(value) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be finite in [0,1]")
        if not 0 < self.near_clipping_level < self.clipping_level <= 1:
            raise ValueError("require 0 < near_clipping_level < clipping_level <= 1")
        if not 0 <= self.terrain_max_weight < self.sky_min_weight <= 1:
            raise ValueError("terrain_max_weight must be below sky_min_weight")
        if not 0 <= self.transition_low_weight < self.transition_high_weight <= 1:
            raise ValueError("transition interval is reversed")
        if self.radiometric_representation != "encoded_rgb_uncalibrated":
            raise ValueError("QC-only release preserves baseline encoded RGB; no linearization")


def _rgb01(rgb):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError("RGB must have shape (y,x,3)")
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255
    if not np.issubdtype(a.dtype, np.floating):
        raise ValueError("use uint8 RGB or floating RGB in [0,1]; other bit depths need explicit conversion")
    if np.any(np.isfinite(a) & ((a < 0) | (a > 1))) or np.isinf(a).any():
        raise ValueError("floating RGB must be in [0,1] or NaN")
    return a.astype(np.float32)


def radiometric_diagnostics(img, bg, config=None):
    """Scene-wide measurement checks; these fields do not identify smoke."""
    c = config or QCConfig()
    image, background = _rgb01(img), _rgb01(bg)
    if image.shape != background.shape:
        raise ValueError("image/background shapes differ")
    coeff = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    li, lb = image @ coeff, background @ coeff
    return {
        "observed_luminance": li,
        "background_luminance": lb,
        "luminance_difference": li - lb,
        "saturation_mask": np.any(image >= c.clipping_level, axis=-1),
        "background_saturation_mask": np.any(background >= c.clipping_level, axis=-1),
        "near_saturation_mask": np.any(image >= c.near_clipping_level, axis=-1),
        "background_near_saturation_mask": np.any(background >= c.near_clipping_level, axis=-1),
        "nonfinite_input_mask": ~np.all(np.isfinite(image) & np.isfinite(background), axis=-1),
    }


def _field(value, shape, name, boolean=False):
    a = np.asarray(value)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if boolean:
        if a.dtype != np.bool_ and not np.all(np.isfinite(a) & ((a == 0) | (a == 1))):
            raise ValueError(f"{name} must be Boolean or binary")
        return a.astype(bool)
    return a.astype(np.float32)


def _quality_field(value, shape, name):
    a = _field(value, shape, name)
    if np.any(np.isfinite(a) & ((a < 0) | (a > 1))) or np.isinf(a).any():
        raise ValueError(f"{name} must be in [0,1] or NaN")
    return a


def quality_from_diagnostics(img, bg, plume, opacity_baseline, diagnostics, *,
                             config=None, smoke_lum=0.95, min_support_pixels=20,
                             scale_clip=(0.25, 4.0), segmentation_confidence=None,
                             background_quality=None, background_inpaint_mask=None):
    """Screen an existing retrieval, retaining baseline and explanatory flags.

    Unknown ancillary reliability is flagged, not inferred from a binary mask.
    Regime 3 records observed sky brightening but intentionally has missing
    opacity: estimating bright-smoke source radiance is outside this release.
    Regime 5 explicitly represents blending rather than pretending one branch
    was used. Hard invalidity overrides regimes 1,2,3,5 with regime 4.
    """
    c = config or QCConfig()
    r = radiometric_diagnostics(img, bg, c)
    shape = r["observed_luminance"].shape
    m = _field(plume, shape, "plume", True)
    baseline = _field(opacity_baseline, shape, "opacity_baseline").copy()
    w = _field(diagnostics["sky_weight"], shape, "sky_weight")
    if np.any(np.isfinite(w) & ((w < 0) | (w > 1))):
        raise ValueError("sky_weight outside [0,1]")
    if not np.isfinite(smoke_lum) or not 0 < smoke_lum <= 1:
        raise ValueError("smoke_lum must be in (0,1]")
    if not isinstance(min_support_pixels, int) or min_support_pixels < 1:
        raise ValueError("min_support_pixels must be a positive integer")
    if len(scale_clip) != 2 or not np.all(np.isfinite(scale_clip)) or not 0 < scale_clip[0] < scale_clip[1]:
        raise ValueError("invalid scale_clip")
    flags = np.zeros(shape, dtype=np.uint32)
    def flag(condition, bit):
        flags[m & condition] |= np.uint32(bit)
    def diagnostic_bool(name):
        return _field(diagnostics[name], shape, name, True)

    delta = r["luminance_difference"]
    bright, dark = delta > c.contrast_tolerance, delta < -c.contrast_tolerance
    terrain, sky = w <= c.terrain_max_weight, w >= c.sky_min_weight
    transition = (w > c.transition_low_weight) & (w < c.transition_high_weight)
    regime = np.full(shape, 4, dtype=np.uint8)
    regime[m & terrain & bright] = 1
    regime[m & sky & dark] = 2
    regime[m & sky & bright] = 3
    regime[m & ~terrain & ~sky & (bright | dark)] = 5
    ambiguous = ~(bright | dark) | (terrain & dark) | (sky & bright) | ~np.isfinite(w)
    flag(ambiguous, QualityFlag.AMBIGUOUS_REGIME)
    flag(transition, QualityFlag.TERRAIN_SKY_TRANSITION)
    clipping = r["saturation_mask"] | r["background_saturation_mask"]
    near = r["near_saturation_mask"] | r["background_near_saturation_mask"]
    flag(clipping, QualityFlag.RGB_CLIPPING)
    flag(near & ~clipping, QualityFlag.NEAR_CLIPPING)
    # Assess denominator only where that branch contributes materially.
    denom_bad = (((w < c.transition_high_weight) & (smoke_lum - r["background_luminance"] <= max(0.02, c.min_terrain_headroom))) |
                 ((w > c.transition_low_weight) & (r["background_luminance"] <= c.min_sky_luminance)))
    flag(denom_bad, QualityFlag.UNSTABLE_DENOMINATOR)
    nonfinite = r["nonfinite_input_mask"] | ~np.isfinite(w) | ~np.isfinite(baseline)
    flag(nonfinite, QualityFlag.NONFINITE_INPUT)
    flag(np.isfinite(baseline) & ((baseline <= 0) | (baseline >= 1)), QualityFlag.OPACITY_AT_BOUND)

    score = np.ones(shape, dtype=np.float32)
    def cap(condition, value):
        score[m & condition] = np.minimum(score[m & condition], value)
    hard_bad = clipping | denom_bad | nonfinite | (baseline < 0) | (baseline > 1)
    ancillary = {}
    for value, name, unknown_bit, bad_bit, minimum in [
        (segmentation_confidence, "segmentation_confidence", QualityFlag.SEGMENTATION_UNASSESSED,
         QualityFlag.SEGMENTATION_UNCERTAIN, c.minimum_segmentation_confidence),
        (background_quality, "background_quality", QualityFlag.BACKGROUND_UNASSESSED,
         QualityFlag.BACKGROUND_UNRELIABLE, c.minimum_background_quality),
    ]:
        a = np.full(shape, np.nan, np.float32) if value is None else _quality_field(value, shape, name)
        ancillary[name] = a.copy()
        unknown, bad = ~np.isfinite(a), np.isfinite(a) & (a < minimum)
        flag(unknown, unknown_bit)
        flag(bad, bad_bit)
        cap(unknown, c.unassessed_confidence_cap)
        score = np.minimum(score, np.where(np.isfinite(a), a, 1))
        hard_bad |= bad
    if background_inpaint_mask is not None:
        inpaint = _field(background_inpaint_mask, shape, "background_inpaint_mask", True)
        ancillary["background_inpaint_mask"] = inpaint.copy()
        flag(inpaint, QualityFlag.BACKGROUND_INPAINTED)
        cap(inpaint, c.modified_confidence_cap)

    nt = int(diagnostics["n_terrain_support"])
    ns = int(diagnostics["n_sky_support"])
    if nt != int(diagnostic_bool("terrain_support").sum()) or ns != int(diagnostic_bool("sky_support").sum()):
        raise ValueError("support counts do not match diagnostic masks")
    # Counts are global: this flag does not certify local support density.
    insufficient = transition & ((nt < min_support_pixels) | (ns < min_support_pixels))
    flag(insufficient, QualityFlag.INSUFFICIENT_SUPPORT)
    cap(insufficient, c.support_confidence_cap)
    scale = _field(diagnostics["sky_to_terrain_scale"], shape, "sky_to_terrain_scale")
    at_limit = transition & ((scale <= scale_clip[0] + 1e-6) | (scale >= scale_clip[1] - 1e-6))
    flag(at_limit, QualityFlag.CALIBRATION_AT_LIMIT)
    cap(at_limit, c.modified_confidence_cap)
    for name, bit in [("seam_gap", QualityFlag.SEAM_REPAIRED), ("use_fallback", QualityFlag.SIGN_FALLBACK)]:
        changed = diagnostic_bool(name)
        flag(changed, bit)
        cap(changed, c.modified_confidence_cap)
    cap(transition, c.transition_confidence_cap)
    cap(near, c.near_clipping_confidence_cap)
    regime[m & hard_bad] = 4
    valid = m & ((regime == 1) | (regime == 2) | (regime == 5)) & ~hard_bad
    opacity = np.where(valid, baseline, np.nan).astype(np.float32)
    score[m & ~valid] = 0
    score[~m] = np.nan
    regime[~m] = 0
    return dict(r, **ancillary, qc_parameters=asdict(c), opacity=opacity, opacity_baseline=baseline, plume_mask=m,
                retrieval_regime=regime, retrieval_confidence=score,
                quality_flag=flags, valid_opacity_mask=valid)


def retrieve_hybrid_opacity_qc(img, bg, plume, sky_weight, *, config=None,
                               segmentation_confidence=None, background_quality=None,
                               background_inpaint_mask=None, retrieval=None, **retrieval_kwargs):
    """Call the established retrieval once and attach QC; never change baseline."""
    if retrieval is None:
        from .opacity import retrieve_hybrid_opacity
        retrieval = retrieve_hybrid_opacity
    kwargs = dict(retrieval_kwargs)
    kwargs["return_diagnostics"] = True
    alpha, terrain, sky, diag = retrieval(img, bg, plume, sky_weight, **kwargs)
    defaults = inspect.signature(retrieval).parameters
    def parameter(name):
        return kwargs.get(name, defaults[name].default)
    result = quality_from_diagnostics(
        img, bg, plume, alpha, diag, config=config,
        smoke_lum=parameter("smoke_lum"), min_support_pixels=parameter("min_support_pixels"),
        scale_clip=parameter("scale_clip"), segmentation_confidence=segmentation_confidence,
        background_quality=background_quality, background_inpaint_mask=background_inpaint_mask,
    )
    result["retrieval_parameters"] = {name: kwargs.get(name, p.default) for name, p in defaults.items()
                                      if p.default is not inspect.Parameter.empty}
    result["diagnostics"] = diag
    result["alpha_terrain_baseline"] = terrain
    result["alpha_sky_baseline"] = sky
    return result


def to_quality_dataset(result, *, config=None, provenance):
    """One-frame NetCDF-ready dataset; callers supply source/camera/frame IDs.

    Time is deliberately not inferred from frame indices or directory labels.
    """
    import xarray as xr
    recorded = result.get("qc_parameters")
    c = config or (QCConfig(**recorded) if recorded is not None else QCConfig())
    if recorded is not None and recorded != asdict(c):
        raise ValueError("serialization config differs from the config used for QC")
    required = {"camera_id", "frame_id", "source_version", "environment", "input_paths"}
    missing = required - provenance.keys()
    if missing:
        raise ValueError(f"missing provenance keys: {sorted(missing)}")
    variables = {}
    for name, value in result.items():
        if isinstance(value, np.ndarray) and value.ndim == 2:
            a = value.astype(np.uint8) if value.dtype == bool else value
            variables[name] = (("y", "x"), a)
    ds = xr.Dataset(variables)
    ds.attrs.update(
        description="QC screening of apparent relative visible smoke opacity",
        qc_version="0.1.0", qc_config=json.dumps(asdict(c), sort_keys=True),
        provenance=json.dumps(provenance, sort_keys=True),
        radiometric_representation=c.radiometric_representation,
        retrieval_parameters=json.dumps(result.get("retrieval_parameters", {}), sort_keys=True),
        confidence_interpretation="uncalibrated heuristic screening score; not a probability or error bar",
        validation_level="software QC; independent physical validation not established",
        qc_source_sha256=hashlib.sha256(inspect.getsource(inspect.getmodule(to_quality_dataset)).encode()).hexdigest(),
    )
    for name in ds:
        ds[name].attrs.update(units="1", long_name=name.replace("_", " "))
    ds.opacity.attrs.update(valid_min=0., valid_max=1., ancillary_variables="quality_flag retrieval_regime retrieval_confidence valid_opacity_mask",
                           comment="Missing outside plume and for unsupported/invalid retrievals; not aerosol optical depth")
    ds.opacity_baseline.attrs["comment"] = "Unmodified legacy result; finite does not imply valid"
    ds.retrieval_regime.attrs.update(flag_values=np.arange(6, dtype=np.uint8), flag_meanings=REGIME_MEANINGS)
    ds.quality_flag.attrs.update(flag_masks=np.array([int(f) for f in QualityFlag], dtype=np.uint32),
                                flag_meanings=" ".join(f.name.lower() for f in QualityFlag))
    ds.retrieval_confidence.attrs.update(valid_min=0., valid_max=1., comment=ds.attrs["confidence_interpretation"])
    for name in ds:
        if name.endswith("mask"):
            ds[name].attrs.update(flag_values=np.array([0, 1], np.uint8), flag_meanings="false true")
    ds.saturation_mask.attrs["comment"] = "Any observed RGB channel at configured encoded endpoint; clipping candidate, not calibrated sensor saturation"
    return ds
