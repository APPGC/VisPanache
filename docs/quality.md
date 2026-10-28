# Pixel quality definitions

The normal package API and reproduction command are documented in the root README. This specification describes the unchanged base quality layer.

## Outputs

All pixel variables use image coordinates `(y,x)` and dimensionless units `1`.

| Variable | Interpretation |
|---|---|
| `opacity` | Screened apparent relative visible opacity; NaN outside plume and in unsupported/invalid pixels |
| `opacity_baseline` | Unmodified legacy output, retained for comparison; finite does not certify validity |
| `retrieval_regime` | Classification below; not a physical aerosol classification |
| `retrieval_confidence` | Heuristic screening score in [0,1]; zero for unsupported/invalid plume pixels, NaN outside plume |
| `quality_flag` | uint32 bitmask; several reasons can apply at one pixel; zero outside plume |
| `valid_opacity_mask` | Finite screened-opacity support; not independent scientific validation |
| `saturation_mask` | Any observed RGB channel at the configured encoded clipping endpoint, scene-wide |
| `background_saturation_mask` | Corresponding background clipping candidate, scene-wide |
| `near_saturation_mask` | Observed channel at/above the near-clipping threshold, including clipping |
| `background_near_saturation_mask` | Corresponding background near-clipping indicator |
| `nonfinite_input_mask` | Nonfinite image or background channel |
| `observed_luminance`, `background_luminance`, `luminance_difference` | Encoded RGB weighted brightness diagnostics; not calibrated radiance |

The wrapper also returns the original branch fields, full diagnostic dictionary, actual retrieval parameters, supplied background/segmentation quality maps and inpainting mask. Missing ancillary scores are exported as NaN. Serialization uses the QC settings actually used and rejects inconsistent settings. `to_quality_dataset()` serializes the two-dimensional fields, QC settings, actual retrieval parameters, flag definitions, source hash and caller-supplied provenance. Required provenance keys are camera ID, frame ID, source version, environment and input paths. No acquisition time is inferred from indices or directory names.

### Regimes

| Value | Meaning |
|---|---|
| 0 | Outside the supplied plume mask; opacity undefined |
| 1 | Terrain-like background with positive contrast |
| 2 | Sky-like background with negative contrast |
| 3 | Sky-like background with positive contrast; bright-sky plume unsupported, opacity missing |
| 4 | Invalid, unreliable or ambiguous retrieval; opacity missing |
| 5 | Intermediate terrain/sky weight with resolved contrast; legacy mixed retrieval retained, transition flagged |

Regime 5 is an explicit addition to the handover proposal: the hybrid blend should not be presented as a pure terrain or sky retrieval. Hard failures override other classifications with regime 4. Regime 3 reports an observed radiometric condition, not proof that scattering alone caused it. Membership always requires the supplied plume mask, so bright clouds outside that mask are not labelled smoke by this layer.

### Quality bits

| Mask | Meaning |
|---|---|
| 1 | Observed or background RGB clipping candidate |
| 2 | Near clipping without clipping |
| 4 | Segmentation reliability unassessed at this pixel |
| 8 | Background reliability unassessed at this pixel |
| 16 | Incompatible/ambiguous radiometric regime or unresolved contrast |
| 32 | Terrain/sky transition |
| 64 | Insufficient terrain headroom or insufficient sky background brightness |
| 128 | Too few terrain or sky support pixels globally for interface calibration |
| 256 | Baseline seam repair applied |
| 512 | Baseline sign-aware fallback applied |
| 1024 | Nonfinite image/background/weight/baseline value |
| 2048 | Background pixel reconstructed by inpainting |
| 4096 | Supplied background quality below threshold |
| 8192 | Supplied segmentation confidence below threshold |
| 16384 | Calibration scale at configured clipping bound near interface |
| 32768 | Legacy opacity at/beyond its lower or upper bound; distinct from RGB clipping |

Support bit 128 assesses global support counts, not local density. The baseline also propagates local support and uses global fallbacks; this release does not certify the physical validity or local representativeness of that calibration.

## Rules and defaults

The defaults are screening choices, not calibrated sensor specifications. All can be set through `QCConfig` unless stated otherwise. Threshold sensitivity must be reported before using them as final publication settings.

| Parameter | Default | Meaning |
|---|---|---|
| `clipping_level` | 254.5/255 | Detects the endpoint 255 in uint8 inputs |
| `near_clipping_level` | 250/255 | Conservative low-headroom warning |
| `contrast_tolerance` | 3/255 | Contrast within this interval is unresolved, not assigned zero opacity |
| `terrain_max_weight`, `sky_min_weight` | 0.45, 0.55 | Terrain/sky-like classification |
| `transition_low_weight`, `transition_high_weight` | 0.05, 0.98 | Material hybrid transition |
| `min_terrain_headroom` | 0.02 | Minimum smoke-reference minus background brightness; cannot relax the legacy hard-coded 0.02 guard |
| `min_sky_luminance` | 0.02 | Avoid dark-background attenuation instability |
| `minimum_background_quality` | 0.5 | Supplied lower quality makes opacity missing |
| `minimum_segmentation_confidence` | 0.5 | Supplied lower confidence makes opacity missing |
| `unassessed_confidence_cap` | 0.5 | Missing ancillary reliability cannot get score 1 |
| `support_confidence_cap` | 0.5 | Insufficient global calibration support |
| `transition_confidence_cap` | 0.5 | Hybrid transition |
| `near_clipping_confidence_cap` | 0.25 | Low headroom |
| `modified_confidence_cap` | 0.25 | Inpainting, seam repair, fallback or bounded calibration |
| `radiometric_representation` | `encoded_rgb_uncalibrated` | Preserves baseline representation; linearization is not introduced |

Score construction starts at 1, takes the minimum of supplied ancillary scores, and applies the applicable caps. Hard-invalid and unsupported regimes receive 0. It is not a posterior probability, error bar or independently calibrated uncertainty. A binary plume mask is not segmentation confidence. An attractive inpainted background is not measured background quality. Passing an object-level SAM score as a pixel probability would require additional justification.

Clipping, unstable denominators, nonfinite values and explicitly low ancillary reliability make opacity missing. Sky brightening, terrain darkening and unresolved contrast are unsupported in this release. Near clipping, inpainting and interface modifications reduce the heuristic score and set flags. Opacity reaching 1 is flagged separately and is not automatically interpreted as sensor clipping.

RGB input must be uint8 or floating point in [0,1]. Other integer bit depths and float [0,255] inputs are rejected rather than silently rescaled. NaNs are handled as invalid; infinities and malformed arrays are rejected. Encoded clipping does not establish where the camera sensor clipped: tone mapping, colour conversion, video legal range and compression can hide sensor clipping below the encoded endpoint. This release preserves the existing weighted encoded-RGB computation and must not be described as calibrated radiance or optical depth.

