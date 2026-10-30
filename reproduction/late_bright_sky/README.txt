VisPanache v1.0.0 supplementary bright-sky case — private snapshot 29 October 2026

This case extends the paper with two 11-frame bottom-camera windows 2495–2505 and2595–2605. It does not change frozen v1.0.0 or the300 bottom / 22 top baseline statistics. The bundle contains 22 original RGB frames, supplied background/weight arrays,22 SAM3 masks plus source/model/prompt provenance, expected per-pixel NetCDF products, the frozen source archive and companion reference-diagnostic code. Model weights, authentication data and caches are excluded. Access remains private.

For CPU reproduction, extract frozen_source/VisPanache_v1.0.0_source.zip into a chosen directory. Locate its smoke_opacity directory and pass that parent as --source. With the recorded NumPy/SciPy/Pillow/xarray/h5netcdf environment, run:
python code/test_reference_diagnostics.py
python code/replay_late_case.py --source /path/to/extracted/source --bundle . --output /new/empty/output

The script uses the frozen public retrieve() API and archived masks and compares 15 array fields per frame exactly, including NaNs, flags and contrast diagnostics. This checks saved-mask CPU reproduction; it does not rerun SAM3, validate mask accuracy or calibrate physical opacity. A different numerical environment may require investigation if bitwise agreement changes. Source and dependency provenance are recorded alongside the archive.

Products retain opacity_baseline, opacity, quality_flag, retrieval_regime and uncalibrated retrieval_confidence. New case_warning_flags use a separate uint16 namespace:1sky_brightening,2low_archived_contrast,4reference_control_shift,8reference_sign_disagreement,16encoded_clipping,32near_encoded_clipping,64insufficient_control_support. Multiple warnings can coexist. Zero outside case_evaluated_mask means UNASSESSED. Original regime and confidence fields refer to the frozen retrieval; no additional calibrated score is asserted.

The assessed domain is savedSAM3 mask AND smoothed skyweight>=0.55 AND image rows 80:220 (stop exclusive). Same-row candidate controls occupy columns 1080:1280, excluding predicted plume and non-sky;80% candidate support is required. Contrast tolerance3/255; control-reference-shift threshold10/255. ReferenceConfig documents all defaults; sensitivity tests use1/3/5encoded levels and three control intervals 1020:1170,1080:1280,1200:1320. Neighbouring controls are not independently verified smoke-free backgrounds and are not replacements in the opacity formula. Changing clouds, illumination and camera response are not uniquely separated.

opacity_case_screened is an exploratory conservative companion: missing values additionally replace estimates with reference-shift/sign-disagreement/insufficient-control warnings in assessed support. Outside that support, frozen QC remains unchanged and reference consistency remains unassessed. Invalid values are never replaced by zero. The full original estimate and original flags remain available. Input values are encoded RGB, not calibrated linear radiance.

Masks primarily cover the elevated plume and do not exhaustively label diffuse margins or low fire-line smoke. The guided anchor masks are 10.2% / 16.5% smaller than median neighbouring masks and have ragged edges. This is an internal sensitivity observation, not a claim of segmentation accuracy. The diagnostic figures and paper percentages use conditional upper-band mask support; they must not be substituted for whole-plume or earlier manually selected rectangle fractions.

For the two publication figures, run:
python code/plot_late_case.py --bundle . --output /new/empty/figure-output
This was tested to regenerate both central-frame PNGs exactly in the recorded environment. Other baseline manuscript figures require the separately preserved VisPanache_reproduction_inputs_v1.0.0.zip.

This data-only supplement excludes manuscript drafts, model weights, credentials and caches. Obtain the frozen source via the v1.0.0 Git tag or preserved source ZIP.
