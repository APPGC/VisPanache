# Release scope and provenance

The numerical algorithm remains frozen as v1.0.0 at commit `98b556fe2b1993c03621d5973ca18522eff66190`. `source_manifest.json` describes that historical frozen tree. Later commits add documentation and unchanged supplementary-case/figure reproduction scripts; they do not retune the retrieval. All files in `smoke_opacity/` remain byte-identical to the frozen tag.

The logo is retained. No public licence has been selected; this repository is intended to remain private. No authentication tokens, model weights, model caches or large image arrays belong in Git.

The repository includes CPU opacity/QC processing, saved-mask reproduction, prompt provenance and documented failure modes. SAM3 inference requires its separately obtained model, source and recorded runtime. Saved-mask reproduction does not require a GPU. Independent physical opacity accuracy and segmentation accuracy remain unvalidated.

The late bright-sky diagnostic uses its own `case_warning_flags` namespace. It does not silently alter production flags, calibrated uncertainty, the frozen baseline, or the original paper-window statistics. See `reproduction/late_bright_sky/README.txt`.
