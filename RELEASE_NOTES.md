# Private v1.0.0 freeze

The freeze preserves the original hybrid retrieval and its exact replay settings, adds explicit quality information, and retains the tested local seam-edge adjustment as an optional companion. The baseline remains primary; algorithm tuning stops here. Original outputs are not replaced.

Thirty package tests and an archive-based seven-frame reproduction check cover the default CPU API, optional refinement, clipping/ambiguity, missing support, unsigned quality metadata and NetCDF roundtrips. The earlier full-window validation covers 300 bottom and 22 original top frames, with accepted numerical values unchanged. The final matched movies reproduce all 291 historical overlay-source image panels exactly and retain all 300 saved frames, including counterexamples.

Data and code are kept separate. The private reproduction archive supplies actual RGB/masks, references, weights, both guided top variants, prompt/model provenance and expected metrics. No credentials, environment directories or model weights are included. A local frozen snapshot is not evidence that the private GitHub repository or release asset has been published; record that separately after authenticated upload.

Known limitations include uncalibrated encoded-RGB brightness, empirical gains, unassessed background/segmentation reliability, unsupported bright-sky regimes, residual interface structure, the local correction's possible upward bias and incomplete provenance of the older segmentation runtime. The top camera remains a failure-mode study; continuity priors are proposed rather than claimed solved. Internal visual/numerical consistency is not independent physical opacity validation.
