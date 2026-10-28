# Reproduction from the private v1.0.0 inputs

The matching data ZIP contains 66 checked files (593,382,222 archive bytes). Its SHA256 is `876e8da94e308c35d2f6ba36f9da25b184a10d9d09baaf88a3c0216ba1730d36`. The archive is separate from source Git history; GitHub blocks ordinary Git files over 100 MiB and supports release assets under 2 GiB. See [GitHub file limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) and [release assets](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).

## Cases

- `bottom`: original 300 RGB frames and masks, reference frame 418, frozen background and sky weights.
- `top_original`: original 22 RGB frames and masks, cleaned background and inpainting indicator.
- `top_guided`: the same 22 RGB frames with the successful full-guidance video masks.
- `top_lower_ablation`: those same RGB frames with lower-positive guidance; prompt sensitivity is not cloud truth.

`--case all` runs these four configurations (366 frame/configuration products, not 366 independent observations). The main baseline statistics use only `bottom` and `top_original`. The later two configurations support the difficult-case/prompt analysis. The optional `--refine-seam-edges` flag applies only to the bottom case; top processing remains unchanged.

`--quick` selects bottom frames 700, 817, 893 and 895 plus top frame 376 in each top configuration. It is the release verification check, not the complete baseline analysis. The full CPU run may take several minutes. No new GPU allocation, download or model execution is implicit.

Use `examples/retrieve_saved_sequence.py --help` for direct processing of a chosen frame list. All paths are supplied by the caller. The source/mask checksums, exact parameters and actual environment are recorded in each output. The input manifest's original local `source` paths are provenance only; processing uses its relative archive paths.

## Figures and movies

`original_renderer.py` contains the exact original notebook RGB-overlay function. The original `inferno_r` scale, fixed 0–1 limits and opacity-dependent display transparency are preserved. This renderer affects appearance only. Use `render_movies.py` with the data and the `bottom` output directory to create original-style baseline and optional refined MP4s. FFmpeg must already be available; the script never installs it. Comparison movies use ten frames per second for inspection, not as a claim about observation cadence.

The historical reference video contains overlay indices 0–290, corresponding to frames 601–891. The complete saved window continues to frame 900; do not silently exclude its last nine frames, which include the refinement counterexamples.

## What is and is not reproduced

RGB frames and masks are archived inputs, so opacity, QC, refinement, statistics and rendering can be reproduced without SAM3. Current model/prompt provenance and both top mask variants are included for audit. Regenerating segmentation requires separately obtained weights and the recorded source/runtime; the older historical mask environment is incompletely recorded. Full original MP4s and full 3016/633-image sequences are outside this data asset because the paper's evaluated windows are supplied directly.

Expected baseline/refinement/ablation statistics are under `expected/` in the data archive, with corresponding validation summaries. Numerical comparisons must distinguish raw opacity, screened support, original versus guided masks and repeated configurations. NaNs, reason flags and low confidence must not be interpreted as no smoke. Passing software checks does not establish independently calibrated opacity accuracy.
