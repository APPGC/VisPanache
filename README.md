<p align="center"><img src="assets/VisPanache_logo.png" width="220" alt="VisPanache smoke plume and camera logo"></p>

# VisPanache v1.0.0

Private research snapshot for retrieving **apparent relative visible smoke opacity** from ground-based RGB imagery. The primary route combines the original terrain/sky hybrid retrieval with explicit pixel quality information. The original numerical retrieval and base QC modules are preserved byte for byte. This is an encoded-image quantity, not independently calibrated aerosol optical depth.

Algorithm development stops at this version. The original baseline is the primary result; the tested local seam-edge refinement is retained as an **optional, flagged companion**. The difficult top-camera scene remains a documented failure mode. No public disclosure or public licence is selected by this private snapshot.

## Reproduce the results

Download the matching private release asset `VisPanache_reproduction_inputs_v1.0.0.zip`, check its SHA256 against `data/archive_verification.json`, and extract it into a separate directory. It contains the actual RGB frames, saved masks, references, weights, prompts, model provenance and expected metrics for the paper windows. Large arrays are kept out of Git history.

Use Python 3.12 and the observed CPU dependency versions in `requirements_cpu_observed.txt`. After installing this package in your own environment, run:

```bash
python reproduction/run.py --data /path/to/extracted_data \
  --output /path/to/new_results --case all
```

For the verified quick check, including both refinement counterexamples:

```bash
python reproduction/run.py --data /path/to/extracted_data \
  --output /path/to/new_quick_results --case all --quick --refine-seam-edges
```

Every archived input is checked against its per-file checksum before processing. Output directories must be new. Each NetCDF is reopened and every array checked. See [reproduction instructions](reproduction/README.md) for the case definitions, optional refinement, figures and provenance limits.

## API and outputs

```python
from smoke_opacity.retrieval import retrieve, annotate_refinement
from smoke_opacity import to_quality_dataset

result = retrieve(rgb, background, plume_mask, sky_weight,
                  refine_seam_edges=False, **retrieval_parameters)
```

`opacity_baseline` preserves the original estimate. `opacity` screens it without changing accepted values. `quality_flag`, `retrieval_regime`, `retrieval_confidence` and `valid_opacity_mask` explain limitations and missing support. Confidence is an uncalibrated score, not a probability or error bar. RGB endpoint checks identify encoded clipping candidates, not calibrated sensor saturation.

With `refine_seam_edges=True`, additional `opacity_refined_raw`, `opacity_refined`, `quality_flag_refined`, `retrieval_confidence_refined`, increment and modified-mask fields are returned. The original fields remain intact. The option feathers the existing repair increment within three pixels, preserves existing repaired values, and never smooths the entire plume. Modified pixels receive bit 262144 and a confidence cap of 0.25. Existing screened-invalid pixels stay missing. The empirical correction can bias local opacity upward and does not establish physical accuracy.

The older `workflow.run_full_case()` and `retrieve_opacity_frame()` interfaces are preserved for compatibility but use a different normalized/interpolated retrieval. **They are not the paper reproduction route.** Use `reproduction/run.py`, the saved-data example or `smoke_opacity.retrieval.retrieve` with `configs/retrieval_paper.json`.

## Validation and scope

The paper baseline comprises 300 bottom-camera frames (000601–000900) and 22 top-camera frames (00355–00376). Guided full/lower top variants reuse those same 22 RGB images; they are not independent datasets. The baseline retains 94.28% and 93.69% of original-mask plume pixel-frames respectively. Accepted values equal the original retrieval exactly. Independent physical opacity accuracy has not been established.

The optional refinement reduces mean repair-edge jumps in 298/300 bottom frames, with a median reduction of 44.2%, modifying 0.60% of plume pixel-frames overall. Frames 893 and 895 are counterexamples and remain included. For the final matched MP4 comparison, all 291 available historical video-source image panels were reproduced pixel for pixel, and all 300 baseline/refinement pairs matched their validated numerical fields. No further parameter tuning is included in this version.

Core tests run without GPU or segmentation imports:

```bash
python -m unittest discover -s tests -v
```

The CPU saved-input route is tested from source in the recorded environment. No claim of a clean installation across other Python/library versions, whole-video segmentation, calibrated radiance, physical uncertainty or optical-depth validation is made.

## Segmentation and failure diagnostics

Saved SAM3 masks are supplied to reproduce the opacity results without a new model run. Current pinned top-video experiment sources/prompts/provenance are preserved under `experiments/gadi_top_video/`; the original historical mask-generation environment is not completely recoverable. SAM3 source and checkpoint access must be obtained separately. Installing the optional dependencies alone does not install or authenticate SAM3. No weights, Hugging Face caches or credentials are bundled.

The top case illustrates separate segmentation and radiometric failures. A connected mask can have disconnected screened opacity, and restoring clipped baseline zeros does not recover opacity continuity. A broader continuity prior is proposed for further evaluation; no validated repair is claimed. See [failure modes](docs/failure_modes.md) and [quality definitions](docs/quality.md).

## Code, data and logo

The intended destination is a private GitHub repository with the reproduction archive attached as a private release asset. Actual publication receipts and repository URL must be recorded after authenticated upload; a local tag does not establish remote publication. Archive/source hashes and lineage are retained. The logo is copied unchanged from the author's Desktop source. The original manuscript, images and historical outputs remain preserved.
