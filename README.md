<p align="center"><img src="assets/VisPanache_logo.png" width="220" alt="VisPanache smoke plume and camera logo"></p>

# VisPanache v1.0.0

Private research snapshot for retrieving **apparent relative visible smoke opacity** from ground-based RGB imagery. The primary route combines the original terrain/sky hybrid retrieval with explicit pixel quality information. The original numerical retrieval and base QC modules are preserved byte for byte. This is an encoded-image quantity, not independently calibrated aerosol optical depth.

Algorithm development stops at this version. The original baseline is the primary result; the tested local seam-edge refinement is retained as an **optional, flagged companion**. The difficult top-camera scene remains a documented failure mode. No public disclosure or public licence is selected by this private snapshot.

## Bottom-camera example

![Visible timelapse alongside apparent smoke opacity superimposed on the RGB imagery](assets/bottom_camera_opacity_overlay.gif)

Bottom-camera frames 601–900: visible imagery (left) and apparent-opacity overlay (right), using the original 0–1 colour scale and the optional seam-edge refinement. All 300 frames are shown at 10 frames/s for inspection; playback does not establish observation cadence. This compact GIF is a display preview of the refined estimate; use the NetCDF quality fields for quantitative screening.

## Requirements and installation

**Opacity/QC from saved masks runs on CPU. New SAM3 segmentation requires a separate GPU environment and Hugging Face model access.**

| Workflow | Required environment |
| --- | --- |
| Saved-mask opacity, QC and paper reproduction | Python **3.12**; NumPy, SciPy, xarray, pandas, Matplotlib, Pillow, h5netcdf, h5py and PyYAML. No GPU, SAM3 or Hugging Face login required. |
| New SAM3 segmentation | Linux, recent NVIDIA CUDA GPU; tested on **A100 80 GB**, Python **3.12.1**, PyTorch **2.7.1+cu126**, torchvision **0.22.1+cu126**, CUDA **12.6**. Separately install the pinned SAM3 source and obtain approved access to [facebook/sam3](https://huggingface.co/facebook/sam3). |

For the GPU reference route, use Ampere or newer with native BF16 support. GADI's alternate H200 route was considered but not validated here; older P100/V100 GPUs are not supported targets for this workflow. A100 80 GB is the tested configuration, not a measured minimum VRAM requirement or a claim that only two GPU models can run SAM3.

See the **[complete setup guide](docs/installation.md)** for CPU/GPU installation commands, exact SAM3 source/checkpoint pins, Hugging Face registration and `hf auth login`, GPU checks, dependency lists, GADI allocation guidance and the recorded native-postprocessing limitation. CPU versions are in [requirements_cpu_observed.txt](requirements_cpu_observed.txt); the separate GPU snapshot is [requirements_sam3_observed.txt](requirements_sam3_observed.txt). FFmpeg is required for MP4 export; Jupyter is optional.

## Reproduce the results

Obtain the matching separately distributed data archive `VisPanache_reproduction_inputs_v1.0.0.zip`, check its SHA256 against `data/archive_verification.json`, and extract it into a separate directory. It contains the actual RGB frames, saved masks, references, weights, prompts, model provenance and expected metrics for the paper windows. Large arrays are kept out of Git history.

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

Code is prepared for a private GitHub repository; the reproduction frames and large arrays are kept in separate checksum-verified archives, prepared for an unpublished Zenodo dataset draft. Actual URLs and upload receipts must be recorded after authenticated upload; a local tag does not establish remote publication. Archive/source hashes and lineage are retained. The logo is copied unchanged from the author's Desktop source. The original manuscript, images and historical outputs remain preserved.

## Additional bright-sky case

Two additional eleven-frame bottom-camera windows (2495–2505 and 2595–2605) diagnose bright smoke and background-reference mismatch. CPU replay and figure scripts are in `reproduction/late_bright_sky/`; the separate case archive supplies the RGB frames, saved masks, references and expected products. These scripts preserve the frozen core and baseline sample counts. The original `source_manifest.json` applies to the v1.0.0 tag; see [release scope](docs/release_scope.md) for subsequent additions.
