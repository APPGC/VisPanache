# Installation and runtime requirements

## Choose the workflow

| Work | Hardware | Python and access |
| --- | --- | --- |
| Reproduce opacity/QC from archived masks; render results | CPU; no GPU or CUDA required | Python 3.12; no SAM3 or Hugging Face account needed |
| Generate new SAM3 image/video masks, then retrieve opacity | Linux with a recent NVIDIA CUDA GPU | Python 3.12, separately installed SAM3 and approved Hugging Face checkpoint access |

Use **Python 3.12** for the documented setup. CPU reproduction was checked with Python **3.12.13**; successful GADI SAM3 jobs recorded **3.12.1**, PyTorch **2.7.1+cu126**, torchvision **0.22.1+cu126**, CUDA runtime **12.6** and an **NVIDIA A100-SXM4-80GB**. The GADI jobs loaded `python3/3.12.1` and `cuda/12.6.2`. The package metadata's broader `>=3.10` range is not a claim that the pinned environments were tested on older Python.

## CPU installation: existing masks

From the repository root, with Python 3.12 already available:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_cpu_observed.txt -e .
python -B -m unittest discover -s tests -v
```

Core libraries are **NumPy, SciPy, xarray, pandas, Matplotlib, Pillow, h5netcdf, h5py and PyYAML**. `requirements_cpu_observed.txt` records the exact tested CPU versions. Use it in its own environment; the GPU environment has a separate observed package snapshot.

Obtain and checksum-verify the separate reproduction archives, then run:

```bash
python reproduction/run.py --data /path/to/extracted_baseline_data \
  --output /path/to/new_results --case all --quick
```

Full saved-mask processing omits `--quick`; optional bottom-camera refinement adds `--refine-seam-edges`. See [reproduction instructions](../reproduction/README.md). FFmpeg is an additional system executable for MP4/movie export; it is not installed by pip. ImageIO is optional (`python -m pip install '.[images]'`). JupyterLab/ipykernel are only needed if choosing to work in notebooks; scripts and HTML reviews do not require them.

## GPU hardware for new SAM3 masks

Use an **Ampere-or-newer NVIDIA GPU with native BF16 support** for the documented reference route. Our successful model exports used **A100 80 GB**. On GADI, the A100 allocation used `dgxa100`; an alternate `gpuhopper` request targeted **H200**, but that request was not used for the validated model runs. H100/H200 are newer GPU candidates, not separately validated configurations in this release. Check current NCI queue policy and project access before requesting resources.

Older GADI **P100/V100** GPUs are not supported targets for this reference workflow. This does not mean SAM3 is universally limited to two GPU models: other modern NVIDIA GPUs may work, subject to architecture, drivers, dependencies and VRAM. We have not established a minimum VRAM requirement. **80 GB is the tested configuration, not a demonstrated minimum.** Memory demand depends on image dimensions, video length and predictor options.

The upstream [SAM3 prerequisites](https://github.com/facebookresearch/sam3#installation) list Python 3.12+, PyTorch 2.7+ and CUDA 12.6+. Our reproducible reference uses the pinned versions below rather than silently following today's upstream main branch or SAM3.1. A CUDA-enabled PyTorch wheel still requires a compatible NVIDIA driver on the host.

On GADI, obtain a GPU allocation before model loading or inference. Do not run it on a login node. A notebook inside an allocation, an interactive allocation, or an explicitly configured batch job can be used; VisPanache does not allocate a GPU automatically. Historical examples used one A100, 16 CPUs and 64 GB host RAM; these are resource requests, not measured minimum requirements.

## Hugging Face registration and checkpoint access

1. Create or sign in to a [Hugging Face account](https://huggingface.co/join).
2. Visit [facebook/sam3](https://huggingface.co/facebook/sam3), review/accept the model conditions and request access if prompted. Wait until the account has access. GitHub access does not grant Hugging Face access.
3. Create a token with **read access** to the gated model. Authenticate in your own terminal using `hf auth login`, installed by `huggingface_hub`. Enter the token at the prompt, and decline saving it as a Git credential. Follow the [official CLI guide](https://huggingface.co/docs/huggingface_hub/guides/cli).
4. Download the approved checkpoint once, then use its local path for inference. Weights and credential caches are excluded from this repository and the data archives.

Never put tokens in notebooks, shell command arguments, source files, job logs or Git. Keep the Hugging Face credential cache private to your account. Download where network access is available; compute-node internet access is not required when the checkpoint is cached locally. SAM3 model terms remain separate from VisPanache's licensing.

## Install the pinned SAM3 reference environment

Use a **separate Linux Python 3.12 environment** for segmentation. Commands below describe installation; they are not automatically run by this package. From the VisPanache repository root:

```bash
python3.12 -m venv .venv-sam3
source .venv-sam3/bin/activate
python -m pip install 'pip<26' 'setuptools<81' wheel
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements_sam3_observed.txt -e '.[segmentation]'
git clone https://github.com/facebookresearch/sam3.git /path/to/new_sam3_checkout
git -C /path/to/new_sam3_checkout checkout --detach f1647e8d0c1d280991f4d1b7d974c2704537fe66
python -m pip install --no-deps -e /path/to/new_sam3_checkout
python -m pip check
hf auth login
hf download facebook/sam3 sam3.pt \
  --revision 3c879f39826c281e95690f02c7821c4de09afae7 \
  --local-dir /path/to/model_directory
sha256sum /path/to/model_directory/sam3.pt
```

Expected checkpoint SHA256: `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`. The SAM3 Git source commit and Hugging Face model revision identify different repositories and must not be confused. Installing `.[segmentation]` supplies helper dependencies but **does not install SAM3 or obtain model access**. Transformers is not required by this pinned native-SAM3 route; substituting a Transformers implementation is not the reference reproduction.

Additional GPU/video dependencies include **torchvision, Triton, timm, Hugging Face Hub, einops, OpenCV-headless, scikit-image, ImageIO, decord, pycocotools, ftfy, regex, iopath, typing_extensions, psutil and tqdm**. `requirements_sam3_observed.txt` records selected installed versions read under the correct GADI modules on 30 October 2026. It is an observed snapshot, not a complete platform-independent lockfile or a clean-install verification. No new installation was performed for this documentation.

Inside a GPU allocation, check the environment before loading a model:

```bash
python - <<'PYGPU'
import torch
print('PyTorch:', torch.__version__, 'CUDA runtime:', torch.version.cuda)
assert torch.cuda.is_available(), 'No allocated CUDA GPU visible'
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))
print('Native BF16:', torch.cuda.is_bf16_supported())
assert torch.cuda.get_device_capability(0)[0] >= 8, 'Reference route expects Ampere or newer'
assert torch.cuda.is_bf16_supported(), 'Reference route expects BF16 support'
PYGPU
```

## Model execution and reproducibility limits

For the paper opacity calculations, use `smoke_opacity.retrieval.retrieve` or `reproduction/run.py` after masks are available. The generic image helper is in `smoke_opacity/segmentation.py`; the preserved video experiment in `experiments/gadi_top_video/` has deliberately fixed case paths, input checks and allocation guards. It must be adapted explicitly for a new dataset; it is not a general whole-video CLI. New imagery also needs a suitable smoke-free background, terrain/sky weighting and reviewed prompts/masks. Installing dependencies alone does not provide those inputs.

Our successful top and late-bottom video exports recorded `postprocess="historical_identity"`: native mask-hole filling failed with a Triton/CUDA error, and the bypass is explicit in the preserved runner/provenance. Do not describe native postprocessing as validated or assume an unchanged upstream model call reproduces those masks. Weights, exact source/checkpoint versions, prompts and postprocessing matter.

Saved masks make opacity/QC reproduction independent of GPU queue delays or new model inference. The oldest historical segmentation environment is incompletely recoverable; the documented current pinned experiment does not establish byte-identical regeneration of all historical masks. Software agreement does not establish independent segmentation or physical-opacity accuracy.
