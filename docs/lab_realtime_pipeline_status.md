# Lab Realtime EEG Reconstruction Status

## Current Branch

- Branch: `lab_data_setup`
- Project root: `/home/marco/Marco/realtime_eeg_recon`
- Lab data copied into the project under `data/derived`
- Current working subject: `P07`

## Available And Verified

- P07 lab replay data loads from `data/derived`.
  - Total P07 trials: 200
  - Train: 156
  - Val: 44
  - P07 `lab_subject_id`: 4
- Starstim31 ATMS checkpoint is present:
  - `checkpoints/hierarchical/atms_starstim31_10sub_best.pth`
- Starstim31 ATMS embedding path is now wired for lab replay.
  - Input: `(31, 250)` Starstim31 EEG epoch
  - Output: `(1, 1024)` EEG embedding
  - Smoke-tested on a P07 epoch.
- Runtime environment issue was fixed by matching PyTorch and torchvision.
  - Verified versions:
    - `torch 2.12.1+cu130`
    - `torchvision 0.27.1+cu130`
    - `numpy 1.26.4`
- SDXL VAE cache is available and verified with `local_files_only=True`.
- VA and six-dimension affect assessor bundles were copied into this project:
  - `checkpoints/assessor/assessor_va_mixed_v4_bundle.pt`
  - `checkpoints/assessor/assessor_six_clip_v3_bundle.pt`
- Reconstructed image assessor hooks are available for low-level and high-level outputs.
  - Target and low-level ratings are written into event metadata and sidecar JSON files.
  - High-level ratings are written as sidecar JSON files when refinement completes.
  - When target, low-level, and high-level images are all available, a final summary figure is generated.

## New Embedding Hook

Direct replay and GUI lab replay can export Starstim31 ATMS embeddings with:

```bash
--enable-lab-atms-embedding
```

Optional flags:

```bash
--lab-atms-checkpoint checkpoints/hierarchical/atms_starstim31_10sub_best.pth
--lab-atms-subject-id 4
```

If `--lab-atms-subject-id` is omitted, the replay code uses `lab_subject_id` from the lab split manifest.

Embeddings are written under:

```text
outputs/<run_name>/embeddings/*_starstim31_atms.pt
```

Each metadata JSON also gets a `starstim31_atms_embedding` entry.

## New Assessor Hook

Direct replay and GUI modes can rate reconstructed images with:

```bash
--enable-assessor-ratings
```

Optional flags:

```bash
--assessor-va-bundle checkpoints/assessor/assessor_va_mixed_v4_bundle.pt
--assessor-six-bundle checkpoints/assessor/assessor_six_clip_v3_bundle.pt
--assessor-device cuda
--assessor-interval-level 0.9
--disable-assessor-ood
```

Assessor sidecars are written under:

```text
outputs/<run_name>/assessments/*_assessor.json
```

Completed target/low/high triplets also write:

```text
outputs/<run_name>/assessments/*_assessor_summary.png
outputs/<run_name>/assessments/*_assessor_summary.json
```

The summary figure shows the original target image, low-level reconstruction,
high-level reconstruction, a target/low/high score matrix, and mean absolute
affect distance from each reconstruction to the target.

The copied bundles contain the VA and six-dimension regression heads, calibration,
conformal interval widths, and OOD statistics. They still require the CLIP
ViT-L/14 timm backbone weights (`vit_large_patch14_clip_224.openai`) to be cached
or downloadable at runtime; those weights were not present inside the source
assessor project.

## Still Missing

- Native Starstim31 low-level EEG-to-VAE-latent checkpoint.
  - Needed for: P07/Starstim31 EEG epoch -> low-level image.
  - The existing `checkpoints/hierarchical/low_level_encoder_sub01_60.pth` is THINGS63, not native Starstim31.
- Starstim31-compatible diffusion prior / reconstruction checkpoint.
  - Needed for: ATMS EEG embedding -> high-level semantic image reconstruction.
  - The existing `checkpoints/hierarchical/prior_sub01_fdn.pt` is from the old THINGS path and should not be treated as the correct Starstim31 prior.
- Full SDXL high-level cache may still need verification if the full-model download was not completed.
  - VAE-only cache is verified.
  - Full high-level generation also needs SDXL UNet, text encoders, tokenizers, scheduler, and config.
- Full offline self-containment for the assessor still needs the CLIP ViT-L/14
  backbone weights copied or cached locally.

## Current Pipeline State

```text
P07 EEG epoch -> Starstim31 ATMS embedding              works
Starstim32 live epoch -> offline-style 31x250 input     wired for native lab-live
SDXL VAE loading from cache                             works
target/low/high images -> VA/six ratings + final summary figure wired, needs CLIP backbone cache
P07 EEG epoch -> low-level image                        blocked by missing native low-level checkpoint
ATMS embedding -> high-level semantic reconstruction    blocked by missing Starstim31 diffusion prior
GUI end-to-end display                                  blocked until the two project checkpoints above exist and are wired in
```

## Useful Verification Commands

Verify SDXL VAE cache:

```bash
python - <<'PY'
import numpy
import torch
import torchvision
from diffusers import AutoencoderKL

print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"numpy: {numpy.__version__}")

AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="vae",
    local_files_only=True,
)

print("SDXL VAE loads from cache")
PY
```

Verify Starstim31 ATMS embedding:

```bash
PYTHONPATH=src python - <<'PY'
from pathlib import Path
from maryam_rt.integration.lab_replay import LabReplayConfig, load_lab_trials
from maryam_rt.integration.starstim31_atms import Starstim31ATMSEmbedder

trial = load_lab_trials(LabReplayConfig(data_root=Path("data"), subject="P07"))[0]
embedder = Starstim31ATMSEmbedder(
    Path("checkpoints/hierarchical/atms_starstim31_10sub_best.pth"),
    device="cpu",
)
embedding = embedder.embed(trial.epoch_lab31, subject_id=trial.lab_subject_id)
print(tuple(embedding.shape), "subject_id", trial.lab_subject_id)
PY
```

Expected output shape:

```text
(1, 1024)
```
