# Lab Realtime EEG Reconstruction Status

## Current Branch

- Branch: `lab_data_setup`
- Project root: `/home/psycontrol/Marco/Maryam/01_maryam_realtime`
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

Lab replay now defaults to raw recorded Starstim replay:

```bash
--replay-source raw
```

or, in GUI mode:

```bash
--lab-replay-source raw
```

This path reads each subject's `.easy` recording, extracts the raw Starstim32
epoch around the recorded marker timing, and applies the same live preprocessing
used by `lab-live`. The old cached `data/derived/lab_starstim31_epochs_250hz.npz`
path remains available with `derived` for debugging.

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

## Remaining External Validation

- Native checkpoints are now present at:
  - `checkpoints/hierarchical/lab_low_level_starstim31_best.pth`
  - `checkpoints/hierarchical/lab_prior_starstim31_best_fdn.pt`
- The interactive live-demo controller, numeric trigger mapping, unseen/shared
  ATMS mode, and per-session output layout are implemented.
- The conservative unseen-participant default is low-level only. Shared-token
  high-level reconstruction is available but remains explicitly experimental
  until its downstream diffusion output is evaluated end to end.
- NIC2 stream discovery, channel order, marker timing, browser rendering, and
  reconnect behavior still require an in-room hardware rehearsal.
- Full offline assessor use still requires the CLIP ViT-L/14 backbone weights to
  be cached locally.

## Current Pipeline State

```text
P07 EEG epoch -> Starstim31 ATMS embedding              works
Starstim32 live/raw-replay epoch -> live-style 31x250 input wired for native lab-live/lab-replay
SDXL VAE loading from cache                             works
target/low/high images -> VA/six ratings + final summary figure wired, needs CLIP backbone cache
P07 EEG epoch -> low-level image                        works with native lab checkpoint
ATMS embedding -> high-level semantic reconstruction    works for known IDs; shared unseen mode is experimental
GUI end-to-end replay                                   works
GUI live setup / Arm / session lifecycle                implemented; hardware-room validation remains
```

## Reproducing The Lab Checkpoints

Run these from:

```bash
cd /home/psycontrol/Marco/Maryam/Hierarchical_EEG2Image_Reconstruction
```

First build the raw/live-style lab cache. This reads `.easy`, drops `Fz`,
average-references, resamples to 250 Hz, baseline-corrects, crops `0..1 s`,
and writes a manifest whose `epoch_index` values match the new cache:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/build_lab_live_preproc_cache.py
```

Then train the native Starstim31 low-level checkpoint:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/train_lab_low_level_vae.py \
  --experiment-name starstim31_livepreproc_sdxlvae \
  --heldout-subject P07 \
  --device cuda:0
```

Expected stable output:

```text
models/lab_low_level/starstim31_livepreproc/best.pth
```

Then train the Starstim31-conditioned diffusion prior. This keeps the existing
`best_model.pth` ATMS checkpoint frozen and trains the prior against
`lab_vith14_features.pt`:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/train_lab_diffusion_prior.py \
  --experiment-name starstim31_livepreproc_fdn \
  --heldout-subject P07 \
  --device cuda:0
```

With `--heldout-subject P07`, the training set uses all usable lab trials from
the other subjects and evaluation uses usable P07 trials. That is the
no-subject-leakage setup for testing later on P07.

Expected stable output:

```text
models/lab_prior/starstim31_livepreproc/lab_prior_best_fdn.pt
```

After training, copy or point realtime to:

```text
checkpoints/hierarchical/lab_low_level_starstim31_best.pth
checkpoints/hierarchical/lab_prior_starstim31_best_fdn.pt
```

Native lab high-level is now allowed when `--lab-prior-checkpoint` is provided.
For P07, use `--subject-id 4` so high-level conditioning matches the manifest
subject ID.

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

trial = load_lab_trials(LabReplayConfig(data_root=Path("data"), subject="P07", max_trials=1))[0]
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
