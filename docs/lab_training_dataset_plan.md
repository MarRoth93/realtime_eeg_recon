# Lab Training Dataset Plan

## Goal

Prepare one reproducible training dataset contract for native Starstim31
reconstruction checkpoints used by the realtime pipeline.

This step must be completed before training either:

- the low-level EEG-to-SDXL-VAE-latent encoder, or
- the Starstim31 diffusion prior.

## Current Source Artifacts

Use the local derived lab artifacts in this repo as the canonical source:

```text
data/derived/lab_starstim31_epochs_250hz.npz
data/derived/finetune/lab_finetune_split_manifest.tsv
data/derived/finetune/lab_target_manifest.tsv
```

The sibling `Hierarchical_EEG2Image_Reconstruction` checkout currently has the
training code, but no local `data/` files in this working tree. Its existing
local model files are ATMS checkpoints, not low-level VAE-latent or diffusion
prior checkpoints.

## Verified Current Dataset Shape

Current lab epoch artifact:

```text
eeg:       (1200, 31, 250), float32
times:     0.000 .. 0.996 seconds, 250 samples
channels:  P8, T8, CP6, FC6, F8, F4, C4, P4,
           AF4, Fp2, Fp1, AF3, FC2, Cz, CP2,
           PO3, O1, Oz, O2, PO4, Pz, CP1, FC1,
           P3, C3, F3, F7, FC5, CP5, T7, P7
```

Current finetune manifest:

```text
rows:              1200
subjects:          P04, P05, P06, P07, melina, zar
rows per subject:  200
split:             936 train, 264 val
usable rows:       1171
usable train rows: 915
usable val rows:   256
unique targets:    160
```

Each subject currently contributes:

```text
156 train rows
44 val rows
```

## Important Blocker

The manifest image paths currently point at:

```text
/media/psycontrol/HDD/Datasets/THINGS/images_THINGS/...
```

Those files are not available on this workstation right now. Before generating
SDXL VAE latent targets, resolve this by either:

- mounting the original THINGS image root at the expected path, or
- adding a deterministic path-rewrite step from manifest `mapped_path` /
  `exact_image_path` to the actual local image root.

The path-rewrite choice must be recorded in the generated dataset metadata.

## Training Variants

Create two variants with identical rows, splits, target images, and channel
order.

### Variant A: Baseline Only

Use the existing model-facing epochs directly:

```text
data/derived/lab_starstim31_epochs_250hz.npz
```

Contract:

```text
eeg: (N, 31, 250)
```

This variant matches the current realtime `lab-live` preprocessing contract:

```text
raw Starstim32 epoch
-> drop Fz
-> average reference
-> resample to 250 Hz
-> baseline correct over -200..0 ms
-> crop 0..1 s
-> 31 x 250
```

### Variant B: MVNN Whitened

Status: keep this as a future experiment for now. The current lab replay
checkpoints were not trained with MVNN-whitened EEG as their input contract, so
enabling replay-time MVNN whitening can create a train/inference mismatch even
if the whitening matrix is numerically valid. For the current GUI runs, prefer
the baseline-only path and omit `--lab-whitening mvnn`.

Compute MVNN from train rows only, then apply the same whitening matrix to
train and val rows.

Use only rows where `usable_for_finetune == True`.

Required saved artifacts:

```text
data/derived/training_variants/mvnn_whitened/lab_starstim31_epochs_250hz_mvnn.npz
data/derived/training_variants/mvnn_whitened/whitening_matrix.npy
data/derived/training_variants/mvnn_whitened/preprocessing_metadata.json
```

Contract:

```text
eeg:              (N, 31, 250)
whitening_matrix: (31, 31)
```

Realtime inference for an MVNN-trained checkpoint must apply:

```text
x_model = whitening_matrix @ x_baseline_only
```

where `x_baseline_only` has exactly the same channel order and shape as the
baseline-only variant.

## Planned Output Layout

```text
data/derived/training_variants/
  baseline_only/
    lab_starstim31_epochs_250hz_baseline.npz
    lab_training_manifest.tsv
    preprocessing_metadata.json
  mvnn_whitened/
    lab_starstim31_epochs_250hz_mvnn.npz
    lab_training_manifest.tsv
    whitening_matrix.npy
    preprocessing_metadata.json
```

Both variant manifests must contain the same logical rows:

```text
row_id
epoch_index
subject_folder
lab_subject_id
split
target_index
trigger
category
target_image_path
usable_for_finetune
```

## Verification Checks

Run these checks before model training:

```text
baseline eeg shape == mvnn eeg shape
baseline manifest rows == mvnn manifest rows
baseline row_id list == mvnn row_id list
baseline split counts == mvnn split counts
baseline subject counts == mvnn subject counts
baseline channel names == mvnn channel names
all target_image_path files exist
whitening_matrix shape == (31, 31)
whitening_matrix is finite
no train/val leakage in MVNN fitting
```

Expected row counts after filtering to usable rows:

```text
all:   1171
train: 915
val:   256
```

## Next Implementation Step

Add a small dataset-preparation script that:

1. reads the current NPZ and finetune manifest,
2. filters to `usable_for_finetune == True`,
3. resolves local target image paths,
4. writes the baseline-only variant,
5. computes train-only MVNN whitening,
6. writes the MVNN variant and whitening metadata,
7. runs the verification checks above.

The script should fail fast if target images are missing. Training should not
start until the image-root issue is resolved.
