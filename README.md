# Maryam Realtime EEG Reconstruction

This project assembles the realtime EEG-to-image reconstruction runtime around
the hierarchical reconstruction models.

Runtime split:

- Fast path: trigger-locked EEG epoch -> preprocessing -> low-level encoder -> VAE decode -> immediate image.
- Slow path: the same EEG epoch -> ATMS / diffusion prior -> SDXL refinement in a background worker.
- Optional assessor path: reconstructed images -> affect ratings for valence/arousal and six affect dimensions.

## Repository Layout

- `src/maryam_rt/realtime`: streaming and preprocessing code.
- `src/maryam_rt/hierarchical`: copied hierarchical model code and support layers.
- `src/maryam_rt/integration`: runtime wrappers, replay runners, workers, assessor hooks.
- `scripts`: CLI entrypoints for GUI/replay/live modes.
- `docs`: status and engineering notes.
- `checkpoints`: local model weights. This is ignored by git.
- `data/derived`: local derived lab replay artifacts. This is ignored by git.
- `lab_data`: local raw participant/lab recordings. This is ignored by git.

## Requirements

Install the runtime dependencies in the active environment:

```bash
pip install -e ".[dev]"
```

For a minimal runtime-only install, use `pip install -r requirements.txt`.

## Model And Data Assets

Expected local assets:

```text
checkpoints/
  hierarchical/
    atms_starstim31_10sub_best.pth
    low_level_encoder_sub01_60.pth
    atms_sub01_40.pth
    prior_sub01_fdn.pt
  assessor/
    assessor_va_mixed_v4_bundle.pt
    assessor_six_clip_v3_bundle.pt
data/
  derived/
    lab_starstim31_epochs_250hz.npz
    finetune/
      lab_finetune_split_manifest.tsv
      lab_target_manifest.tsv
```

The assessor bundles contain the trained heads, calibration, conformal
intervals, and OOD statistics. They still need the timm CLIP backbone weights
for `vit_large_patch14_clip_224.openai` cached locally or downloadable on first
use.

## GUI Modes

Use `scripts/run_realtime_gui.py` for the browser monitor. It supports:

- `things-replay`: replays THINGS raw EEG from disk using true `stim` channel events.
- `lab-replay`: replays derived Starstim31 lab epochs from `data/derived`.
- `live`: reads EEG and marker LSL streams.
- `lab-live`: reads Starstim-style live EEG and marker LSL streams.

The browser runs at `http://127.0.0.1:<port>/`.

## THINGS Replay

From the repo root:

```bash
conda activate BCI
python scripts/run_realtime_gui.py \
  --mode things-replay \
  --data-root ../Hierarchical_EEG2Image_Reconstruction/data \
  --subject sub-01 \
  --session ses-01 \
  --split test \
  --disable-high-level \
  --max-trials 30 \
  --sleep-seconds 1.0 \
  --host 127.0.0.1 \
  --port 8010
```

With calibration block:

```bash
python scripts/run_realtime_gui.py \
  --mode things-replay \
  --data-root ../Hierarchical_EEG2Image_Reconstruction/data \
  --subject sub-01 \
  --session ses-01 \
  --split test \
  --disable-high-level \
  --calibration-block \
  --calibration-split training \
  --calibration-max-conditions 40 \
  --calibration-repetitions 3 \
  --calibration-sleep-seconds 0.5 \
  --max-trials 30 \
  --sleep-seconds 1.0 \
  --host 127.0.0.1 \
  --port 8010
```

Without `--disable-high-level`, the slow refinement worker also runs in the
background. That path requires the high-level checkpoint stack and SDXL assets.

## Lab Replay

Low-level only:

```bash
conda activate BCI
python scripts/run_realtime_gui.py \
  --mode lab-replay \
  --lab-data-root data \
  --lab-subject P07 \
  --lab-split all \
  --lab-low-level-checkpoint /path/to/starstim31_low_level.pth \
  --lab-model-channels 31 \
  --disable-high-level \
  --max-trials 30 \
  --sleep-seconds 1.0 \
  --host 127.0.0.1 \
  --port 8010
```

Direct non-GUI replay:

```bash
python scripts/run_lab_replay.py \
  --data-root data \
  --subject P07 \
  --split all \
  --lab-low-level-checkpoint /path/to/starstim31_low_level.pth \
  --lab-model-channels 31 \
  --disable-high-level \
  --max-trials 10
```

Notes:

- The default lab adapter is `none`; lab epochs stay Starstim31 native.
- The current derived lab epoch file is shaped `(N, 31, 250)` because preprocessing drops `Fz`.
- For an explicit old THINGS63 compatibility run, pass `--lab-adapter starstim31-to-things63`.

## Starstim31 ATMS Embeddings

Lab replay can export Starstim31 ATMS EEG embeddings:

```bash
python scripts/run_lab_replay.py \
  --data-root data \
  --subject P07 \
  --split all \
  --lab-low-level-checkpoint /path/to/starstim31_low_level.pth \
  --lab-model-channels 31 \
  --disable-high-level \
  --enable-lab-atms-embedding
```

Embeddings are written to:

```text
outputs/<run_name>/embeddings/*_starstim31_atms.pt
```

## Assessor Ratings

To rate reconstructed low-level and high-level images with the local VA and
six-dimension assessors, add:

```bash
--enable-assessor-ratings
```

Ratings are written to:

```text
outputs/<run_name>/assessments/*_assessor.json
```

Low-level ratings are also linked in the event metadata. High-level ratings
are written after the background refinement worker saves the refined image.

## Live Mock Mode

Terminal 1: start EEG stream:

```bash
python scripts/run_mock_streamer.py \
  --data-path ../Hierarchical_EEG2Image_Reconstruction/data/raw_eeg/sub-01/ses-01/raw_eeg_test.npy \
  --sampling-rate 1000 \
  --stream-name MockEEG
```

Terminal 2: start marker stream:

```bash
python scripts/run_mock_marker_streamer.py \
  --stream-name TaskMarkers \
  --marker stim_onset \
  --image-id 00154_sailboat \
  --interval-seconds 2
```

Terminal 3: start GUI:

```bash
python scripts/run_realtime_gui.py \
  --mode live \
  --eeg-stream-name MockEEG \
  --marker-stream-name TaskMarkers \
  --trigger-values stim_onset \
  --image-root ../Hierarchical_EEG2Image_Reconstruction/data/test_images \
  --disable-high-level
```

Markers can be plain strings or JSON payloads with `event`, `image_id`, or
`image_path` so the GUI can display the target image.

## Lab Live Mode

Use this for Starstim-style 32-channel live EEG:

```bash
python scripts/run_realtime_gui.py \
  --mode lab-live \
  --eeg-stream-name StarstimEEG \
  --marker-stream-name TaskMarkers \
  --trigger-values stim_onset \
  --lab-low-level-checkpoint /path/to/starstim32_low_level.pth \
  --lab-model-channels 32 \
  --image-root /path/to/object_images \
  --disable-high-level \
  --host 127.0.0.1 \
  --port 8010
```

Expected Starstim32 channel order:

```text
P8, T8, CP6, FC6, F8, F4, C4, P4,
AF4, Fp2, Fp1, AF3, Fz, FC2, Cz, CP2,
PO3, O1, Oz, O2, PO4, Pz, CP1, FC1,
P3, C3, F3, F7, FC5, CP5, T7, P7
```

## Current Blockers

See `docs/lab_realtime_pipeline_status.md` for the current lab pipeline status.
The project still needs a native Starstim31 low-level checkpoint and a
Starstim31-compatible diffusion prior before the lab GUI can run end to end.
