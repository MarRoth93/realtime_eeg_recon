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
    lab_low_level_starstim31_best.pth
    lab_prior_starstim31_best_fdn.pt
    low_level_encoder_sub01_60.pth
    atms_sub01_40.pth
    prior_sub01_fdn.pt
  assessor/
    assessor_va_mixed_v4_bundle.pt
    assessor_six_clip_v3_bundle.pt
data/
  derived/
    lab_starstim31_epochs_250hz.npz
    live_preproc/
      lab_starstim31_epochs_250hz.npz
      lab_finetune_split_manifest.tsv
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
- `lab-replay`: replays Starstim lab recordings from raw `.easy` files by default, using the same preprocessing contract as `lab-live`.
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

By default, lab replay re-extracts raw Starstim32 epochs from each subject's
`.easy` file, then runs the same live preprocessing used by `lab-live`: drop
`Fz`, average reference, resample to 250 Hz, baseline-correct, and crop the
model-facing `31 x 250` window. The cached derived epoch file can still be used
for debugging with `--lab-replay-source derived` or `--replay-source derived`.

Low-level only:

```bash
conda activate BCI
python scripts/run_realtime_gui.py \
  --mode lab-replay \
  --lab-replay-source raw \
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
  --replay-source raw \
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
- Raw lab replay and live mode both produce `(31, 250)` model epochs with the shared live preprocessing path.
- The derived lab epoch file remains available as a cache/debug source with `--replay-source derived`.
- Native lab high-level needs `--lab-prior-checkpoint` and uses the Starstim31 ATMS checkpoint.
- For P07 high-level refinement, pass `--subject-id 4` so the ATMS subject conditioning matches the lab manifest.
- For an explicit old THINGS63 compatibility run, pass `--lab-adapter starstim31-to-things63`.

Native low-level plus native high-level:

```bash
python scripts/run_realtime_gui.py \
  --mode lab-replay \
  --lab-replay-source raw \
  --lab-data-root data \
  --lab-subject P07 \
  --lab-split all \
  --lab-low-level-checkpoint checkpoints/hierarchical/lab_low_level_starstim31_best.pth \
  --lab-prior-checkpoint checkpoints/hierarchical/lab_prior_starstim31_best_fdn.pt \
  --lab-model-channels 31 \
  --subject-id 4 \
  --max-trials 30 \
  --sleep-seconds 1.0 \
  --host 127.0.0.1 \
  --port 8010
```

## Starstim31 ATMS Embeddings

Lab replay can export Starstim31 ATMS EEG embeddings:

```bash
python scripts/run_lab_replay.py \
  --replay-source raw \
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

## Training Lab-Native Checkpoints

The lab-native low-level and prior checkpoints are trained in the reconstruction
repo:

```bash
cd /home/psycontrol/Marco/Maryam/Hierarchical_EEG2Image_Reconstruction
```

Build the raw/live-style lab training cache first:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/build_lab_live_preproc_cache.py
```

Train the Starstim31 low-level VAE-latent encoder:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/train_lab_low_level_vae.py \
  --experiment-name starstim31_livepreproc_sdxlvae \
  --heldout-subject P07 \
  --device cuda:0
```

Train the Starstim31-conditioned diffusion prior:

```bash
/home/psycontrol/miniforge3/envs/BCI/bin/python scripts/train_lab_diffusion_prior.py \
  --experiment-name starstim31_livepreproc_fdn \
  --heldout-subject P07 \
  --device cuda:0
```

With `--heldout-subject P07`, training uses all usable non-P07 lab trials and
evaluation uses usable P07 trials. Omit the flag only for a calibrated
multi-subject model where P07 is allowed to contribute training data.

Expected outputs:

```text
models/lab_low_level/starstim31_livepreproc/best.pth
models/lab_prior/starstim31_livepreproc/lab_prior_best_fdn.pt
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

For the new-participant demo, use the one-command launcher:

```bash
./scripts/start_live_demo.sh
```

It opens an interactive setup page that discovers LSL devices, validates the
32-channel/500 Hz Starstim stream, waits for a warmed EEG buffer, asks the
operator to confirm channel order, and exposes explicit Connect, Retry, Arm,
Disarm, Stop, and Disconnect controls. The runner waits for late equipment
instead of failing after ten seconds.

The launcher reads `config/lab_demo.json`. By default it uses the native plain
Starstim31 low-level checkpoint and treats the participant as unseen, so no
P07/known-subject ID is reused. Experimental high-level shared conditioning can
be enabled explicitly in that configuration.

Native `lab-live` applies the same model-facing preprocessing as raw replay:
drop `Fz`, average-reference, resample to 250 Hz, baseline-correct on
`-200..0 ms`, and crop `0..1 s` to `31 x 250`.

Native `lab-live` defaults to `--eeg-sampling-rate 500`, `--pre-event-ms 200`,
and `--post-event-ms 1000`. Override those only if the NIC2 LSL stream is
configured differently.

Expected Starstim32 channel order:

```text
P8, T8, CP6, FC6, F8, F4, C4, P4,
AF4, Fp2, Fp1, AF3, Fz, FC2, Cz, CP2,
PO3, O1, Oz, O2, PO4, Pz, CP1, FC1,
P3, C3, F3, F7, FC5, CP5, T7, P7
```

The complete setup sequence, marker contract, recovery behavior, model policy,
and hardware rehearsal are in `docs/live_demo_operator_guide.md`.
