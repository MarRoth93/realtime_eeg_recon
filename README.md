# Maryam Realtime V1

This project is a first V1 assembly of the live EEG streaming runtime and the hierarchical reconstruction models.

V1 split:
- fast path: trigger-locked EEG epoch -> preprocessing -> low-level encoder -> VAE decode -> immediate image
- slow path: the same trigger-locked epoch -> ATMS -> diffusion prior -> SDXL refinement in a background worker

Current scope:
- task-triggered epoch inference from an EEG LSL stream plus a marker LSL stream
- mock or real LSL EEG input
- mock or real LSL marker input
- saved low-level outputs in `outputs/low_level`
- saved refined outputs in `outputs/high_level`
- browser monitor for EEG, markers, target image, and low-level reconstruction

Not implemented yet:
- channel remapping validation for a real amplifier montage

Structure:
- `src/maryam_rt/realtime`: copied runtime streaming and preprocessing code
- `src/maryam_rt/hierarchical`: copied hierarchical model code and support layers
- `src/maryam_rt/integration`: V1 wrappers and background worker
- `checkpoints/hierarchical`: ATMS, low-level encoder, and prior checkpoints

Quick start:

1. Install dependencies from `requirements.txt`.
2. Start an EEG stream:
   `python scripts/run_mock_streamer.py --data-path /path/to/eeg.npy --sampling-rate 1000 --stream-name MockEEG`
3. Start a marker stream:
   `python scripts/run_mock_marker_streamer.py --stream-name TaskMarkers --marker stim_onset --interval-seconds 2`
4. Run the V1 pipeline:
   `python scripts/run_realtime_v1.py --eeg-stream-name MockEEG --marker-stream-name TaskMarkers --trigger-values stim_onset`

Browser GUI

There are two different GUI launch modes. They serve different purposes.

- one-terminal `things-replay` mode:
  reads the THINGS raw EEG files directly, uses the true `stim` channel onsets from the dataset, and maps each event to the correct target image automatically
- one-terminal `lab-replay` mode:
  reads derived Starstim31 lab epochs from this project's `data/derived` folder and replays them through the same browser reconstruction stack
- three-terminal `live` mode:
  simulates the final online setup with a separate EEG LSL stream and a separate marker LSL stream
- three-terminal `lab-live` mode:
  simulates a future Starstim lab stream with 32 incoming channels and a native 32-channel lab low-level checkpoint

When to use which

- use the one-terminal method when you want the correct THINGS dataset demo
  this is the right choice for prerecorded THINGS test data because it uses the real dataset event timing
- use the three-terminal method when you want to test the live system architecture
  this is the right choice for checking whether EEG streaming, marker streaming, and the GUI work together like a future real experiment
- use `lab-replay` when you want to test the local Starstim lab recordings after they have already been extracted into model-ready epochs
- use `lab-live` when you want to test the live architecture with Starstim-style 32-channel input

Important difference

- one-terminal `things-replay`:
  one process does everything from the THINGS files
  EEG comes from disk, event onsets come from the THINGS `stim` channel, and target images are matched automatically
- three-terminal `live`:
  one process streams EEG, one process streams markers, and one process runs the GUI/reconstruction
  event onsets come from the marker stream, not from the THINGS `stim` channel inside the EEG file
- lab modes:
  lab input is Starstim-native, not THINGS63. `lab-live` receives 32 incoming Starstim channels. The current derived `lab-replay` artifact is 31-channel because the existing preprocessing drops `Fz`. By default, lab modes require a native 31/32-channel lab low-level checkpoint. The old `starstim31-to-things63` path is now explicit compatibility mode only.

# One-terminal THINGS replay demo

Use this when you want the correct THINGS replay with true dataset event timing.

```bash
conda activate BCI
python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py   --mode things-replay   --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data   --subject sub-01   --session ses-01   --split test   --disable-high-level   --max-trials 30   --sleep-seconds 1.0   --host 127.0.0.1   --port 8010
```

Then open `http://127.0.0.1:8010/`.

Notes for one-terminal mode:
- this is the recommended mode for your THINGS test data demo
- the EEG panel shows a replayed moving window around each true THINGS event
- if you start at very early trials, the first seconds may have a shorter window because the recording has less pre-event data available at the beginning

## With visual caibration
```bash
python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode things-replay \
    --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data \
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


## Without Visual calibration
```bash
python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/
  scripts/run_realtime_gui.py \
      --mode things-replay \
      --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data \
      --subject sub-01 \
      --session ses-01 \
      --split test \
      --disable-high-level \
      --calibration-block \
      --calibration-split training \
      --calibration-max-conditions 40 \
      --calibration-repetitions 3 \
      --max-trials 30 \
      --sleep-seconds 1.0 \
      --host 127.0.0.1 \
      --port 8010
```


## With high level recon
```bash
python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode things-replay \
    --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data \
    --subject sub-01 \
    --session ses-01 \
    --split test \
    --calibration-block \
    --calibration-split training \
    --calibration-max-conditions 40 \
    --calibration-repetitions 3 \
    --max-trials 30 \
    --sleep-seconds 1.0 \
    --host 127.0.0.1 \
    --port 8010
```


# Three-terminal live mock demo

Use this when you want to simulate the future real online setup with separate EEG and marker streams.

1. Terminal 1: start the EEG stream
   ```bash
   conda activate BCI
   python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_mock_streamer.py      --data-path /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data/raw_eeg/sub-01/ses-01/raw_eeg_test.npy      --sampling-rate 1000      --stream-name MockEEG
   ```

2. Terminal 2: start the marker stream
   ```bash
   conda activate BCI
   python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_mock_marker_streamer.py      --stream-name TaskMarkers      --marker stim_onset      --image-id 00154_sailboat      --interval-seconds 2
   ```

3. Terminal 3: start the browser GUI
   ```bash
   conda activate BCI
   python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py      --mode live      --eeg-stream-name MockEEG      --marker-stream-name TaskMarkers      --trigger-values stim_onset      --image-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data/test_images      --disable-high-level
   ```

4. Open `http://127.0.0.1:8000/` in the browser.

Notes for three-terminal mode:
- this is the recommended mode for architecture testing, not for the most accurate THINGS replay
- if the marker stream sends fake markers every 2 seconds, the event timing is artificial
- to make this mode scientifically match THINGS replay, the marker stream would need to send the true THINGS event times

Live target image support:
- if the marker is a plain string like `stim_onset`, the GUI can trigger reconstruction but cannot infer the shown image
- if the marker is JSON with `event`, `image_id`, or `image_path`, the GUI can show the target image next to the low-level reconstruction
- `run_mock_marker_streamer.py` now supports `--image-id`, `--image-path`, or `--payload-json`

# One-terminal lab replay demo

Use this when you want to replay the local Starstim lab recordings after preprocessing.

Expected data layout under this project:

```text
data/
  derived/
    lab_starstim31_epochs_250hz.npz
    finetune/
      lab_finetune_split_manifest.tsv
      lab_target_manifest.tsv
```

Those files correspond to the lab artifacts produced in the main reconstruction project. The epoch file should contain `(N, 31, 250)` Starstim31 epochs.

Low-level only:

```bash
conda activate BCI
python /home/psycontrol/Marco/Maryam/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode lab-replay \
    --lab-data-root /home/psycontrol/Marco/Maryam/01_maryam_realtime/data \
    --lab-split val \
    --lab-low-level-checkpoint /path/to/starstim31_low_level.pth \
    --lab-model-channels 31 \
    --disable-high-level \
    --max-trials 30 \
    --sleep-seconds 1.0 \
    --host 127.0.0.1 \
    --port 8010
```

Then open `http://127.0.0.1:8010/`.

Direct non-GUI replay:

```bash
python /home/psycontrol/Marco/Maryam/01_maryam_realtime/scripts/run_lab_replay.py \
    --data-root /home/psycontrol/Marco/Maryam/01_maryam_realtime/data \
    --split val \
    --lab-low-level-checkpoint /path/to/starstim31_low_level.pth \
    --lab-model-channels 31 \
    --disable-high-level \
    --max-trials 10
```

Notes for lab replay:
- the default adapter is `none`, so the lab epoch tensor is not remapped into THINGS63
- the current derived artifact is `lab_starstim31_epochs_250hz.npz`, shaped `(N, 31, 250)`
- if you want to run the existing THINGS-trained checkpoint only as an integration demo, pass `--lab-adapter starstim31-to-things63`

# Three-terminal lab live mock demo

Use this when the incoming EEG stream is Starstim-style 32-channel data in this order:

```text
P8, T8, CP6, FC6, F8, F4, C4, P4,
AF4, Fp2, Fp1, AF3, Fz, FC2, Cz, CP2,
PO3, O1, Oz, O2, PO4, Pz, CP1, FC1,
P3, C3, F3, F7, FC5, CP5, T7, P7
```

```bash
conda activate BCI
python /home/psycontrol/Marco/Maryam/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode lab-live \
    --eeg-stream-name StarstimEEG \
    --marker-stream-name TaskMarkers \
    --trigger-values stim_onset \
    --lab-low-level-checkpoint /path/to/starstim32_low_level.pth \
    --lab-model-channels 32 \
    --image-root /media/psycontrol/HDD/Datasets/THINGS/images_THINGS/object_images \
    --disable-high-level \
    --host 127.0.0.1 \
    --port 8010
```

For `lab-live`, marker payloads can include `image_id` or `image_path` JSON fields so the GUI can display the target image.

Compatibility-only THINGS63 lab demo:

```bash
python /home/psycontrol/Marco/Maryam/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode lab-live \
    --eeg-stream-name StarstimEEG \
    --marker-stream-name TaskMarkers \
    --trigger-values stim_onset \
    --lab-adapter starstim31-to-things63 \
    --disable-high-level \
    --host 127.0.0.1 \
    --port 8010
```

Raw THINGS demo:

1. Use the true `stim` channel events in the raw THINGS files:
   `python scripts/run_things_raw_demo.py --data-rootpython /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode things-replay \
    --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data \
    --subject sub-01 \
    --session ses-01 \
    --split test \
    --disable-high-level \
    --calibration-block \
    --calibration-split training \
    --calibration-max-conditions 40 \
    --calibration-repetitions 3 \
    --calibration-sleep-seconds 0.5 \
    --max-trials 30 \python /home/psycontrol/01_Marco_ssd/01_maryam_realtime/scripts/run_realtime_gui.py \
    --mode things-replay \
    --data-root /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data \
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
    --sleep-seconds 1.0 \
    --host 127.0.0.1 \
    --port 8010 /home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/data --subject sub-01 --session ses-01 --split test --max-trials 10`

This raw demo path:
- reads `raw_eeg_<split>.npy`
- extracts true image onsets from the `stim` channel
- applies epoching, baseline correction, channel selection, and resampling to 250 Hz
- maps event codes to image files using the sorted THINGS image folders
- runs low-level reconstruction immediately and optional high-level refinement in the background

Notes:
- The low-level checkpoint is large (`~547 MB`) and is copied into this project.
- The slow path uses Hugging Face model loading for SDXL Turbo and IP-Adapter, so the machine must have the required cached models or internet access when first run.
- The copied V1 models expect 1000 samples at 1000 Hz per trigger-locked epoch, so `pre-event-ms + post-event-ms` must equal 1000.
- The browser monitor is local-only and polls JSON endpoints from a FastAPI app.
