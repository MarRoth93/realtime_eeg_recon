# Project Analogy Guide

## 1. Project Overview

This project is a real-time industrial telemetry-to-scene reconstruction system. It receives continuous multi-channel machine telemetry from a streaming transport, converts event-aligned telemetry into synchronized model windows, runs a fast reconstruction path for immediate operator feedback, and optionally starts a slower high-detail refinement path in the background.

The system is split across two codebases:

- The real-time runtime contains stream acquisition, marker handling, rolling buffers, live and replay runners, browser monitoring, runtime adapters, checkpoint loading, and output writing.
- The hierarchical reconstruction stack contains the offline training and inference machinery: the ATMS state encoder, a low-level latent predictor, a diffusion prior, a generator wrapper, dataset builders, and training scripts.

The best mental model is a smart-factory inspection line. A monitored machine emits many telemetry channels. A task controller emits markers when a relevant machine state or visual target occurs. The runtime extracts a window around that marker, normalizes it into the same format used by offline training, and reconstructs a coarse diagnostic image immediately. A second path uses a higher-level state embedding and a generative refiner to produce a more detailed inspection scene after the real-time loop has moved on.

## 2. High-Level Architecture

```text
Live Telemetry Stream + Marker Stream
        |
        v
LSL Inlets and Timestamp Synchronizer
        |
        v
Ring Buffer + Event-Aligned Window Extraction
        |
        v
Live-Style Preprocessing / Replay Loader
        |
        v
Model-Ready Telemetry Tensor
        |
        +------------------------------+
        |                              |
        v                              v
Low-Level Latent Encoder         ATMS State Encoder
        |                              |
        v                              v
VAE Decoder Preview              Diffusion Prior
        |                              |
        v                              v
Immediate Diagnostic Image       1024-D Scene Embedding
        |                              |
        +--------------+---------------+
                       v
          Background Generator Worker
                       |
                       v
          Refined Scene Reconstruction
                       |
                       v
       Browser Monitor, Metadata, Ratings, Files
```

The real-time layer is built around two timing classes:

- `LSLInletWrapper` and `MarkerInletWrapper` collect continuous telemetry and irregular event markers in background threads.
- `TriggeredReconstructionRunner` waits for accepted markers, extracts the matching buffered segment by timestamp, and runs inference once per event.

The model layer is deliberately split:

- The fast path uses `LowLevelRealtimeEncoder`, `LowLevelStarstimNativeRealtimeEncoder`, or `LowLevelEpochEncoder` plus `LowLevelVAEDecoder`.
- The slow path uses `HighLevelRefiner`, which combines an ATMS embedding, a diffusion prior, and `Generator4Embeds`.
- `SemanticRefinementWorker` keeps the slow path out of the event-processing loop by queueing refinement jobs.

## 3. Repository Structure

### Real-Time Runtime

`src/maryam_rt/realtime/streaming/lsl_inlet.py`

Provides the continuous stream reader. It resolves streams, connects with reconnect handling, pulls chunks in a background thread, stores data as `(channels, samples)`, and can extract either the latest window or a timestamp-aligned segment. It also tracks stream-to-local timestamp correction through `TimestampSynchronizer`.

`src/maryam_rt/realtime/streaming/marker_inlet.py`

Provides the irregular marker reader. It resolves a marker stream, converts marker samples into `MarkerEvent(value, timestamp)` records, stores them in a bounded queue, and exposes `pop_events()` to the runner.

`src/maryam_rt/realtime/preprocessing/buffer.py`

Defines a fixed-size ring buffer with shape checks, overwrite behavior, and padded window reads. The streaming inlet has a similar internal buffer implementation for live acquisition.

`src/maryam_rt/realtime/preprocessing/normalize.py`, `filters.py`, and `artifact.py`

Provide generic runtime preprocessing: filter chains, artifact validation, online normalization, and a `PreprocessingPipeline` that returns `(processed, is_valid)`. The triggered live runner uses this generic path when no custom event preprocessor is supplied.

`src/maryam_rt/integration/starstim_preprocessing.py`

Defines the live-style preprocessing contract used by native lab-live and raw replay. It selects the 31-channel machine layout, applies average reference, resamples to 250 Hz, baseline-corrects the pre-event interval, and crops the model-facing `31 x 250` window.

`src/maryam_rt/integration/resample.py`

Contains shape adapters for compatibility modes: `64 x 1000` to `63 x 250`, native `32 x 1000` to `31/32 x 250`, and `31 x 250` into a zero-filled `63 x 250` compatibility layout.

`src/maryam_rt/integration/low_level.py`

Loads the low-level encoder checkpoint, prepares incoming tensors for the expected channel and sample layout, records the last model-ready `x250` tensor, and decodes low-level latents through the SDXL VAE into an immediate image.

`src/maryam_rt/integration/high_level.py`

Loads the ATMS encoder or Starstim31 ATMS embedder, loads a diffusion prior checkpoint, wraps it in `Pipe`, initializes the SDXL/IP-Adapter generator, and exposes `refine(x250, low_level_image, text_prompt)`.

`src/maryam_rt/integration/worker.py`

Runs slow refinement in a background queue. It stores low-level image, model-ready tensor, optional prompt, and target paths in a `RefinementJob`, then writes refined images or error files without blocking marker processing.

`src/maryam_rt/integration/triggered_runner.py`

Coordinates live acquisition. It starts the telemetry and marker inlets, accepts configured marker labels, waits until the post-event window has arrived, extracts the timestamped segment, preprocesses it, runs low-level reconstruction, writes metadata, updates the monitor, and submits a high-level job if enabled.

`src/maryam_rt/integration/lab_replay.py`

Implements replay from raw `.easy` files or cached model-ready arrays. Raw replay reuses `StarstimLivePreprocessor`, so replay and lab-live use the same `31 x 250` contract. It writes low-level images, metadata, copied targets, optional ATMS embeddings, and optional assessments.

`scripts/run_realtime_gui.py`

Main browser-monitor entrypoint. It selects one of four modes: `live`, `lab-live`, `things-replay`, and `lab-replay`. It creates encoders, decoders, the optional refinement worker, monitor state, output folders, and the FastAPI application.

### Hierarchical Reconstruction Stack

`atms_pipeline/atms_utils.py`

Defines core model components used by both offline and runtime code: transformer configuration, inverted transformer encoder, patch embedding, projection heads, view dropout, the ATMS encoder, and `encoder_low_level`.

`atms_pipeline/ATMS.py`

Provides the newer merged ATMS implementation. It combines an iTransformer backbone, optional source-conditioning dropout, optional pretrained backbone loading, view dropout, a patch encoder, and a projection head into a 1024-dimensional state embedding.

`atms_pipeline/diffusion_prior.py`

Defines the diffusion prior variants and `Pipe`. `DiffusionPriorUNet_FDN` uses a U-shaped MLP with timestep embeddings, skip connections, and feature denormalization from the ATMS condition. `Pipe.generate()` starts from random 1024-dimensional noise and denoises it under the ATMS condition.

`atms_pipeline/custom_pipeline_low_level.py`

Adds custom generator behavior around SDXL Turbo and IP-Adapter. `Generator4Embeds` loads the generator, loads the IP-Adapter weights, accepts a 1024-dimensional image embedding, and can use a low-level image as img2img guidance.

`atms_pipeline/atms_dataset.py`

Loads the original offline model-ready telemetry arrays and image/text feature caches. It slices time windows, creates labels, and provides matching visual embedding features for contrastive ATMS training and retrieval-style evaluation.

`atms_pipeline/lab_finetune_dataset.py`

Loads manifest-aligned `31 x 250` live-style telemetry epochs and optional image/text feature caches for lab-native fine-tuning.

`scripts/build_lab_live_preproc_cache.py`

Builds the offline cache that matches the live preprocessing contract. It reads raw `.easy` files and a manifest, extracts event-aligned windows, drops one configured channel, average-references, resamples, baseline-corrects, crops to `31 x 250`, and writes an aligned NPZ plus manifest.

`scripts/train_lab_low_level_vae.py`

Trains a 31-channel low-level encoder against SDXL VAE latent targets. The output checkpoint is the native fast-path checkpoint used by `LowLevelEpochEncoder`.

`scripts/train_lab_diffusion_prior.py`

Freezes the ATMS encoder, computes ATMS condition embeddings from `31 x 250` windows, and trains an FDN diffusion prior to predict image-feature noise. The output checkpoint is loaded by `HighLevelRefiner`.

`dvc.yaml`

Documents the older offline inference graph: ATMS feature extraction, low-level latent/image inference, final reconstruction, and scoring.

## 4. Data Flow

In live mode, data enters through two streams:

1. A continuous multi-channel telemetry stream.
2. A marker stream carrying event labels and optional target metadata.

The telemetry inlet stores samples in a ring buffer shaped `(channels, samples)`. The marker inlet stores marker events with transport timestamps. When a marker is accepted, the runner waits until the telemetry buffer contains the requested post-event interval. It then calls `get_segment_by_lsl_times(start_lsl, end_lsl)`.

Native lab-live uses this model-facing route:

```text
Raw event window:       32 x 600 at 500 Hz for -0.2..1.0 s
Channel selection:      31 x 600
Average reference:      31 x 600
Resample:               31 x 300 at 250 Hz
Baseline correction:    31 x 300
Crop 0..1 s:            31 x 250
Batch dimension:         1 x 31 x 250
```

Compatibility live mode uses:

```text
Raw event window:       64 x 1000 at 1000 Hz
Drop final channel:     63 x 1000
Mean downsample 4x:     63 x 250
Batch dimension:         1 x 63 x 250
```

Raw replay follows the same native lab-live preprocessing. Derived replay skips raw extraction and reads model-ready `31 x 250` arrays from an NPZ file.

The low-level encoder maps the model-ready tensor to a latent image tensor. In the current low-level implementation, each channel is projected from 250 samples to 128 features, flattened into a channel-feature vector, reshaped to `dim x 1 x 1`, and expanded through transposed convolutions into a `4 x 64 x 64` latent. The VAE decoder turns that latent into an RGB preview.

The high-level path reuses the same `x250` tensor. ATMS maps it to a 1024-dimensional operational state vector. The diffusion prior converts that condition into a 1024-dimensional image embedding. `Generator4Embeds` combines the embedding, a text prompt, and the low-level preview image to create the refined output.

Outputs are written under `outputs/<mode>/` with separate folders for low-level images, high-level images, metadata, targets, embeddings, and optional ratings. The browser monitor reads monitor state and image paths through FastAPI endpoints.

## 5. Real-Time Pipeline

The live runner is event-triggered rather than continuous-frame decoding. This matters for latency: the ring buffer is continuous, but inference starts when a marker arrives and the requested telemetry window is complete.

The runtime sequence is:

1. Start telemetry and marker inlet threads.
2. Wait for both streams to connect.
3. Poll markers at the configured interval.
4. Parse marker payloads and filter by accepted event names.
5. Enforce optional marker cooldown.
6. Wait until the buffered telemetry reaches the event end timestamp.
7. Extract the timestamp-aligned window.
8. Apply either generic preprocessing or the native live-style epoch preprocessor.
9. Run low-level encoder and VAE decoder immediately.
10. Write image and metadata.
11. Update the browser monitor.
12. Submit a slow refinement job if high-level mode is enabled.

The stream inlet handles several runtime hazards:

- Missing streams cause repeated resolve attempts.
- Idle streams are disconnected and re-resolved.
- Unexpected sample orientation is detected and corrected when possible.
- Unexpected channel counts raise shape errors.
- Timestamped window requests fail if they ask for data beyond buffer capacity, insufficient buffered samples, or a future end time.

The older `RealTimePipeline` executive loop remains available for fixed-interval inference. It includes a latency monitor, GPU memory manager, watchdog warning, error recovery, automatic LSL reconnect, and callback mechanism. The current GUI path uses `TriggeredReconstructionRunner` because the system is marker-aligned.

## 6. Hierarchical Model Pipeline

The hierarchical stack has two complementary branches.

The fast branch is a direct latent predictor:

```text
Model-Ready Telemetry Tensor
        |
        v
Per-Channel Temporal Projection
        |
        v
Flattened Channel-State Vector
        |
        v
Transposed-Convolution Upsampler
        |
        v
4 x 64 x 64 VAE Latent
        |
        v
Immediate Preview Image
```

This branch is reusable in real time because it is a single forward pass plus VAE decoding.

The high-level branch is a state-to-scene branch:

```text
Model-Ready Telemetry Tensor
        |
        v
iTransformer Encoder + Source Conditioning
        |
        v
Patch Embedding + Projection Head
        |
        v
1024-D Operational State Vector
        |
        v
Diffusion Prior
        |
        v
1024-D Scene Embedding
        |
        v
SDXL/IP-Adapter Generator with Low-Level Image Guidance
        |
        v
Refined Scene Reconstruction
```

ATMS is reusable for runtime inference when the input channel count, sequence length, source-conditioning index, and checkpoint architecture match. Its training-only features include view dropout, source-conditioning dropout, contrastive loss, and optional pretrained backbone loading.

The diffusion prior is reusable at runtime through `Pipe.generate()`, but it is too slow for the marker loop if run synchronously. That is why `HighLevelRefiner` is called only inside `SemanticRefinementWorker`.

## 7. Encoding and Decoding Logic

### Low-Level Encoding

Expected input is a batched tensor shaped `(B, C, 250)`, where `C` is usually `31` for native lab artifacts or `63` for legacy compatibility. The low-level encoder applies a source-wise temporal linear layer to each channel, producing `(B, C, 128)`. It reshapes this to `(B, C * 128, 1, 1)` and uses a transposed-convolution upsampler to produce `(B, 4, 64, 64)`.

The low-level decoder is the SDXL VAE decoder. It expects the `4 x 64 x 64` latent scale learned during training and returns an RGB image. If the offline target latents and runtime VAE scaling diverge, the preview can become washed out or unstable.

### ATMS State Encoding

ATMS expects `(B, C, 250)` plus a source-conditioning index tensor shaped `(B,)`. The DataEmbedding layer can include source embedding tokens. The transformer encoder operates over the embedded channel/time representation. The output is sliced back to the original channel count, passed through a patch embedding module, flattened, and projected to 1024 dimensions.

The resulting vector is the high-level condition for the diffusion prior. It is not an image; it is a compact operational state vector aligned to a visual embedding space during offline training.

### Diffusion Prior Decoding

The prior starts with random 1024-dimensional noise and iteratively denoises it. At each step, timestep embeddings and optional ATMS conditioning are injected into a U-shaped MLP. In the FDN variant, feature denormalization layers use the ATMS vector to produce condition-specific scale and shift values.

Classifier-free guidance is implemented in `Pipe.generate()` by evaluating both conditioned and unconditioned prior predictions and combining them with a guidance scale.

### Final Generator

`Generator4Embeds` loads SDXL Turbo and an IP-Adapter. It supplies the prior output as `ip_adapter_embeds`, optionally supplies a text prompt, and optionally supplies the low-level image as img2img guidance. The refined output is a PIL image.

## 8. ATMS / Core System Mechanics

ATMS is the high-level state encoder. Its job is to translate a fixed-length multi-channel telemetry window into a compact 1024-dimensional state vector that the diffusion prior can use.

Inputs:

- Model-ready tensor: `(B, 31, 250)` for native lab mode or `(B, 63, 250)` for legacy mode.
- Source-conditioning index: integer tensor `(B,)`.

Outputs:

- State embedding: usually `(B, 1024)`.

Internal stages:

- View dropout can zero individual channels, channel groups, or time spans during training.
- Data embedding converts the input into a transformer-ready representation and can inject source-conditioning tokens.
- The iTransformer encoder applies multi-head attention and feed-forward layers.
- Patch embedding applies temporal convolution, pooling, full-channel convolution, and flattening.
- Projection maps the flattened representation into the 1024-dimensional visual embedding space.

Runtime constraints:

- Checkpoint architecture must match channel count, sequence length, transformer layer count, source table size, and projection input dimension.
- The native `Starstim31ATMSEmbedder` infers these values from checkpoint keys before instantiating the runtime model.
- For lab-native high-level mode, the same source-conditioning index used in the training manifest must be supplied at runtime.

Extension options:

- Add a clean interface that returns both low-level latent and ATMS state vector from one shared adapter.
- Add checkpoint metadata files so channel count, sequence length, and source table size do not need to be inferred from weight shapes.
- Add explicit runtime validation that checks input shape before model execution and reports which contract was violated.

## 9. Offline Training Versus Real-Time Inference

Offline training creates the artifacts. Real-time inference consumes them.

Offline development includes:

- Building live-style cache files with `scripts/build_lab_live_preproc_cache.py`.
- Training low-level latent predictors with `scripts/train_lab_low_level_vae.py`.
- Training the diffusion prior with `scripts/train_lab_diffusion_prior.py`.
- Running older offline stages through `dvc.yaml`.
- Creating or loading visual feature caches used as training targets.

Runtime inference includes:

- Running the GUI through `scripts/run_realtime_gui.py`.
- Loading low-level checkpoints into `LowLevelEpochEncoder` or compatibility encoders.
- Loading ATMS and prior checkpoints into `HighLevelRefiner`.
- Loading SDXL VAE, SDXL Turbo, and IP-Adapter weights through diffusers.
- Writing images and metadata to mode-specific output folders.

Preprocessing must stay identical wherever a checkpoint is shared. For native lab artifacts, both raw replay and lab-live must produce `31 x 250` tensors with the same channel order, average-reference step, resampling behavior, baseline interval, and `0..1 s` crop. If offline training uses a cache built by `build_lab_live_preproc_cache.py`, then live inference should use the same `StarstimLivePreprocessor` contract.

If offline and real-time preprocessing diverge, the failures are usually silent at first: the tensor has the right shape but the scale, alignment, or channel semantics are wrong. The low-level VAE latent can decode to low-detail or near-flat images, and the ATMS vector can condition the prior toward the wrong region of embedding space.

## 10. Integration Between The Two Projects

```text
Real-Time Runtime
    acquisition:       LSL telemetry + markers
    buffering:         ring buffer with timestamp lookup
    preprocessing:     live-style window builder
    adapter:           low-level encoder + ATMS embedder
            |
            v
Hierarchical Stack
    low-level branch:  encoder_low_level -> SDXL VAE decode
    state branch:      ATMS -> diffusion prior -> scene embedding
    generator branch:  SDXL/IP-Adapter refinement
            |
            v
Real-Time Output Layer
    low image, refined image, metadata, monitor state, optional ratings
```

Reusable hierarchical components:

- `encoder_low_level` for direct latent prediction.
- `ATMS` or the runtime `Starstim31ATMS` embedder for 1024-dimensional state embeddings.
- `DiffusionPriorUNet_FDN` or `DiffusionPriorUNet`.
- `Pipe.generate()` for prior sampling.
- `Generator4Embeds` for final scene generation.

Adapter requirements:

- Convert stream windows into `(B, C, 250)` float tensors on the same device as the checkpoint.
- Preserve channel order exactly.
- Use the same source-conditioning index conventions as training.
- Keep dtype as `float32` for model inputs unless a wrapper explicitly casts.
- Keep the slow diffusion and generator path out of the marker loop.
- Save the model-ready `x250` tensor from the low-level wrapper so the background refiner can reuse the same exact window.

Current integration is partly direct and partly duplicated:

- The real-time repo carries a copied subset under `src/maryam_rt/hierarchical`.
- Native lab ATMS has a dedicated runtime implementation in `starstim31_atms.py`.
- The high-level refiner imports copied ATMS, prior, and generator classes from the runtime package.
- Offline training remains in the sibling hierarchical repo.

A cleaner integration would package the hierarchical runtime components as a versioned library and require checkpoint metadata for every trained artifact.

## 11. Practical Developer Guide

Start with these files:

- `scripts/run_realtime_gui.py` for mode selection and runtime wiring.
- `src/maryam_rt/integration/triggered_runner.py` for live event processing.
- `src/maryam_rt/integration/lab_replay.py` for replay behavior and output metadata.
- `src/maryam_rt/integration/starstim_preprocessing.py` for native live preprocessing.
- `src/maryam_rt/integration/low_level.py` for fast-path checkpoint loading and VAE decoding.
- `src/maryam_rt/integration/high_level.py` for ATMS, prior, and generator loading.
- `atms_pipeline/atms_utils.py`, `atms_pipeline/ATMS.py`, and `atms_pipeline/diffusion_prior.py` in the hierarchical repo for model internals.
- `scripts/build_lab_live_preproc_cache.py`, `scripts/train_lab_low_level_vae.py`, and `scripts/train_lab_diffusion_prior.py` in the hierarchical repo for offline artifact creation.

Install the runtime package from the real-time repo:

```bash
pip install -e ".[dev]"
```

Run low-level raw replay through the browser monitor:

```bash
python scripts/run_realtime_gui.py \
  --mode lab-replay \
  --lab-replay-source raw \
  --disable-high-level \
  --host 127.0.0.1 \
  --port 8010
```

Run replay with high-level refinement only after providing the native low-level and prior checkpoints required by the CLI.

To debug shape issues:

- Check metadata JSON under `outputs/<mode>/events` or `outputs/<mode>/metadata`.
- Confirm `processed_shape` or `epoch_model_shape`.
- Confirm native lab paths produce `31 x 250`.
- Confirm compatibility paths produce `63 x 250`.
- Check whether high-level jobs wrote `*_error.txt`.

To test without live hardware:

- Use `lab-replay` with `--lab-replay-source raw` to exercise the raw-to-model preprocessing path.
- Use `lab-replay` with `--lab-replay-source derived` to test only checkpoint loading and inference on cached model-ready arrays.
- Use `things-replay` for the older 63-channel compatibility path.
- Use the mock stream scripts only when testing stream connectivity and marker flow rather than final artifact quality.

To verify integration:

- The event metadata should contain the expected window shape.
- A low-level image should be written for each accepted marker.
- When high-level mode is enabled, the background queue should write a refined image or an error file for each submitted job.
- Browser state should show latest target, latest low-level image, recent marker status, and optional refined image.
- Latency-sensitive marker processing should continue while high-level refinement is running.

## 12. Extension Points And Improvement Ideas

- Add a single model adapter that returns both low-level latent and ATMS state vector, avoiding repeated preprocessing and making output contracts explicit.
- Store checkpoint metadata next to every model artifact: channel count, sample count, source table size, transformer layer count, preprocessing recipe, feature target type, and expected dtype.
- Move shared preprocessing into one package used by both offline cache builders and live runtime.
- Add numeric equivalence tests between raw replay preprocessing and the offline cache builder for the same manifest row.
- Add stricter runtime validation for channel order, sample count, dtype, and device before model execution.
- Replace implicit source-conditioning integers with a manifest-backed resolver that fails loudly on missing mappings.
- Add bounded high-level queue policies such as drop-oldest or keep-latest to avoid stale refinements during dense event streams.
- Add per-stage timing for low-level encoding, VAE decode, ATMS embed, prior sampling, and generator refinement.
- Add replay reports that summarize accepted events, rejected events, error files, and output counts.
- Add a modular decoder interface so the fast VAE decoder and slow generator can be swapped independently.
- Add CPU-friendly smoke tests that instantiate tiny models or mocked models to verify data flow without downloading large generator assets.
- Add a model hot-swap layer that can load a new low-level checkpoint after validating metadata against the active stream contract.

## 13. Mapping Table

| Analogy term | Technical meaning |
| --- | --- |
| Multi-sensor telemetry stream | Continuous multi-channel time-series input |
| Marker stream | Irregular event stream that defines when to extract a window |
| Rolling telemetry buffer | Fixed-capacity ring buffer storing recent samples |
| Event-aligned telemetry window | Timestamped segment around a marker |
| Live-style preprocessing | Channel selection, reference transform, resampling, baseline correction, and crop |
| Model-ready telemetry tensor | Batched `(B, C, 250)` float tensor |
| Compatibility layout | Zero-filled or channel-adapted tensor used for older checkpoints |
| Low-level latent encoder | Model that maps telemetry directly into VAE latent space |
| Immediate diagnostic image | Fast VAE-decoded preview |
| ATMS state encoder | Transformer-based encoder producing a 1024-dimensional condition vector |
| Operational state vector | Learned intermediate representation used by the prior |
| Diffusion prior | Model that converts state vectors into scene embedding samples |
| Scene embedding | 1024-dimensional generator conditioning vector |
| Generator worker | Background queue that runs slow refinement |
| Refined scene reconstruction | Final generated image |
| Control dashboard | Browser monitor and API endpoints |
| Source-conditioning index | Integer conditioning code for acquisition/source identity |
| Offline cache builder | Script that produces model-ready arrays from raw recordings |
| Checkpoint stack | Low-level encoder, ATMS encoder, prior, VAE, and generator assets |
