# Technical Capability Report: Realtime EEG-to-Image Reconstruction and Image Assessment

## Summary

This project is designed to run an end-to-end EEG-to-image reconstruction
pipeline for visual stimulus experiments. In the intended live setup, a
participant views images while EEG is recorded with a Neuroelectrics Starstim
system. At each stimulus onset, the system extracts a short EEG epoch, preprocesses
it into the model format, reconstructs an image from the EEG response, and shows
the result in a browser GUI. Reconstructed images can then be rated automatically
by assessor models for affective and perceptual dimensions.

The core technical capability is:

```text
visual stimulus -> live Starstim EEG -> trigger-aligned EEG epoch
-> low-level reconstruction -> delayed high-level reconstruction
-> assessor ratings -> GUI / saved result artifacts
```

The live reconstruction path is architecturally in place. The remaining practical
requirements are the correct native Starstim31 low-level reconstruction checkpoint,
a Starstim-compatible high-level reconstruction prior, and a verified local
runtime environment with the required model caches.

## Experimental Input

The target hardware setup is a Starstim EEG/tES system operated through NIC2.
For this project, the relevant Starstim properties are:

- wireless EEG acquisition,
- 32 EEG channels,
- nominal EEG sampling around 500 samples/second,
- Lab Streaming Layer (LSL) support through NIC2 for live data streaming.

The software does not directly communicate with the Starstim device at a low
hardware-driver level. Instead, it consumes the LSL stream that NIC2 exposes.
This is an important design choice: the realtime reconstruction pipeline only
needs a correctly named EEG LSL stream and a separate marker LSL stream.

The live system therefore expects two streams:

```text
EEG stream:     continuous Starstim EEG samples
Marker stream:  stimulus onset events from the experiment/task script
```

The marker stream should emit an event when the image appears. A simple marker
can be:

```text
stim_onset
```

The preferred marker is a JSON payload, because it also lets the GUI identify
and display the target image:

```json
{"event": "stim_onset", "image_id": "example_image_id"}
```

## Live Reconstruction Flow

During live operation, the GUI process listens for both LSL streams. The EEG
stream is continuously buffered. When a stimulus marker arrives at LSL time `T`,
the pipeline extracts the EEG segment:

```text
T - 200 ms  to  T + 1000 ms
```

This raw Starstim epoch is then converted into the same model-facing format as
the offline lab preprocessing:

```text
32-channel Starstim epoch
-> drop Fz
-> average reference
-> resample to 250 Hz
-> baseline-correct using -200..0 ms
-> crop 0..1 s after stimulus onset
-> 31 channels x 250 samples
```

The resulting `31 x 250` tensor is the standard Starstim lab EEG input format
used by the project.

## Reconstruction Stages

### Low-Level Reconstruction

The low-level reconstruction path is the fast path. It is intended to produce
an image shortly after the EEG epoch is available.

Conceptually:

```text
preprocessed EEG epoch, 31 x 250
-> low-level EEG encoder
-> VAE latent representation
-> SDXL VAE decoder
-> low-level reconstructed image
```

This image is expected to capture low-level visual structure more directly than
semantic detail. It is the reconstruction that should appear in the GUI first.

### High-Level Reconstruction

The high-level reconstruction path is slower and runs in the background. It is
intended to refine or reconstruct semantic image content from an EEG embedding.

Conceptually:

```text
preprocessed EEG epoch, 31 x 250
-> Starstim31 ATMS EEG encoder
-> EEG embedding
-> diffusion prior / semantic reconstruction model
-> high-level reconstructed image
```

The project already contains the Starstim31 ATMS embedding checkpoint and code
path. The missing part is the correct Starstim-compatible high-level diffusion
prior / reconstruction checkpoint. Until that exists, the EEG embedding can be
computed, but the final high-level reconstruction should not be treated as
complete.

## Offline Replay Capability

The offline replay mode uses pre-extracted lab epochs rather than live LSL
streams. This is useful for debugging, demonstrations, and reproducible testing
without the Starstim device connected.

The offline lab replay file contains model-ready epochs:

```text
N trials x 31 channels x 250 samples
```

For subject `P07`, the replay mode can iterate through these epochs, resolve
the target image from the manifest, run the reconstruction stack, and update the
same browser GUI used for live mode.

Offline replay and live mode are intended to converge on the same model-facing
input representation. This makes offline replay a useful approximation of the
live experiment pipeline.

## GUI Behavior

The GUI is a local browser monitor served by the Python process. It shows:

- EEG stream activity,
- recent stimulus markers,
- the current target image when available,
- the latest low-level reconstruction,
- high-level reconstruction status when enabled.

The expected timing behavior is:

```text
stimulus onset
-> wait until enough post-stimulus EEG has arrived
-> run low-level reconstruction
-> display low-level image
-> optionally run high-level reconstruction in background
-> update GUI when high-level image is available
```

Thus, low-level reconstruction is the immediate output. High-level
reconstruction is expected to lag behind because diffusion-style generation is
more computationally expensive.

## Assessor Ratings

The project also integrates image assessor models. These models rate images
after reconstruction. The intended use is to apply the same assessor pipeline to:

- the original target image,
- the low-level reconstruction,
- the high-level reconstruction.

This enables a compact comparison between what was shown and what the EEG-based
models reconstructed.

The assessor system currently supports:

- valence,
- arousal,
- approach,
- attention,
- control,
- dominance,
- out-of-distribution percentile estimates where enabled.

The output is written as JSON sidecar files next to the reconstruction outputs.
The project also includes a triplet summary figure concept: target image,
low-level reconstruction, and high-level reconstruction shown side by side with
their associated assessor scores.

This creates a useful evaluation layer. Instead of only asking whether the
reconstructed image visually resembles the target, the system can also ask
whether the reconstruction preserves measurable affective or semantic image
properties.

## Output Artifacts

For a complete run, the system is expected to save:

```text
outputs/<run_name>/
  low_level/        low-level reconstructed images
  high_level/       high-level reconstructed images
  targets/          copied target images when available
  events/           event and reconstruction metadata
  assessments/      assessor JSON sidecars
  embeddings/       optional Starstim31 ATMS EEG embeddings
```

These artifacts make it possible to inspect individual trials after the session,
debug timing or marker issues, and build summary figures for reports.

## Current Implementation Status

Already implemented:

- offline lab replay using derived Starstim31 `31 x 250` epochs,
- live LSL EEG and marker ingestion,
- native Starstim live preprocessing to `31 x 250`,
- browser GUI for live/replay monitoring,
- Starstim31 ATMS embedding path,
- assessor sidecar output for reconstructed and target images,
- tests for channel mapping, resampling, marker parsing, ATMS shape contract,
  and assessor output format.

Still required for full end-to-end capability:

- native Starstim31 low-level EEG-to-image checkpoint,
- Starstim-compatible high-level reconstruction prior,
- verified full SDXL/high-level model cache,
- cached or locally available assessor CLIP backbone weights,
- hardware validation with the actual NIC2 LSL stream names, sampling rate, and
  channel metadata.

## Practical Capability Statement

If the required checkpoints and model assets are available, this project can
operate as a realtime EEG reconstruction system for Starstim-based visual
experiments. It can receive live EEG through NIC2/LSL, align EEG epochs to
stimulus onset markers, preprocess the EEG into the trained model format,
produce low-level and high-level image reconstructions, display the current
trial in a browser GUI, and automatically rate the target and reconstructed
images along multiple affective dimensions.

In its current state, the architecture and data flow are in place. The most
important remaining dependency is not the live-streaming design, but the
availability of the correct Starstim-trained reconstruction checkpoints.
