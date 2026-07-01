# Real-Time Integration: Hierarchical Pipeline + Streaming Infrastructure

## Overview

This document describes how to combine the hierarchical EEG-to-image reconstruction pipeline
with the real-time streaming infrastructure from
`/home/psycontrol/01_Marco_ssd/stimulation_pipeline/real_time_image_reconstruction`.

The goal is to run the full hierarchical reconstruction pipeline on live EEG data streamed
over LSL, rather than on pre-recorded `.pt` files.

---

## What Each Project Contributes

**Real-time project** provides the infrastructure:
- LSL streaming (`lsl_inlet.py`) with ring buffer and auto-reconnect
- Preprocessing pipeline: bandpass/notch filter, artifact rejection, online normalization
- `RealTimePipeline` executive loop: latency monitoring, GPU memory management, error recovery, callback system
- Mock streamer for testing without hardware (64ch @ 1000 Hz)

**Hierarchical project** provides the models:
- **ATMS** (iTransformer): EEG → 1024-D CLIP-aligned semantic embedding
- **Low-level encoder** (`encoder_low_level`): EEG → VAE latent `(4, 64, 64)`
- **Diffusion Prior** (UNet or FDN variant): EEG embedding → CLIP image embedding
- **`Generator4Embeds`** (SDXL img2img): prior output + low-level image + text → final image

---

## Interface Mismatches to Resolve

| Issue | Real-time project | Hierarchical project | Fix |
|---|---|---|---|
| Sample rate | 1000 Hz default | 250 Hz expected | Downsample 4× in preprocessor |
| Channel count | 64 channels (mock) | 63 channels | Drop 1 channel (e.g. last reference) |
| Window size | 1000 samples @ 1kHz | 250 samples @ 250Hz | Same 1-second window, different rate |
| Encoder output | single array (latent) | two outputs: embedding + VAE latent | Wrap both into one encoder output dict |
| Decoder input | single array | embedding + low-level image + text | Custom decoder class |
| Text prompt | not present | needed by `Generator4Embeds` | Use a fixed generic prompt or skip |
| Subject ID | not present | `torch.long` tensor needed by ATMS | Config parameter at pipeline creation |
| Retrieval DB | not present | `img_features_all` tensor | Load at startup, pass to encoder |

---

## Latency Problem

The real-time pipeline targets **<30 ms per frame** (20 Hz). The hierarchical decoder is much slower:
- Diffusion Prior (50 DDIM steps): ~200–800 ms
- SDXL img2img (10 steps): ~1–5 s

These cannot share the same 50 ms loop. The solution is to **split the pipeline into two
asynchronous streams**:

```
LSL → Preprocess → ATMS + Low-level encoder  →  queue  →  (async) → Prior + SDXL → display
         fast loop ~20 Hz                                    slow loop ~0.1–0.5 Hz
```

The `add_callback()` method already exists in `RealTimePipeline` for exactly this purpose —
the callback posts encoder outputs to a queue that the decoder thread consumes independently.

---

## Concrete Integration Plan

### 1. Add downsampling to the preprocessor

Create `src/preprocessing/resample.py` in the real-time project with a mean-decimation step
(1000 → 250 Hz, factor 4). Slot it into `PreprocessingPipeline.process()` after filtering,
before normalization.

```python
def mean_downsample(data: np.ndarray, factor: int) -> np.ndarray:
    """Downsample (n_channels, n_samples) by averaging groups of `factor` samples."""
    n_channels, n_samples = data.shape
    trimmed = n_samples - (n_samples % factor)
    return data[:, :trimmed].reshape(n_channels, -1, factor).mean(axis=2)
```

### 2. Create `HierarchicalEEGEncoder`

New file: `src/models/hierarchical_encoder.py` in the real-time project (or
`atms_pipeline/rt_encoder.py` in this project).

```python
class HierarchicalEEGEncoder(nn.Module):
    """
    Input:  (batch, 64, 1000)  — raw at 1000 Hz after preprocessing
    Output: dict {
        'eeg_embedding':    (batch, 1024),        # ATMS semantic embedding
        'low_level_latent': (batch, 4, 64, 64)    # low-level encoder VAE latent
    }
    """
    def __init__(self, atms_ckpt, low_level_ckpt, subject_id, img_features_all=None, device='cuda'):
        super().__init__()
        # load ATMS from atms_pipeline/atms_utils.py
        self.atms = ATMS(num_subjects=1).to(device)
        self.atms.load_state_dict(torch.load(atms_ckpt, map_location=device))
        self.atms.eval()

        # load encoder_low_level from atms_pipeline/atms_utils.py
        self.low_level_enc = encoder_low_level().to(device)
        self.low_level_enc.load_state_dict(torch.load(low_level_ckpt, map_location=device))
        self.low_level_enc.eval()

        self.subject_id = subject_id
        self.img_features_all = img_features_all  # optional retrieval database

    @torch.inference_mode()
    def forward(self, x):   # x: (batch, 64, 1000)
        B = x.shape[0]
        x63  = x[:, :63, :]                          # drop channel 64 → (B, 63, 1000)
        x250 = F.avg_pool1d(x63, kernel_size=4)      # downsample 1000→250 → (B, 63, 250)

        subject_ids = torch.full((B,), self.subject_id, dtype=torch.long, device=x.device)
        eeg_emb  = self.atms(x250, subject_ids).squeeze(1)   # (B, 1024)
        low_lat  = self.low_level_enc(x250)                  # (B, 4, 64, 64)

        return {'eeg_embedding': eeg_emb, 'low_level_latent': low_lat}
```

### 3. Create `HierarchicalDecoder`

New file: `src/models/hierarchical_decoder.py` in the real-time project (or
`atms_pipeline/rt_decoder.py` in this project).

```python
class HierarchicalDecoder:
    """
    Input:  dict from HierarchicalEEGEncoder
    Output: PIL Image

    Intended to run in its own thread via a queue, NOT in the 50 ms encoder loop.
    """
    def __init__(self, prior_ckpt, vae, sdxl_pipe, use_fdn=True, device='cuda',
                 num_prior_steps=50, num_sdxl_steps=10, guidance_scale=5.0,
                 img2img_strength=0.8, text_prompt="a photo of an object"):
        if use_fdn:
            self.prior = DiffusionPriorUNet_FDN(cond_dim=1024, dropout=0.1).to(device)
        else:
            self.prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
        self.prior.load_state_dict(torch.load(prior_ckpt, map_location=device))
        self.prior.eval()

        self.prior_pipe = Pipe(self.prior, device=device)
        self.vae = vae
        self.generator = Generator4Embeds(
            num_inference_steps=num_sdxl_steps,
            device=device,
            img2img_strength=img2img_strength,
        )
        self.num_prior_steps = num_prior_steps
        self.guidance_scale = guidance_scale
        self.text_prompt = text_prompt
        self.device = device

    @torch.inference_mode()
    def decode(self, encoder_output) -> "PIL.Image":
        eeg_emb  = encoder_output['eeg_embedding']   # (1, 1024)
        low_lat  = encoder_output['low_level_latent']  # (1, 4, 64, 64)

        # Stage 1: diffusion prior → CLIP image embedding
        prior_output = self.prior_pipe.generate(
            c_embeds=eeg_emb,
            num_inference_steps=self.num_prior_steps,
            guidance_scale=self.guidance_scale,
        )

        # Stage 2: low-level latent → PIL image via VAE
        x_rec = self.vae.decode(low_lat).sample
        x_rec = (x_rec / 2 + 0.5).clamp(0, 1)
        low_level_image = to_pil_image(x_rec[0].cpu()).resize((512, 512))

        # Stage 3: SDXL img2img conditioned on prior + low-level image
        image = self.generator.generate(
            prior_output,
            text_prompt=self.text_prompt,
            low_level_image=low_level_image,
        )
        return image
```

### 4. Wire into `RealTimePipeline`

```python
import queue
import threading
from pathlib import Path
from diffusers import AutoencoderKL

# --- Load models ---
device = "cuda"

vae = AutoencoderKL.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    subfolder="vae", torch_dtype=torch.float32
).to(device).eval()

encoder = HierarchicalEEGEncoder(
    atms_ckpt="models/contrast/ATMS/sub-01/.../40.pth",
    low_level_ckpt="models/contrast_v2/encoder_low_level/sub-01/.../60.pth",
    subject_id=0,
    device=device,
)

decoder = HierarchicalDecoder(
    prior_ckpt="ckpt/sub-01_prior_final_fdn.pt",
    vae=vae,
    sdxl_pipe=None,   # Generator4Embeds loads SDXL internally
    use_fdn=True,
    device=device,
)

# --- Build pipeline (encoder only in the fast loop) ---
config = PipelineConfig(
    window_samples=1000,       # 1s @ 1000 Hz
    n_channels=64,
    sampling_rate=1000.0,
    inference_interval_ms=50.0,
    device=device,
)

pipeline = RealTimePipeline(encoder=encoder, decoder=None, config=config)

# --- Async decoder via callback + queue ---
decode_queue = queue.Queue(maxsize=2)

def on_encoder_output(enc_out, _image):
    if not decode_queue.full():
        decode_queue.put_nowait(enc_out)

pipeline.add_callback(on_encoder_output)

def decoder_thread_fn():
    while True:
        enc_out = decode_queue.get()
        image = decoder.decode(enc_out)
        image.save(f"outputs/rt_{time.time():.3f}.png")   # or display

threading.Thread(target=decoder_thread_fn, daemon=True).start()

# --- Start ---
pipeline.connect_lsl("YourAmplifierStream")   # or "MockEEG" for testing
pipeline.start(blocking=True)
```

---

## What Is Reused Unchanged

| Component | Location | Status |
|---|---|---|
| `LSLInletWrapper` + `RingBuffer` | `real_time/src/streaming/lsl_inlet.py` | Use as-is |
| `PreprocessingPipeline` | `real_time/src/preprocessing/normalize.py` | Use as-is (add downsample step) |
| `RealTimePipeline` + `LatencyMonitor` + `GPUMemoryManager` | `real_time/src/main_pipeline.py` | Use as-is |
| `mock_streamer.py` | `real_time/src/streaming/mock_streamer.py` | Use for testing |
| `ATMS` model + checkpoint | `atms_pipeline/atms_utils.py` | Drop in as encoder stage 1 |
| `encoder_low_level` + checkpoint | `atms_pipeline/atms_utils.py` | Drop in as encoder stage 2 |
| `DiffusionPriorUNet_FDN` + checkpoint | `atms_pipeline/diffusion_prior.py` | Drop in as decoder stage 1 |
| `Generator4Embeds` (SDXL) | `atms_pipeline/custom_pipeline_low_level.py` | Drop in as decoder stage 2 |

**New code required:** ~200–300 lines total across:
- `resample.py` — mean-decimation downsampler
- `hierarchical_encoder.py` — wraps ATMS + low-level encoder
- `hierarchical_decoder.py` — wraps diffusion prior + SDXL
- Async queue + callback glue in the startup script

---

## Notes

- **Subject ID**: must match the checkpoint the ATMS model was trained with (`sub-01` → id=0).
- **Text prompt**: `Generator4Embeds` accepts a text string for SDXL conditioning. Without a
  trigger-aligned label, use a generic prompt like `"a photo of an object"` or leave it empty.
  Quality will be lower than the offline pipeline where the ground-truth label is known.
- **Retrieval mode**: optionally load `ViT-H-14_features_test.pt` as `img_features_all` in the
  encoder to use the top-5 retrieved embedding instead of the raw ATMS prediction (ablation case 2/4/6/8).
- **Diffusion steps**: reduce `num_prior_steps` to 10–20 and `num_sdxl_steps` to 4–6 to
  increase decoder throughput at the cost of some image quality.
- **Channel mapping**: the mock streamer streams the full 64-channel 10-20 layout. Real
  amplifiers may use a different order — verify that channels 0–62 in the live stream correspond
  to the 63-channel layout the ATMS model was trained on.
