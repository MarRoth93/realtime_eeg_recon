# Differences Between Offline And Online EEG Reconstruction

## Purpose

This note summarizes the current differences between:

- the original offline reconstruction flow in this repository
- the new online/live-oriented reconstruction flow in `/home/psycontrol/01_Marco_ssd/01_maryam_realtime`

It focuses on why the outputs, especially the low-level reconstructions, do not currently look identical.

## Short Summary

The offline and online pipelines are conceptually similar, but they are not yet numerically identical.

The biggest points are:

- the offline pipeline uses already saved, model-ready preprocessed EEG tensors
- the online pipeline starts from raw EEG and recomputes preprocessing on the fly
- the offline pipeline uses the exact preprocessing recipe used for training
- the online pipeline is currently an approximation of that recipe
- the final good-looking offline results come from the full reconstruction stack, not only the low-level stage

## 1. Input Data Source

### Offline

The offline pipeline reads EEG from:

- `data/Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_test.npy`
- `data/Preprocessed_data_250Hz/sub-XX/preprocessed_eeg_training.npy`

These files are already:

- event-locked
- channel-selected and reordered
- baseline-corrected
- downsampled to `250 Hz`
- whitened with MVNN
- saved in the exact tensor format expected by the trained models

### Online / Live-style

The online pipeline starts from raw EEG:

- continuous raw EEG stream in the final live setting
- or raw THINGS `.npy` files in the replay/demo setting

This means it must rebuild the model-ready tensor from raw EEG every time.

## 2. Where Preprocessing Happens

### Offline

Preprocessing is done ahead of time by:

- [preprocessing.py](/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/preprocessing.py)
- [preprocessing_utils.py](/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/preprocessing_utils.py)

That code performs:

- event extraction from the `stim` channel
- channel selection and ordering to the 63-channel model layout
- epoching from `-0.2` to `1.0 s`
- baseline correction
- resampling to `250 Hz`
- cropping to the final `63 x 250` window
- MVNN whitening
- merging and reshaping across sessions

### Online / Live-style

Preprocessing is recomputed from raw EEG in:

- [`things_raw_demo.py`](</home/psycontrol/01_Marco_ssd/01_maryam_realtime/src/maryam_rt/integration/things_raw_demo.py>)

It now performs:

- event extraction from `stim` for replay/demo mode
- channel selection and ordering
- epoching
- baseline correction
- resampling to `250 Hz`
- cropping to the final `63 x 250` window
- whitening

This is much closer than the first version, but still not guaranteed to be byte-for-byte identical to the saved offline tensors.

## 3. Whitening / MVNN

This is the most important difference.

### Offline

The offline pipeline computes MVNN whitening using the original preprocessing code. This produces EEG tensors roughly on the scale expected by the low-level encoder.

### Online / Live-style

The first online raw-demo version was missing whitening. As a result:

- raw epoched EEG had values around `1e-5`
- the low-level encoder effectively saw near-zero input
- the decoded low-level images became flat gray

This was fixed by adding whitening in the online path. After that fix:

- the scale became much closer to the offline tensors
- the outputs stopped being numerically flat

However, the whitening used online is still a recomputed version, not the exact original saved tensor from disk.

## 4. Session Handling

### Offline

The preprocessed offline files are merged across sessions.

That means the model sees EEG in the same combined format used when the preprocessed dataset was built.

### Online / Live-style

The raw replay tests were run from one raw session at a time, for example:

- `sub-01/ses-01`

So even if the per-trial preprocessing is similar, the source data pool is narrower than the offline merged tensor.

## 5. Averaging

This is a smaller issue than whitening.

### Offline low-level DVC path

The low-level DVC path uses:

- [`atms_pipeline/atms_vae_inference.py`](/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/atms_pipeline/atms_vae_inference.py)
- [`atms_training/eegdatasets_leaveone_latent_vae_no_average.py`](/home/psycontrol/01_Marco_ssd/Hierarchical_EEG2Image_Reconstruction/atms_training/eegdatasets_leaveone_latent_vae_no_average.py)

That path is explicitly `no_average`.

So the original offline low-level stage is not mainly better because it averages many repetitions.

### Online / Live-style

The raw demo has an optional `--average-by-event` mode. That is only a replay/demo convenience. It is not required for the low-level model and it is not representative of true real-time operation.

In real-time live use, reconstruction must normally run from a single current epoch.

## 6. Low-Level Versus Full Reconstruction

Another important source of confusion:

### Offline final pipeline

The DVC pipeline is:

1. `atms_inference`
2. `atms_vae_inference`
3. `atms_reconstruction`

The final images in `generated_images` are not just low-level outputs. They also include:

- semantic EEG embeddings
- diffusion prior
- low-level image guidance
- text/image conditioning
- generative refinement

### Online low-level preview

The online low-level preview currently does only:

- EEG epoch -> low-level encoder -> VAE decode

So even if preprocessing matched perfectly, the online low-level image should still be expected to be rougher than the final offline generated image.

## 7. What Was Verified

The following checks were performed:

- decoding a stored true image latent with the current VAE decoder produces a normal image
- therefore the VAE decoder is not the main problem
- the missing-whitening bug in the first raw-demo version was real and was fixed
- after whitening, raw EEG scale became much closer to offline-preprocessed scale
- however, even the original low-level checkpoint on stored preprocessed EEG can still look blurry or near-gray in some cases

This suggests:

- the online preprocessing bug was real
- but the low-level checkpoint itself is also limited and weak

## 8. Why Offline And Online Tensors Still Differ

Even when they represent the same event, the offline-preprocessed tensor and the online-recomputed tensor can still differ because:

- the offline tensor is the final saved output of the original preprocessing pipeline
- the online tensor is a fresh recomputation from raw EEG
- small details matter a lot:
  - exact whitening implementation
  - exact covariance estimation
  - exact session logic
  - exact repetition grouping
  - exact resampling behavior
  - exact crop alignment

So the two tensors are based on the same idea, but they are not yet guaranteed to be identical.

## 9. Practical Interpretation

In simple terms:

- offline EEG tensor = lab-prepared, model-ready representation
- online EEG tensor = raw EEG transformed on the fly to imitate that representation

Right now the online version is much closer than before, but it is still an approximation.

## 10. Current Best Conclusion

The difference between offline and online low-level outputs is not explained by averaging alone.

The main causes are:

- offline uses the exact saved preprocessed tensors
- online recomputes preprocessing from raw EEG
- the first online version missed whitening entirely
- the low-level checkpoint itself is weak and can already produce blurry outputs even on offline-preprocessed EEG

## 11. What Would Be Needed To Match Offline More Closely

To make the online/raw path resemble the offline path as closely as possible:

- port the offline preprocessing logic as exactly as possible
- match MVNN whitening behavior exactly
- match session handling exactly
- verify that raw-recomputed tensors numerically align with saved offline tensors for the same event
- treat low-level output as a fast coarse preview, not the final quality result

## 12. Bottom Line

The online pipeline is not wrong in principle. The largest initial bug was the missing whitening step, and that has been corrected.

But the offline and online low-level paths are still not identical because:

- one uses saved model-ready tensors
- the other rebuilds those tensors from raw EEG in real time

And beyond preprocessing, the low-level model itself is only a coarse reconstruction stage.
