# Project Improvements

Prioritized list of engineering and correctness issues to address in the
realtime EEG reconstruction project. The two missing Starstim31 checkpoints
(native low-level EEG→VAE latent, and diffusion prior / reconstruction) are
tracked separately in `docs/lab_realtime_pipeline_status.md` and are
intentionally excluded here.

## Current status

- Done: `lab_data/` and `data/derived/` are ignored to reduce the chance of
  committing participant data or derived artifacts by accident.
- Done: the README was rewritten as a local runbook and the THINGS GUI default
  data root now points at the sibling hierarchical project on this machine.
- Done: a focused pytest suite covers marker parsing, Starstim31/32 channel
  transforms, downsampling contracts, the ATMS embedder shape contract, and
  assessor sidecar / triplet summary outputs.
- Done: `pyproject.toml` provides editable-install package metadata with
  `dev` and `full` optional dependency groups.
- Done: montage constants are centralized and the GUI FastAPI app uses lifespan
  startup/shutdown handling.
- Remaining: live preprocessing still needs training-time whitening parity once
  the native checkpoints are available; the assessor pipeline still depends on
  the timm CLIP ViT-L/14 backbone cache for fully offline inference; vendoring
  Plotly remains low priority if the rig must run fully air-gapped.

## 1. Automated tests

A starter pytest suite now exists under `tests/`. For a realtime scientific
pipeline, numeric correctness (channel ordering, resampling, whitening, epoch
alignment, marker parsing) is exactly what silently breaks, so coverage should
continue to expand as the runtime matures.

Covered now:

- `adapt_starstim31_to_things63` and the montage constants
- the `resample.py` helpers (`prepare_realtime_window_torch`,
  `prepare_starstim32_native_window_torch`, `starstim31_to_things63_torch`, etc.)
- `parse_marker_payload`
- `Starstim31ATMSEmbedder` shape contract (`(B,31,250)` → `(1,1024)`)
- assessor JSON sidecars and the triplet summary figure/JSON contract

Still worth adding:

- epoch-sample math in `triggered_runner` (`epoch_samples == 1000` guard)
- a smoke test for low-level metadata linking assessor JSON sidecars
- a high-level worker test that confirms assessor sidecars are written after
  refined-image save
- eventual end-to-end tests with tiny fixture images and mocked models

## 2. Local large data and participant data

`lab_data/` and `data/derived/` are now in `.gitignore`. This prevents a broad
`git add .` from capturing raw participant EEG or derived artifacts.

`lab_data/` also contains named participant recordings (Maryam, Melina, Zar,
P04–P07): raw `.nedf`/`.easy`/`.xdf` EEG plus impedance logs. This should never
be one command away from a commit.

**Remaining action:**
- Consider DVC or git-lfs for derived artifacts that should be versioned.
- Review whether raw participant data should live inside the repo tree at all.

## 3. README and local defaults

The README has been rewritten as a concise local runbook and the corrupted
`run_things_raw_demo.py` / `run_realtime_gui.py` block has been removed. The
default `--data-root` in `run_realtime_gui.py` now points at
`../Hierarchical_EEG2Image_Reconstruction/data`.

**Remaining action:** keep the README in sync as the Starstim31 native
reconstruction checkpoints and high-level diffusion path are completed.

## 4. Packaging

`pyproject.toml` now exists, so the project can be installed in editable mode:

```bash
pip install -e ".[dev]"
```

The `full` extra carries optional dataset/export/training-style dependencies
that are not required for the standard realtime GUI path.

**Remaining action:** decide whether to keep `requirements.txt` and
`requirements.base.txt` as legacy install files or replace them with
`pyproject.toml` as the canonical dependency source.

## 5. Live path preprocessing does not match training

`triggered_runner._run_inference` builds a fresh generic `PreprocessingPipeline`
(z-score) per event, while the models were trained on MVNN-whitened data. This
is documented in `docs/differences_offline_online.md` and is the change most
likely to make live reconstructions look worse than offline.

Creating the pipeline fresh per event also means normalization has no
warmup/history; if per-epoch stats are intentional, make that explicit.

**Action:** port the training-time whitening into the live path (prioritize
once the native checkpoints land); document the per-epoch normalization choice.

## 6. Assessor self-containment and final summary artifact

The VA and six-dimension assessor head bundles are present in
`checkpoints/assessor/`, and assessor JSON sidecars are integrated into the
low-level and high-level output paths. The summary figure writer is implemented
and tested for target / low-level / high-level image triplets.

**Remaining action:**

- Cache or vendor the `vit_large_patch14_clip_224.openai` timm backbone needed
  by the assessor bundles for fully offline inference.
- Wire `write_triplet_summary(...)` into the end-to-end high-level completion
  path once every trial has a resolved target image, low-level reconstruction,
  and high-level reconstruction.
- Decide the final layout used for result export. The current tested figure is
  a three-column target / low-level / high-level panel with per-image assessor
  scores underneath.

## 7. Minor cleanups

Done:

- **Duplicated constants:** `THINGS_63_CHANNELS` / `STARSTIM_31/32` now live in
  `integration/montage.py` and are imported by replay/resampling code.
- **Deprecated FastAPI hooks:** `gui/server.py` now uses a lifespan handler.

Remaining:

- **CDN dependency:** the GUI loads Plotly from a CDN, which breaks on an
  air-gapped lab machine. Vendor it locally if the recording rig has no internet.

## Suggested order

1. Done: gitignore + participant-data safety guard (#2).
2. Done: README and default path cleanup (#3).
3. Done: starter test suite around numeric helpers and assessor contracts (#1).
4. Done: minimal package metadata (#4).
5. Next: cache/vendor the assessor CLIP backbone and wire the final triplet
   summary into the completed high-level path (#6).
6. Then: whitening/preprocessing fidelity and Plotly vendoring if needed (#5, #7).
