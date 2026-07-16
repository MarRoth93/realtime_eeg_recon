# Aligning live lab preprocessing to the derived-data contract

The derived training epochs were produced by the custom offline pipeline
(`EEGPreprocessor` + `apply_mvnn` in `Hierarchical_EEG2Image_Reconstruction`),
which cleans the signal with **bandpass filtering, ICA eye-blink removal, MVNN
whitening, and peak-to-peak trial rejection**. The live path
(`StarstimLivePreprocessor`) previously did only the geometric steps (drop Fz,
average reference, resample, baseline, crop). This change closes that gap.

Because ICA cleaning and MVNN whitening are both linear, each is a static
`31 x 31` matrix applied to the model epoch. They are fit **once, offline**
(calibration), so the live path only does matrix multiplies and needs no MNE at
inference time.

## Pipeline order (live and raw replay)

```
select 31 ch (drop Fz) -> average reference -> [ICA operator]
  -> [bandpass 0.01-30 Hz] -> resample 250 Hz -> baseline -> crop 0..1 s
  -> peak-to-peak rejection check -> [MVNN whitening]
```

This mirrors the offline order (reference → ICA → filter → resample → epoch →
reject → MVNN). Rejection is measured on the pre-whitening epoch, matching where
the offline pipeline drops trials.

### Note on the 0.01 Hz highpass

The offline highpass runs on minutes of continuous data. A 0.01 Hz highpass is
not realizable on a ~1 s epoch, so it is skipped (below `_MIN_HIGHPASS_NORM`);
DC/drift is already handled by the average reference and baseline steps. The
30 Hz lowpass — the part that matters per epoch — is applied.

## 1. Calibrate the operators (offline, once per subject)

ICA (plan option a) is fit from a calibration recording — e.g. the resting-state
block, which is also where the offline pipeline learns ocular components:

```bash
python scripts/calibrate_lab_operators.py \
    --easy data/P04/<rec>.easy --info data/P04/<rec>.info \
    --start-sample 0 --n-samples 60000 \
    --ica-output checkpoints/lab_calibration/P04_ica_operator.npy
```

Pass `--exclude-indices` to bypass ICLabel (the `mne-icalabel` dependency, in the
`calibration` extra). Applying ICA to resting state is optional — omit
`--lab-ica-operator` at run time to skip it.

## 2. MVNN whitening from the first image block

The whitener is treated as per-subject train data: the **first block** of image
trials calibrates it, and the rest of the session reuses the same matrix, which
is cached (`outputs/lab_replay/cache/`) and reused on later runs. You can also
supply a precomputed matrix.

```bash
# fit-from-first-block (cached, custom estimator like offline apply_mvnn)
python scripts/run_lab_replay.py ... \
    --lab-whitening mvnn --lab-whitening-method custom \
    --lab-whitening-first-block 40

# or reuse a frozen matrix
python scripts/run_lab_replay.py ... \
    --lab-whitening mvnn --lab-whitening-matrix checkpoints/lab_calibration/P04_whitener.npy
```

## 3. Run with the full derived contract

Raw replay (reproduces the live path on recorded data):

```bash
python scripts/run_lab_replay.py --replay-source raw --subject P04 \
    --lab-low-level-checkpoint checkpoints/hierarchical/lab_low_level_starstim31_best.pth \
    --lab-prior-checkpoint checkpoints/hierarchical/lab_prior_starstim31_best_fdn.pt \
    --lab-bandpass-l-freq 0.01 --lab-bandpass-h-freq 30 \
    --lab-ica-operator checkpoints/lab_calibration/P04_ica_operator.npy \
    --lab-whitening mvnn --lab-whitening-first-block 40 \
    --lab-reject-uv 250
```

Live GUI:

```bash
python scripts/run_realtime_gui.py --mode lab-live ... \
    --lab-bandpass-l-freq 0.01 --lab-bandpass-h-freq 30 \
    --lab-ica-operator checkpoints/lab_calibration/P04_ica_operator.npy \
    --lab-whitening-matrix checkpoints/lab_calibration/P04_whitener.npy \
    --lab-reject-uv 250
```

Rejected epochs are dropped in the live runner (no reconstruction) and flagged in
replay metadata (`artifact_rejected`, `peak_to_peak_uv`).

## Caveats

- **ICA** cannot be re-fit per epoch live; the calibrated operator is a fixed
  linear approximation of the offline whole-recording ICA. It reproduces
  `ica.apply` exactly for the calibrated solution (see tests) but does not adapt
  to within-session drift.
- The derived source (`--replay-source derived`) applies only whitening; bandpass
  and ICA belong to the raw→epoch transform and would double-process cached
  epochs.
- Whether the shipped lab checkpoints were trained with MVNN whitening should be
  confirmed against their training config before enabling it in production; the
  machinery here supports both on/off.
</content>
