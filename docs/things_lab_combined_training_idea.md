# THINGS + Lab Training Idea For Lab-Native Models

## Question

The current setup uses:

- a Starstim31 ATMS encoder trained mainly from THINGS EEG adapted/preprocessed
  into the Starstim31-compatible format,
- a lab-native low-level model trained only on lab EEG,
- a lab-native diffusion prior trained only on lab EEG.

The question is whether the low-level model and diffusion prior may improve if
they are trained with both:

- preprocessed THINGS EEG in the Starstim31-compatible format,
- lab EEG from all subjects except the held-out test subject.

For a P07 holdout, this would mean:

```text
training lab subjects: P04, P05, P06, melina, zar
held-out test subject: P07
```

## Why This Could Help

The lab-only training set is small. With P07 held out, the current lab training
set has 971 usable trials. That is limited for both:

- EEG-to-VAE-latent low-level reconstruction,
- EEG-embedding-to-image-embedding diffusion prior training.

The THINGS data provides much more EEG-image supervision. If THINGS is
preprocessed into the same Starstim31-compatible shape, it can act as a larger
source dataset and may make the models less fragile and less overfit to the
small lab set.

The diffusion prior is a particularly plausible candidate for this, because it
learns:

```text
ATMS EEG embedding -> image CLIP embedding
```

and the ATMS encoder itself already comes from the THINGS-compatible training
path.

The low-level model may also benefit, but it is more sensitive to hardware,
preprocessing, and signal-distribution differences.

## Main Risk

The main risk is domain mismatch.

Even if both datasets are shaped as:

```text
31 channels x 250 samples
```

THINGS EEG adapted to Starstim31 format is not identical to real Starstim lab
EEG. The recording hardware, original channel layout, preprocessing history,
task timing, subject pool, noise profile, and stimulus protocol may differ.

If THINGS dominates training, the model may become better for THINGS-like EEG
but worse for the real lab EEG.

## Recommended Strategy

The preferred approach is not to blindly mix everything from scratch. A cleaner
approach is:

```text
1. Pretrain on Starstim31-compatible THINGS EEG
2. Fine-tune on lab subjects excluding the held-out subject
3. Evaluate on the held-out lab subject
```

For P07:

```text
pretrain:  THINGS Starstim31-compatible data
fine-tune: P04, P05, P06, melina, zar
test:      P07
```

This keeps the held-out subject clean while still using the larger THINGS
dataset to initialize the models.

## Comparisons To Run

The useful comparison would be:

```text
A. Lab-only, P07 held out
B. THINGS pretrain -> lab fine-tune, P07 held out
C. THINGS + lab mixed training from scratch, P07 held out
```

Expected outcome:

```text
B is most likely to improve.
C might help, but could hurt if THINGS dominates.
A is the clean baseline.
```

## Leakage Notes

Using P07 as the held-out subject avoids subject leakage only if P07 is excluded
from all low-level and diffusion-prior training.

There may still be image/stimulus leakage if the same exact target images appear
in THINGS training and P07 evaluation. That may be acceptable if the goal is to
reconstruct known experiment images from a new subject, but it is not a strict
unseen-image evaluation.

For the current realtime goal, the most relevant no-subject-leakage test is:

```text
train: all available non-P07 data
test:  P07
```

The stronger transfer experiment is:

```text
pretrain: THINGS Starstim31-compatible data
fine-tune: non-P07 lab data
test: P07
```
