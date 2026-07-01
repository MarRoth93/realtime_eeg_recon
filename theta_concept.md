# Theta Concept

## Design

Treat this as a closed-loop control problem in latent space.

The goal is not:
- make EEG match some invented target waveform

The goal is:
- apply stimulation so the future encoded EEG moves in a desired direction, like "more hue"

So the system would have four parts.

### 1. `E(x)` EEG encoder

- Input: preprocessed EEG trial `x`
- Output: latent `z`
- This is the existing EEG-to-latent model

### 2. `d_hue` latent direction

- A vector in latent space representing "more hue"
- If current latent is `z`, target latent is:
  - `z_target = z + alpha * d_hue`

### 3. `F(x, u)` stimulation-response model

- Predicts what happens to the brain state after stimulation `u`
- Could predict:
  - next EEG: `x_next`
  - or directly next latent: `z_next`
- Prefer predicting latent directly if possible:
  - `z_next = F(z, u)` or `z_next = F(x, u)`

### 4. Controller

- Chooses stimulation `u` so predicted future latent moves toward `z_target`

## Why Latent-Space Control

Because that is the actual objective.

The target is:
- image property shift

The model measures that in:
- latent space

So it is cleaner to control:
- latent movement

than to control:
- exact EEG waveform

Exact EEG is messy, high-dimensional, and non-unique.

## How To Build It

### Step 1. Define the hue direction

Need a robust `d_hue`.

Possible ways:
- fit a linear direction from image or latent annotations
- contrast latents from low-hue vs high-hue images
- regress image hue statistics onto latent coordinates

Result:
- a normalized vector `d_hue`

Then for any trial:
- current latent `z0 = E(x0)`
- desired latent `z* = z0 + alpha * d_hue`

### Step 2. Collect stimulation calibration data

Need data of the form:
- pre-stim EEG or latent
- stimulation parameters
- post-stim EEG or latent

For each trial, record:
- baseline EEG before stimulation
- stimulation settings `u`
  - timing
  - amplitude
  - frequency
  - location or montage
  - pulse train parameters
- EEG after stimulation
- encoded latent before and after

So each sample looks like:
- `(x_t, z_t, u_t) -> (x_t+1, z_t+1)`

This is the crucial dataset.

### Step 3. Fit a stimulation-response model

Simplest useful model:
- predict latent change instead of raw latent:
  - `delta_z = F(z_t, u_t)`
  - then `z_t+1 = z_t + delta_z`

That is better than predicting `z_t+1` directly because:
- easier to learn small changes
- closer to the intervention effect

Could start with:
- linear model
- small MLP
- recurrent model if history matters

For example:
- input: current latent `z_t` + stimulation params `u_t`
- output: predicted `delta_z`

Then the model answers:
- if stimulation is applied like this, how will the encoded brain state shift?

### Step 4. Choose stimulation by optimization

Once `F` exists, the controller solves:

- choose `u` that makes predicted `z_next` close to `z_target`

Simple objective:
- latent target loss:
  - `|| z_pred(u) - z_target ||^2`

Also need penalties:
- stimulation energy penalty
- smoothness or stability penalty
- hard safety bounds

So more realistically:

- minimize:
  - target error
  - plus `lambda1 * stimulation magnitude`
  - plus `lambda2 * deviation from safe/default settings`

subject to:
- amplitude limits
- frequency limits
- duration limits
- montage constraints

Then:
- apply the best safe `u`
- observe actual EEG
- re-encode
- repeat

That gives a closed loop.

## Two Design Choices

### A. One-step controller

- one stimulation decision per trial
- simpler
- good first prototype

Flow:
1. encode baseline EEG
2. compute `z_target`
3. optimize one `u`
4. stimulate
5. observe next EEG
6. check whether latent moved toward target

### B. Multi-step controller

- plan several stimulation steps
- more powerful
- much harder

Then use model-predictive control:
- simulate several future latent states
- optimize a sequence `u_1, ..., u_T`

That is the more advanced version.

## What To Model

Start with:

- `z_next = z + F(z, u)`

not:
- `x_target -> stimulation`

and not:
- `u -> exact EEG waveform`

Because latent control is:
- lower-dimensional
- closer to the image objective
- easier to optimize

## Where Target EEG Fits

If still wanting an EEG-like target, use it only as a regularizer.

For example:
- infer a plausible `x_target` whose encoding is near `z_target`
- but the controller still optimizes stimulation for latent movement, not waveform matching

So target EEG is optional, not central.

## How To Validate It

Before any real closed loop:

### 1. Offline simulation

- take recorded EEG
- encode it
- define target latent shifts
- use learned `F` to choose virtual stimulations
- test whether predicted latent moves correctly

### 2. Held-out stimulation data

- compare predicted vs actual latent shifts

### 3. Only then real closed loop

- small target shifts
- conservative stimulation set
- monitor whether latent moves in the intended direction

Success metrics:
- latent displacement along `d_hue`
- image hue shift after decoding
- safety or stability of EEG
- reproducibility across trials

## What Can Go Wrong

Main failure modes:
- `d_hue` is not actually reachable from EEG latents
- stimulation effect is too weak or noisy
- response varies too much by subject or state
- stimulation artifacts corrupt EEG
- model learns spurious correlations

So the first real question is not control. It is:
- does stimulation reliably move the latent at all?

If not, the rest collapses.

## Minimal Version To Build First

1. compute `d_hue`
2. encode many natural EEG trials
3. verify that moving along `d_hue` changes decoded hue meaningfully
4. collect stimulation calibration data
5. fit `delta_z = F(z, u)`
6. test whether some `u` reliably increases projection onto `d_hue`

That projection is:
- `(z_next - z) dot d_hue`

If that term can be controlled, then the design is viable.
