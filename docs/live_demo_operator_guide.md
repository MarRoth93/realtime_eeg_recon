# Live Demo Operator Guide

This is the operating procedure for a new, previously unseen participant. The
live demo is deliberately separate from the P07 replay setup: participant codes
are session labels only and are never reused as model subject IDs.

## Start the demo

From the project root, run:

```bash
./scripts/start_live_demo.sh
```

The launcher uses `config/lab_demo.json`, loads the native Starstim31 low-level
checkpoint, starts the browser server on port 8010, and opens the setup page.
Model loading can take a little while; the server starts after the models are
ready. If startup fails, the browser remains available with the startup error.

The configuration file stores the model paths, preferred LSL names, trigger
map, image root, port, and high-level policy. Operators should not need to type
these values during a demonstration.

## Setup sequence

1. Fit the Starstim cap and perform the impedance/contact check in NIC2. The
   reconstruction GUI cannot measure impedance itself.
2. Start the NIC2 32-channel, 500 Hz LSL EEG stream.
3. Start the PsychoPy experiment and leave it at its waiting screen so that its
   marker outlet exists.
4. In the demo browser, enter an anonymous participant code.
5. Select **Refresh devices**. The recorded lab convention is:
   - EEG: `eeg_recon_pyexp-EEG`
   - markers: `eeg_recon_pyexp`
6. Select **Connect and check**. The runner waits indefinitely for late streams;
   a missing device no longer terminates the session after ten seconds.
7. Wait for live EEG, the expected `32 channels / 500 Hz` format, and at least
   one second of buffered data.
8. If NIC2 exposes only generic labels such as `Ch1..Ch32`, verify that NIC2 uses
   the expected Starstim order and select **Confirm NIC2 channel order**.
9. Select **Arm session** only after all readiness checks are green.
10. Start stimulus presentation in PsychoPy.

The GUI does not claim that signal contact is good. It reports flat, non-finite,
or unusually large signal variation as a warning; NIC2 remains authoritative
for impedance.

## What starts reconstruction

The preferred PsychoPy marker contract is:

```json
{"event":"marker_test","session_id":"demo-001"}
{"event":"experiment_start","session_id":"demo-001"}
{"event":"stim_onset","session_id":"demo-001","trial_id":1,"block":1,"image_id":"beaver_13n"}
{"event":"experiment_pause","session_id":"demo-001"}
{"event":"experiment_resume","session_id":"demo-001"}
{"event":"experiment_end","session_id":"demo-001"}
```

`stim_onset` must be sent on the actual image flip. Reconstruction markers are
ignored before the session is armed and, for the JSON protocol, before
`experiment_start`. Duplicate trial IDs and mismatched session IDs are rejected
and recorded. A task that sends periodic `heartbeat` markers can enable
`marker_heartbeat_timeout_seconds` in the demo configuration so a disappeared
PsychoPy outlet invalidates readiness even while no stimuli are being shown.

The existing PsychoPy task can continue to emit numerical image markers. The
demo launcher loads the authoritative whitelist in
`data/derived/finetune/lab_target_manifest.tsv`; only its 160 mapped stimulus
codes are treated as targets. Other task codes such as fixation and instruction
markers are shown as ignored. For this legacy path, the first mapped stimulus
after Arm starts the run and is also reconstructed, so the first trial is not
lost. The operator ends a legacy session with **Stop session**.

## New-participant model policy

The default demo uses the plain Starstim31 low-level checkpoint. It does not ask
the operator to invent a subject ID.

High-level reconstruction is disabled by default for an unseen participant. The
shipped ATMS checkpoint contains a trained shared/unknown-participant token, and
the runtime supports it as `shared_unseen`, but the downstream diffusion prior
was trained with manifest participant IDs. To try that path while keeping its
status explicit, set:

```json
"enable_experimental_unseen_high_level": true
```

in `config/lab_demo.json`. The GUI labels this output as experimental. It never
silently substitutes P07, subject 0, or another known participant.

## Runtime states

The phase shown at the top of the browser is authoritative:

```text
LOADING MODELS
-> WAITING FOR EQUIPMENT
-> READY
-> ARMED
-> RUNNING
-> FINISHED
```

A lost EEG/marker connection or cleared EEG buffer disarms the session. After
recovery, the operator must confirm readiness and Arm again; the runner never
resumes reconstruction unexpectedly. Missing post-stimulus EEG drops that trial
after a timeout instead of hanging the whole session. A preprocessing, target,
disk, or inference error is isolated to its trial where possible.

## Outputs

Every launch receives an independent directory:

```text
outputs/gui_lab_live/sessions/<timestamp>_live/
  session.json
  events/
  low_level/
  high_level/
  targets/
```

`session.json` records the anonymous participant code, model policy, selected
streams, expected acquisition format, and final trial counts. Event files retain
the raw marker, normalized event, session/trial metadata, target identity, and a
clear accepted/dropped reason.

## Required hardware rehearsal

Before a public or participant-facing run:

1. Run the full setup with NIC2 and PsychoPy but no participant.
2. Send a marker test or one short dummy block.
3. Confirm the first stimulus is not lost and that target images resolve.
4. Disconnect and reconnect EEG once; verify the GUI disarms and requires Arm
   again.
5. Compare one saved live event against its recorded `.easy`/XDF replay.

Software tests cover the lifecycle and marker contracts, but NIC2 discovery,
browser rendering, impedance, and real device reconnect behavior still require
this room-level check.
