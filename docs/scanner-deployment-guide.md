# iQSM_Plus Open Recon — Scanner Deployment Guide

This covers installing, running, debugging, and uninstalling the `iQSM_Plus` Open Recon
application on the scanner console, plus how it relates to the local build/packaging steps
covered elsewhere in this repo (`docker/qsm.dockerfile`, `docker/build_openrecon_package.py`).

## 1. Prerequisites

- **Numaris XA60 or XA61** (or the more limited XA51 version).
- **High-end (HE) MaRS** hardware, with the **Open Recon product license** installed and active.
  Verify via: "?" icon (top right) → About → System Info → confirm Open Recon is listed among
  installed product licenses. Without this, the Inline card won't show an Open Recon sub-card
  at all, regardless of whether the app itself installs correctly.
- A packaged `.zip` built via `docker/build_openrecon_package.py` (see the main repo README/
  conversation history for how this is produced) — e.g.
  `OpenRecon_HMRIImagingCentre_iQSMPlus_V1.0.1.zip`.

## 2. Installation

1. On the scanner console, **exit kiosk mode**: `Tab + Del + NumPad+`, then enter admin
   credentials.
2. **Copy the `.zip`** into:
   ```
   C:\Program Files\Siemens\Numaris\OperationalManagement\FileTransfer\incoming
   ```
   This folder requires escalated privileges (same level as kiosk mode). If a direct copy
   fails, copy to `Downloads` first, then move it into `incoming` from there.
3. **Wait 2-3 minutes.** The folder is continuously monitored; the `.zip` disappears
   automatically once successfully installed.
4. **Confirm installation**: check the per-app install log at
   `C:\ProgramData\Siemens\Numaris\log\syngo.MR.HostInfra.OpenRecon.Watcher\`, or look for the
   app in `C:\ProgramData\Siemens\Numaris\MriSiteData\Scanner\OpenReconStore.json` (a cache of
   all installed apps' parsed configs).

## 3. Configuring a protocol to use it

1. Open **myExam Cockpit**, open your ME-GRE protocol in **edit mode**.
2. Go to **Inline → Open Recon**, select **"iQSM_Plus"** from the dropdown.
3. Under the protocol's **`Execution` tab**, confirm **`Prio Recon` is disabled** — a hard
   requirement for any Open Recon app to run at all.
4. Confirm the sequence outputs **both magnitude and phase images** — `qsm.py` will error out
   (visibly, via a logged message, no images returned) if phase images aren't emitted, or if
   magnitude/phase counts don't match.
5. Some ME-GRE protocols emit **two magnitude series** (raw + B1/intensity-corrected) alongside
   phase — three series total. `qsm.py` handles this automatically: it inspects each series'
   DICOM `ImageType` and prefers whichever one carries Siemens' `NORM` flag (PreScan Normalize /
   B1-intensity-corrected); if that flag isn't present or is ambiguous, it falls back to the
   lowest `image_series_index`. Either way it logs a warning naming which series indices it saw
   and which one it picked, and ignores the rest. Which one is actually used shouldn't matter for
   correctness — iQSM+ only uses magnitude for the final multi-echo weighted combination of
   per-echo chi maps, and a purely spatial per-voxel correction (which is what B1/intensity
   correction is) cancels out of that weighted average. If you ever need to force a specific
   series regardless, set the `QSM_MAGNITUDE_SERIES_INDEX` environment variable on the container.
6. Note: once installed, this app is available for **any** protocol on the system, not scoped
   to one sequence — Open Recon has no mechanism to restrict an app to specific sequence types.
   It's up to whoever operates the scanner to only select it for genuinely compatible
   acquisitions (multi-echo GRE with magnitude + phase output).

## 4. Running a scan

1. Position the patient/subject and run the protocol normally.
2. ICE streams acquired data to the container automatically once the protocol starts — no
   separate action needed.
3. **Reconstruction takes several minutes** (measured 4-8 minutes in testing, GPU-dependent) —
   the acquisition itself finishes well before QSM reconstruction completes. Images appear once
   processing finishes, not incrementally during the scan.
4. The reconstructed QSM series appears as a new image series (DICOM), viewable on the console
   or via your normal PACS workflow.
5. App documentation (the packaged `docs.pdf`) is accessible via "?" → Help → Library →
   double-click "iQSM_Plus" in the File column.

## 5. Debugging

### General log locations

| Log/file | Purpose |
|---|---|
| `C:\ProgramData\Siemens\Numaris\log\syngo.MR.HostInfra.OpenRecon.Watcher\` | Install-time log, one file per app |
| `C:\ProgramData\Siemens\Numaris\log\syngo.MR.Exam.Appl.utr` | Parses `OpenReconStore.json` into the UI — check here if the app doesn't appear in the Inline dropdown |
| `C:\ProgramData\Siemens\Numaris\log\OpenRecon.utr` | **Main runtime log** — includes stdout/logging output from inside the Docker container itself (the same kind of messages you'd see via `docker logs` when testing locally: `Buffered N magnitude and N phase images...`, `QSM parameters: ...`, `QSM reconstruction time: ...`, or any `logging.error(...)` from `qsm.py`) |
| `C:\ProgramData\Siemens\Numaris\MriSiteData\Scanner\OpenReconStore.json` | Cached, parsed config of all installed apps — can be hand-edited for temporary debugging (gets regenerated on every install, so treat edits as throwaway) |
| Message Viewer (gear icon → Administration Portal → log in as `medadmin` → Message Viewer, filter "Open Recon") | Installation-stage messages/errors |

### Benign log messages (not errors)

**`OpenRecon.utr` shows "Used OpenRecon image has NO valid signature"**
→ Expected, not a failure. Image signing is reserved for Siemens-released clinical apps (see
`or_sdk/README.md`: *"Open Recon is to add clinical reconstructions to the system, if signed and
released for clinical use by Siemens Healthineers. Any other recon used e.g., by researchers is
automatically labelled not for diagnostic use..."*). Since `iQSM_Plus` is configured with
`content_qualification_type: "RESEARCH"` and nothing in this build pipeline signs the image, this
line is expected to appear on every install/run and does not block installation or reconstruction
(confirmed on real hardware: the app installed and reconstructed successfully alongside this log
line). Only worth investigating further if the app *also* fails to install or run.

### Common failure modes

**Open Recon card missing entirely from Inline**
→ License not installed. Check System Info (see Prerequisites above).

**`.zip` stays in the `incoming` folder for more than ~5 minutes**
→ Installation was rejected. Most common causes:
- Filename mismatch — the `.zip`, the `.tar` inside it, and the `.pdf` inside it must **all**
  be named exactly `OpenRecon_<Vendor>_<Name>_V<version>` (e.g.
  `OpenRecon_HMRIImagingCentre_iQSMPlus_V1.0.1`). `docker/build_openrecon_package.py` handles
  this automatically — only relevant if you build a package by hand.
- Wrong zip compression (Deflate64 instead of Deflate/Store) — not an issue if you used
  `build_openrecon_package.py`'s `zip -X -0` approach, but is a common trap with Windows
  Explorer's "Send to → Compressed folder" for files over 2GB.
- Docker image manifest format incompatibility — if the image was `docker save`'d with a
  Docker Engine version ≥25.0.0 without the `docker:24.0-dind` conversion step, Open Recon
  can't parse it. Check for an `index.json`/`oci-layout` file at the tar root (OCI format,
  incompatible) vs `manifest.json`/`repositories` only (legacy format, compatible).

**App doesn't appear in the Inline → Open Recon dropdown, despite installing successfully**
→ Usually a JSON config schema validation failure. Check
`C:\ProgramData\Siemens\Numaris\log\syngo.MR.Exam.Appl.utr` in Logviewer. Validate the source
JSON against `OpenReconSchema_1.1.0.json` before packaging next time (already automated in
`docker/build_openrecon_package.py`'s `validate_json()` step).

**Reconstruction fails / no images returned after a scan**
→ Check `OpenRecon.utr` for the actual Python traceback/error message. Common root causes given
how `qsm.py` is written:
- "No phase images received" — the protocol didn't have phase image output enabled.
- "Could not determine echo time (TE) for all N echoes" — TE metadata wasn't present/parseable
  from the incoming MRD images for one or more echoes.
- A CUDA out-of-memory error — see the GPU memory discussion above; if this happens,
  `min_required_gpu_memory` in the JSON config needs to be raised so Open Recon doesn't offer
  the app on insufficient hardware.
- If the container fails to even start, check that the image genuinely is `linux/amd64` (not
  accidentally built for a different architecture) and that the Docker version conversion step
  was applied correctly during packaging.

**Reconstruction runs but produces obviously wrong-looking DICOM images**
→ This exact class of bug came up repeatedly during development — a stale server process still
running old in-memory code after a `qsm.py` edit will silently keep using the outdated logic
indefinitely, even though the file on disk (and even a freshly-rebuilt Docker image) has the
fix. On the scanner, the equivalent is: if you install an *updated* version of the app, make
sure the previous container isn't somehow still running — Open Recon should start a fresh
container per scan, but if you're troubleshooting via manual `docker run` on a test machine,
always confirm you're running the intended image tag/version, not a leftover container from an
earlier test.

### Testing changes before touching the scanner again

Recall the offline test loop already established for this project — always exhaust this before
reinstalling on real hardware:
1. Local venv + `main.py`/`client.py` test (fastest iteration).
2. Containerized test (`docker run` the built image, same `client.py` test) — validates the
   actual packaged artifact, not just the source code.
3. Only once both pass: rebuild via `docker/qsm.dockerfile`, repackage via
   `docker/build_openrecon_package.py`, and reinstall.

## 6. Uninstalling / updating

### Remove a single app
Use the **Package Remover** tool (`or_sdk/tooling/PackageRemover/`), version-specific for
VA60A vs VA61A(SP01) — follow `OpenReconPackageRemover guide.pdf` in that folder for install
instructions and usage.

### Remove all installed Open Recon apps
Run `or_sdk/tooling/RemoveInstalledOpenReconApps.bat` **with administrative privileges**. A
"Restart Workspace" or full scanner reboot is required afterward for the removal to take full
effect.

### Updating to a new version
There's no separate "update" mechanism — installing a new package with the same `id` (from the
JSON config's `general.id` field, e.g. `"iQSMPlus"`) but a different `version` should be
supported as an update via the normal install path (copy the new `.zip` into `incoming`), but
if in doubt, uninstall the old version first via the Package Remover before installing the new
one, to avoid any ambiguity about which container image actually runs.

## 7. Reference: what's actually running

For context when debugging — a quick recap of the architecture from local testing:

- The scanner always sends the literal config value `"openrecon"`, never the app's actual name.
  `qsm.py` gets selected because the Docker image's `CMD` passes `--defaultConfig=qsm`
  (`docker/qsm.dockerfile`), not because of anything in the JSON config's naming.
- The JSON config (embedded as a base64-encoded Docker image `LABEL`, not a loose file anywhere
  on the scanner) only controls the UI/parameter card and hardware requirement gating — it has
  no direct link to which Python module actually executes.
- One container is started per scan; it listens on port 9002 (must match the JSON config's
  `reconstruction.port` field) and receives the same MRD streaming messages
  (`MRD_MESSAGE_CONFIG_FILE`, `MRD_MESSAGE_METADATA_XML_TEXT`, one `MRD_MESSAGE_IMAGE` per
  image, `MRD_MESSAGE_CLOSE`) that `client.py` sends during local testing — the server-side code
  can't distinguish a real scan from an offline test.
