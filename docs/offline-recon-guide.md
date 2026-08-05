# Offline QSM/R2* Reconstruction — Usage Guide

This covers running `offline-qsm-recon` — a standalone Docker image for batch QSM/R2*
reconstruction against a folder of archived DICOMs. **This is not the Siemens Open Recon
scanner app** (see the main [readme.md](../readme.md) for that) — there's no
Injector/Emitter protocol, no scanner install step, and no `.zip` packaging. You just run a
container against a folder of DICOMs and get DICOMs back out. Useful for research/offline
use: reprocessing archived exams, testing against a dataset without a scanner, or comparing
`masked` vs `wholehead` output for the same acquisition.

Internally it's the exact same reconstruction pipeline as the scanner app
([iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus) for QSM,
[DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo) for R2*) — `offline_recon.py` just
orchestrates the same `dicom2mrd.py` → `qsm.py` → `mrd2dicom.py` chain used everywhere else
in this repo, via a local, loopback-only MRD server+client pair, so there's nothing
different about the actual reconstruction math or output values versus the scanner app.

## 1. Prerequisites

- **Docker**, with the **NVIDIA Container Toolkit** if you want GPU acceleration (strongly
  recommended — see [Expected timing](#5-expected-timing) below). CPU-only works too, just
  much slower.
- **~8GB of free disk space** for the image itself, plus room for your input/output DICOMs.
  Each reconstruction also writes a full copy of your output DICOMs and an intermediate
  `.mrd` file to a temp directory inside the container, cleaned up automatically when the
  run finishes.
- A folder of **multi-echo GRE DICOMs** (magnitude + phase) from the same acquisition —
  see [Input format](#3-input-format) below.

## 2. Getting the image

If you already have this repo cloned and `openrecon-qsm:prod` built (see the main
[readme.md](../readme.md)'s "Building the Docker image" section), building the offline
image on top of it is fast (~1s — it just adds one small script layer):

```bash
docker build -f docker/offline_recon.dockerfile -t offline-qsm-recon .
```

Confirm it's ready:

```bash
docker run --rm offline-qsm-recon --help
```

If you're on Apple Silicon and see a `platform (linux/amd64) does not match ... (linux/arm64)`
warning on `docker run` — that's expected and harmless (the image runs under emulation);
it's not needed on `docker build` for this particular file since it inherits its platform
from the already-built `openrecon-qsm:prod` base image.

## 3. Input format

A folder containing the **magnitude and phase DICOMs of one multi-echo GRE acquisition**
(any number of echoes ≥ 1; R2* mapping needs ≥ 2). Mixed studies, single-echo-only folders,
or a folder missing either magnitude or phase will fail with a clear error rather than
produce wrong output. Nested subfolders are fine — the folder is walked recursively.

## 4. Running it

```bash
docker run --rm --gpus all \
    -v /path/to/dicoms:/input:ro \
    -v /path/to/output:/output \
    offline-qsm-recon --input /input --output /output --mode masked
```

- `-v /path/to/dicoms:/input:ro` — your input DICOM folder, mounted read-only.
- `-v /path/to/output:/output` — where the result DICOMs get written (created if it
  doesn't exist).
- `--gpus all` — optional. Drop it entirely on a host with no GPU/NVIDIA Container Toolkit;
  the pipeline falls back to CPU automatically (just far slower).
- `--mode masked` (default) or `--mode wholehead` — see [Options reference](#6-options-reference).
- Add `-v` (verbose) for detailed per-stage logging, useful the first time you run this or
  when troubleshooting.

The container prints top-level progress (`Converting DICOMs...`, `Starting reconstruction
server...`, `Running reconstruction...`, `Converting results to DICOM...`, `Done.`) as it
goes, so `docker logs -f <container>` (if run detached) or plain foreground output shows
where it's at.

## 5. Expected timing

Measured on this repo's own test dataset (256×192×176 matrix, 176 slices, 5 echoes) under
CPU-only execution (no GPU available in that test environment):

| Mode | QSM | R2* | Total (CPU) |
|---|---|---|---|
| `masked` | ~3-5 min | ~10-12 min | ~15-20 min |
| `wholehead` | ~20-25 min | ~35-40 min | ~1 hour |

`wholehead` is substantially slower than `masked` for both QSM and R2* — the brain mask
lets both pipelines skip most of the non-brain field of view. **A real GPU should be
dramatically faster** (the underlying networks are small enough that GPU inference is
typically an order of magnitude or more faster than CPU for volumes this size), but this
hasn't been independently timed on GPU hardware as part of this repo's own testing --
budget accordingly until you've measured it on your own hardware, and don't be alarmed if
a CPU-only run takes the better part of an hour for `wholehead`.

## 6. Options reference

Run `docker run --rm offline-qsm-recon --help` for the authoritative list. Summary:

| Flag | Default | Purpose |
|---|---|---|
| `--input` | *(required)* | Folder of input DICOMs (mount into the container, e.g. `/input`) |
| `--output` | *(required)* | Folder to write result DICOMs to (e.g. `/output`) |
| `--mode` | `masked` | `masked` (brain-extracted via FSL's `bet2`) or `wholehead` |
| `--port` | `9002` | Internal loopback port for the local reconstruction server -- not exposed outside the container, essentially never needs changing |
| `-v` / `--verbose` | off | Detailed per-stage logging |

## 7. Output format

DICOM files in `--output`, one file per slice per series, named
`<series>_<protocol-name>_<slice>.dcm` (matching `mrd2dicom.py`'s existing convention). One
QSM series and, if the input had ≥ 2 echoes, one R2* series — plus the original
input magnitude/phase series passed through unmodified (so nothing from the source
acquisition is lost).

| Series | Content | Units | Present when |
|---|---|---|---|
| `100` | QSM, brain-extracted | ppb (`RescaleType=PPB`) | `--mode masked` |
| `101` | QSM, whole-head | ppb (`RescaleType=PPB`) | `--mode wholehead` |
| `102` | R2*, brain-extracted | s⁻¹ (`RescaleType=R2S`) | `--mode masked` and ≥2 echoes |
| `103` | R2*, whole-head | s⁻¹ (`RescaleType=R2S`) | `--mode wholehead` and ≥2 echoes |
| *(other)* | Original input magnitude/phase, unmodified | — | always |

Real-world values are recovered the standard DICOM way: `real = raw_pixel * RescaleSlope +
RescaleIntercept` (any compliant DICOM viewer does this automatically).

## 8. Troubleshooting

- **`torch.cuda.is_available()` is `False` / everything runs on CPU despite `--gpus all`** —
  this is a host GPU/Docker configuration issue, not specific to this image. Check, in
  order: (1) `docker inspect <container> --format '{{.HostConfig.DeviceRequests}}'` to
  confirm `--gpus all` was actually attached; (2) `docker exec <container> nvidia-smi`
  works from inside the container; (3) the host driver supports at least CUDA 11.8
  (`nvidia-smi`'s header on the host, not just the container); (4) if `nvidia-smi` works in
  the container but `torch.cuda.is_available()` still returns `False`, check for a stale
  CDI spec (`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml` regenerates it) --
  a known issue where `/dev/nvidia-uvm`'s kernel-assigned major number drifts after a
  driver update and the CDI spec falls out of sync.
- **Reconstruction seems to hang with no output for a long time** — expected for
  `wholehead` mode's R2* stage specifically (DeepRelaxo's per-voxel estimator has no
  internal progress logging over that ~35-40 min CPU stretch — see
  [Expected timing](#5-expected-timing)). `-v` shows the surrounding stages clearly; the
  silence is bounded to just that one stage.
- **Disk space** — the base image is ~7GB. Each run additionally needs temporary space for
  an intermediate `.mrd` file (roughly proportional to your input DICOM count) and a full
  copy of the output DICOMs, both cleaned up automatically when the container exits. Keep
  an eye on `docker system df` if you're running this repeatedly — Docker's build cache and
  old image layers can accumulate significant space over many rebuilds.
- **`--input directory not found`** — check the host path in `-v host_path:/input:ro`
  actually exists and is readable; this error means the container-internal `/input` path
  (post-mount) wasn't a directory, usually because the host-side path was wrong or the
  mount itself failed silently.
