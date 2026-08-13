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

> **Rebuild the base image first if any of the reconstruction code changed.**
> `offline_recon.dockerfile` is `FROM openrecon-qsm:prod` and only copies
> `offline_recon.py` on top — everything that does the actual reconstruction (`qsm.py`,
> `dicom2mrd.py`, `mrd2dicom.py`, iQSM_Plus, DeepRelaxo) comes from that base. So a
> rebuild of *this* file picks up changes to `offline_recon.py` only, and silently keeps
> whatever version of the pipeline the base image was last built with — no error, no
> warning, just stale results. After changing anything else, rebuild both:
>
> ```bash
> docker build --platform linux/amd64 -f docker/qsm.dockerfile -t openrecon-qsm:prod .
> docker build -f docker/offline_recon.dockerfile -t offline-qsm-recon .
> ```
>
> To check what a built image actually contains:
> ```bash
> docker run --rm --entrypoint sh offline-qsm-recon -c 'grep -n "teWeights = " /opt/code/python-ismrmrd-server/qsm.py'
> ```

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
mkdir -p /path/to/output
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
    -v /path/to/dicoms:/input:ro \
    -v /path/to/output:/output \
    offline-qsm-recon --input /input --output /output --mode masked
```

- `-v /path/to/dicoms:/input:ro` — your input DICOM folder, mounted read-only.
- `-v /path/to/output:/output` — where the result DICOMs get written.
- `--user "$(id -u):$(id -g)"` — **recommended.** Containers run as `root` by default, so
  without this every output DICOM lands owned by `root:root` and you need `sudo` to
  delete, move or overwrite them. Running as yourself makes the output ordinary files you
  own. The pipeline needs no root privileges, so this is safe.
- `mkdir -p /path/to/output` beforehand — **required when using `--user`**, not just tidy.
  If the mount target doesn't exist, Docker creates it as `root:root` *before* the
  container starts; the container then runs as your (non-root) UID and cannot write into
  it, so the run fails at the very last step with:

  ```
  PermissionError: [Errno 13] Permission denied: '/output/00_..._001_e05.dcm'
  ```

  Frustratingly this comes *after* the whole reconstruction has completed. Pre-creating
  the folder is worth doing even without `--user`: removing a file depends on write
  permission on its **parent directory**, not on who owns the file, so a root-owned output
  folder needs `sudo` even to clear out.

  To recover from an already-created root-owned folder, no `sudo` is needed as long as you
  own its parent — `rmdir /path/to/output && mkdir -p /path/to/output` (use
  `sudo rm -rf` instead if it already has files in it).
- `--gpus all` — optional. Drop it entirely on a host with no GPU/NVIDIA Container Toolkit;
  the pipeline falls back to CPU automatically (just far slower).
- `--mode masked` (default) or `--mode wholehead` — see [Options reference](#6-options-reference).
- `--bet-threshold` — only relevant with `--mode masked`; see
  [Tuning the brain mask](#tuning-the-brain-mask) below.
- Add `-v` (verbose) for detailed per-stage logging, useful the first time you run this or
  when troubleshooting.

### Tuning the brain mask

`--mode masked` runs FSL's `bet2` on the first echo to get a brain mask, using a
fractional intensity threshold that defaults to `0.4` (bet2's own default is `0.5`; this
repo defaults slightly lower after finding `0.5` under-inclusive on real scanner data). If the
resulting QSM/R2* mask comes out **over-inclusive** (extra skull/scalp/non-brain tissue
included) or **under-inclusive** (brain tissue cut off), re-run the same input with
`--bet-threshold` adjusted:

- **Smaller** values (e.g. `0.3`) → a **larger** brain outline.
- **Larger** values (e.g. `0.7`) → a **smaller/tighter** brain outline.

```bash
docker run --rm --gpus all \
    -v /path/to/dicoms:/input:ro \
    -v /path/to/output:/output \
    offline-qsm-recon --input /input --output /output --mode masked --bet-threshold 0.35
```

Must be between `0.0` and `1.0` — out-of-range values fail fast with a clear error before
any reconstruction work starts. Has no effect in `--mode wholehead` (no brain extraction
runs at all in that mode).

The container prints top-level progress (`Converting DICOMs...`, `Starting reconstruction
server...`, `Running reconstruction...`, `Converting results to DICOM...`, `Done.`) as it
goes, so `docker logs -f <container>` (if run detached) or plain foreground output shows
where it's at.

## 5. Expected timing

Measured on this repo's own test dataset (`data/SWI` — 256×192×176 matrix, 176 slices,
5 echoes, 1760 input DICOMs).

**On GPU** (NVIDIA RTX A4500 20GB, driver 550.163.01, torch 2.3.0+cu118):

| Mode | QSM (iQSM+) | R2* (DeepRelaxo) | Total wall-clock |
|---|---|---|---|
| `masked` | ~8 s | ~10 s | **~50 s** |
| `wholehead` | ~11 s | ~44 s | **~85 s** |

**On CPU only** (no GPU available):

| Mode | QSM | R2* | Total (CPU) |
|---|---|---|---|
| `masked` | ~3-5 min | ~10-12 min | ~15-20 min |
| `wholehead` | ~20-25 min | ~35-40 min | ~1 hour |

Two things to note about the GPU totals. First, roughly **30 s of each total is fixed
overhead** — DICOM→MRD conversion in, MRD→DICOM conversion out, server startup and MRD
streaming — and is independent of mode and of GPU/CPU; it scales with the *number of input
DICOMs*, not with reconstruction difficulty. Actual inference is only ~20 s (`masked`) and
~55 s (`wholehead`). Second, `masked` additionally spends ~1.6 s on `bet2` brain
extraction, which `wholehead` skips entirely.

`wholehead` is substantially slower than `masked` for both QSM and R2* — the brain mask
lets both pipelines skip most of the non-brain field of view. The gap is widest for R2*
(~4× on GPU: 44 s vs 10 s), since DeepRelaxo's per-voxel estimator cost scales directly
with the number of voxels it has to visit.

Scale roughly linearly with slice count for other matrix sizes, and expect the fixed
overhead above to grow with input DICOM count.

## 6. Options reference

Run `docker run --rm offline-qsm-recon --help` for the authoritative list. Summary:

| Flag | Default | Purpose |
|---|---|---|
| `--input` | *(required)* | Folder of input DICOMs (mount into the container, e.g. `/input`) |
| `--output` | *(required)* | Folder to write result DICOMs to (e.g. `/output`) |
| `--mode` | `masked` | `masked` (brain-extracted via FSL's `bet2`) or `wholehead` |
| `--bet-threshold` | `0.4` | `bet2`'s fractional intensity threshold (0.0-1.0), only used in `--mode masked`. Smaller = larger brain outline; larger = smaller/tighter outline. See [Tuning the brain mask](#tuning-the-brain-mask). |
| `--port` | `9002` | Internal loopback port for the local reconstruction server -- not exposed outside the container, essentially never needs changing |
| `-v` / `--verbose` | off | Detailed per-stage logging |

## 7. Output format

DICOM files in `--output`, one file per slice per echo per series, named
`<series>_<protocol-name>_e<echo>_<slice>.dcm` (`mrd2dicom.py`'s convention). One
QSM series and, if the input had ≥ 2 echoes, one R2* series — plus the original
input magnitude/phase series passed through unmodified (so nothing from the source
acquisition is lost).

The `e<echo>` field is 1-based and always present, including on the single-echo derived
QSM/R2* series (always `e01`). It is load-bearing, not decorative: DICOM `InstanceNumber`
restarts at 1 for every echo, so `<series>_<name>_<slice>` alone is not unique for
multi-echo data — without the echo field each echo overwrites the previous one and only
the last-written survives.

Echo comes **before** slice so that a plain alphabetical listing groups each echo into one
contiguous run of slices. That's the order viewers load a folder in, so
`File > Import > Image Sequence` in ImageJ gives one clean 3D stack per echo rather than
interleaving all echoes together.

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
  `wholehead` mode's R2* stage specifically: DeepRelaxo's per-voxel estimator has no
  internal progress logging, so it goes quiet for ~45 s on GPU, or ~35-40 min on CPU (see
  [Expected timing](#5-expected-timing)). `-v` shows the surrounding stages clearly; the
  silence is bounded to just that one stage.
- **Disk space** — the base image is ~7GB. Each run additionally needs temporary space for
  an intermediate `.mrd` file (roughly proportional to your input DICOM count) and a full
  copy of the output DICOMs, both cleaned up automatically when the container exits. Keep
  an eye on `docker system df` if you're running this repeatedly — Docker's build cache and
  old image layers can accumulate significant space over many rebuilds.
- **`PermissionError: [Errno 13] Permission denied: '/output/...'` at the very end, after
  the reconstruction finished** — you used `--user` but the output folder didn't exist, so
  Docker created it as `root:root` and your non-root container UID can't write into it.
  Fix with `rmdir /path/to/output && mkdir -p /path/to/output` (no `sudo` needed if you
  own the parent directory), then re-run. See [Running it](#4-running-it).
- **Output DICOMs are owned by `root` / "permission denied" deleting them** — you ran
  without `--user "$(id -u):$(id -g)"` (see [Running it](#4-running-it)). The files are
  mode `644`, so they still *open* fine in any viewer; only writing/deleting is blocked.
  Take ownership of what's already there with
  `sudo chown -R "$(id -u):$(id -g)" /path/to/output`, and add `--user` next time. If the
  output *folder* itself is root-owned (Docker created it because it didn't exist), you'll
  need `sudo` for that `chown` even though you can read everything inside.
- **Fewer output files than expected / only one echo per input series** — fixed; make sure
  your image is current. Output filenames carry an `_e<echo>` suffix precisely because
  `InstanceNumber` restarts per echo (see [Output format](#7-output-format)). An image
  built before that fix silently overwrites echoes, so `Wrote N DICOM files` reports far
  more files than end up on disk. Rebuild both images (see [Getting the image](#2-getting-the-image))
  — and note that rebuilding only `offline-qsm-recon` is not enough, since `mrd2dicom.py`
  lives in the base image.
- **`--input directory not found`** — check the host path in `-v host_path:/input:ro`
  actually exists and is readable; this error means the container-internal `/input` path
  (post-mount) wasn't a directory, usually because the host-side path was wrong or the
  mount itself failed silently.
