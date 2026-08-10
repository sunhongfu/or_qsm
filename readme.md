# OpenRecon QSM (iQSM+)

A Siemens [Open Recon](https://www.siemens-healthineers.com/magnetic-resonance-imaging/options-and-upgrades/open-recon) app that reconstructs quantitative susceptibility maps (QSM) and R2* maps from multi-echo GRE magnitude/phase images, using [iQSM+](https://github.com/sunhongfu/iQSM_Plus) (QSM) and [DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo) (R2*), both deep-learning pipelines. Runs on the scanner as a Docker container, invoked via Siemens' image-to-image (i2i) Open Recon workflow.

**Built on top of [kspaceKelvin/python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server)** (MIT licensed, see [LICENSE](LICENSE)), a reference MRD client/server framework for building modular MRI reconstruction/analysis pipelines. That repo's own README documents the underlying framework in depth (module structure, client/server protocol, generic Docker setup) -- worth reading for background, but not duplicated here.

**Research use only. Not for diagnostic use.**

## Table of Contents
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Building the Docker image](#building-the-docker-image)
- [Packaging for scanner deployment](#packaging-for-scanner-deployment)
- [UI parameters](#ui-parameters)
- [Local testing](#local-testing)
- [Offline batch reconstruction](#offline-batch-reconstruction)
- [Diagnostics](#diagnostics)
- [Requirements](#requirements)
- [License](#license)

## How it works

1. The scan operator runs a multi-echo 3D GRE sequence with magnitude + phase output, with this app selected on the Open Recon card.
2. Siemens' Emitter functor streams the reconstructed magnitude/phase images to the container over MRD ([qsm.py](qsm.py)'s `process()`).
3. [qsm.py](qsm.py) buffers every image (QSM needs the whole 3D multi-echo volume, not one image at a time -- see the module's own docstring), then:
   - Per the "Reconstruction Mode" UI parameter (see [UI parameters](#ui-parameters)), optionally runs FSL's `bet2` on the first echo to get a brain mask ("Brain-masked", the default) or skips it ("Whole-head").
   - If "Reconstruct QSM" is enabled (default: on), calls into [iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus) (`run_iqsm_plus()`, cloned locally as a gitignored subfolder of this repo -- see [Building the Docker image](#building-the-docker-image) -- rather than tracked in this repo's own git history) to run the actual deep-learning QSM reconstruction.
   - If "Reconstruct R2* Mapping" is enabled (default: **off** -- noticeably slower than QSM alone; enable via retro-recon afterward if needed) and the acquisition has at least 2 echoes, also calls into [DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo) (`estimate_r2s()` + `denoise_r2s_map()`, same gitignored-subfolder pattern) to run its two-stage R2* mapping, reusing the same magnitude NIfTIs and bet2 mask already prepared above -- no separate brain extraction, and independent of whether QSM itself ran. Single-echo acquisitions can't support R2* mapping (needs at least 2 points to fit a decay curve) regardless of the toggle.
   - Quantizes the resulting maps (ppm for QSM, s⁻¹ for R2*) into uint16 DICOM pixel data with fixed rescale slopes/intercepts.
4. Whichever of QSM (`image_series_index=100` brain-masked / `101` whole-head) and R2* (`102` brain-masked / `103` whole-head) were enabled -- **and** the original acquisition series, row-flip-corrected (see below) -- are sent back -- Open Recon only saves/displays images an app explicitly returns, so passing through the originals is what keeps them from being silently discarded. This holds even if reconstruction itself fails partway through: the original images are always returned, degrading to "no QSM/R2* this exam" rather than "no images at all" (see `process()`'s exception handling in [qsm.py](qsm.py)). A single exam only ever produces one mask variant's output; re-run with a different "Reconstruction Mode" / toggle selection (via retro-recon, if supported) to get another combination.

**Pass-through orientation fix:** on real scanner hardware, the original magnitude/phase images' pixel data was found to be row-flipped relative to their own `ImageOrientationPatient`/`ImagePositionPatient` tags -- confirmed against real exports, where the QSM/R2* output (built from the same acquisition geometry) rendered correctly while the passed-through originals didn't. `qsm.py`'s `_fix_passthrough_row_flip()` corrects this (pixel data + `ImagePositionPatient` only; every other header field and all DICOM metadata is left untouched) before the originals are sent back -- see that function's docstring for the full investigation (several code-level explanations were ruled out; the true upstream root cause, likely in the scanner's own Emitter/Injector, remains unconfirmed).

## Repository layout

- [qsm.py](qsm.py) -- the QSM/R2* reconstruction module itself (the `config` this server runs; see `--defaultConfig=qsm` in the Dockerfile `CMD`).
- `iQSM_Plus/` -- **not tracked in this repo** (gitignored); a local clone of [sunhongfu/iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus) that you create yourself before building -- see [Building the Docker image](#building-the-docker-image).
- `DeepRelaxo/` -- **not tracked in this repo** (gitignored); a local clone of [sunhongfu/DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo) (R2* mapping), same as `iQSM_Plus/` above.
- [vendor/bet2/](vendor/bet2/) -- FSL's `bet2` binary + its ~15 runtime shared libraries, vendored directly (not a full FSL install).
- [docker/qsm.dockerfile](docker/qsm.dockerfile) -- builds the deployable image. `docker/qsm-cuda-conda.dockerfile` is kept as a rollback to the original conda-based build (larger, ~8.7GB vs ~6.3GB) if the slim pip-based one ever misbehaves.
- [docker/build_openrecon_package.py](docker/build_openrecon_package.py) -- packages the built image + `docs.pdf` into the `.zip` Open Recon expects for scanner installation.
- [qsm_json_ui.json](qsm_json_ui.json) -- the Open Recon app manifest: UI parameters, GPU/memory/CPU requirements, versioning. Gets base64-encoded into a Docker image label during packaging.
- [RunQSMRecon.ipynb](RunQSMRecon.ipynb) -- local test/validation workflow: DICOM-to-MRD conversion, running a reconstruction, converting the result to DICOM, and displaying the QSM map.
- [offline_recon.py](offline_recon.py) + [docker/offline_recon.dockerfile](docker/offline_recon.dockerfile) -- a separate, standalone Docker image for offline/batch use (research, testing against archived DICOMs) -- **not** the Siemens Open Recon scanner framework. See [Offline batch reconstruction](#offline-batch-reconstruction).
- [docs/scanner-deployment-guide.md](docs/scanner-deployment-guide.md) -- scanner-side installation notes.

## Building the Docker image

Every command below runs from wherever the first one leaves you (the repo root) -- no further
`cd` needed at any step, including packaging and local testing further down this page.

```bash
git clone https://github.com/sunhongfu/or_qsm.git
cd or_qsm

# iQSM_Plus (model/inference code) as a subfolder -- gitignored, so this doesn't affect
# this repo's own git history, but it IS included in the Docker build context, so the
# build below picks it up automatically (no --build-context needed). Also means
# iQSM_Plus/ is covered by the live-edit bind-mount in .vscode/tasks.json's "Start QSM
# server (Docker)" task (and the devcontainer) -- edit its code, restart, no rebuild
# needed, same as qsm.py.
git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus

# Pretrained checkpoints are hosted on Hugging Face, not committed to the iQSM_Plus git
# repo. Downloaded here with plain urllib (stdlib only -- no pip installs needed on a
# fresh clone) rather than iQSM_Plus's own `run.py --download-checkpoints` (which needs
# torch/nibabel/etc. already installed). The Docker build below would also fetch them,
# but only into the image itself, which doesn't help the devcontainer or the "Start QSM
# server (Docker)" task, since their live bind-mount overwrites the image's checkpoints/
# with this (empty) local one -- so do it here once and every workflow has them.
mkdir -p iQSM_Plus/checkpoints
python3 -c "
import os, urllib.request
base = 'https://huggingface.co/sunhongfu/iQSM_Plus/resolve/main'
for name in ['iQSM_plus.pth', 'LoTLayer_chi.pth']:
    local = f'iQSM_Plus/checkpoints/{name}'
    if not os.path.exists(local):
        urllib.request.urlretrieve(f'{base}/{name}', local)
"

# Same one-time step for DeepRelaxo (R2* mapping) -- same gitignored-subfolder pattern,
# same plain-urllib download (its own `run_deeprelaxo_pipeline.py --download-checkpoints`
# needs huggingface_hub/pyyaml already installed, same reasoning as iQSM_Plus above).
git clone https://github.com/sunhongfu/DeepRelaxo.git DeepRelaxo
mkdir -p DeepRelaxo/checkpoints
python3 -c "
import os, urllib.request
base = 'https://huggingface.co/sunhongfu/DeepRelaxo/resolve/main'
for name in ['transformer_mlp_epoch_80.pth', 'unet3d_epoch_140.pth']:
    local = f'DeepRelaxo/checkpoints/{name}'
    if not os.path.exists(local):
        urllib.request.urlretrieve(f'{base}/{name}', local)
"

# --platform linux/amd64 is required on Apple Silicon: the base image (python:3.12-slim)
# publishes a native arm64 manifest, so without this flag Docker silently builds for
# arm64 and the CUDA-only torch wheels fail to resolve with a confusing "no matching
# distribution" error. Not needed (but harmless) on a native linux/amd64 machine.
docker build --platform linux/amd64 -f docker/qsm.dockerfile \
    -t openrecon-qsm:prod .
```

See `docker/qsm.dockerfile`'s header comment for troubleshooting registry/BuildKit issues.

**Using the devcontainer instead?** Do the `git clone ... iQSM_Plus` and checkpoint-download steps
above (and put test DICOMs under `data/dicoms/`, see [Local testing](#local-testing))
*before* opening the devcontainer -- it no longer clones/downloads anything for you (see
`.devcontainer/devcontainer.json`).

## Packaging for scanner deployment

Requires the `jsonschema` and `packaging` packages (schema validation and Docker version
comparison, respectively) -- `pip install jsonschema packaging` if you don't already have them.

```bash
python3 docker/build_openrecon_package.py
```

Validates `qsm_json_ui.json` against the Open Recon JSON schema, builds a labeled image (`docker/OpenRecon_qsm.dockerfile`, generated automatically from the JSON config), `docker save`s it, converts the manifest format if the local Docker version exceeds Open Recon's supported maximum, and zips it with `docs.pdf`. Output goes to `OpenRecon_package/` in the repo root (gitignored).

**Versioning**: bump `qsm_json_ui.json`'s `general.version` (and the matching `regulatory_information` fields) before repackaging for any functional change. Commit and push to GitHub whenever the version bumps, so the repo history stays in sync with what's actually been shipped.

## UI parameters

Defined in `qsm_json_ui.json`'s `parameters` array, rendered as the Open Recon card's UI, and delivered back to `qsm.py`'s `process(connection, config, metadata)` as a dict (`config['parameters'][...]`) when the scan runs -- see `server.py`'s `configAdditional` handling for the underlying mechanism.

| id | type | default | purpose |
|---|---|---|---|
| `config` | choice | `qsm` | Which module the server dispatches to. |
| `customconfig` | string | `""` | Override `config` with an arbitrary module name not in the dropdown. |
| `reconmode` | choice (`masked`/`wholehead`) | `masked` | Which mask variant to reconstruct with: brain-masked (`bet2`) or whole-head. Applies to whichever of QSM/R2* below is enabled -- a single exam only ever produces one variant. |
| `qsmenabled` | boolean | `true` | Run iQSM+ QSM reconstruction. |
| `r2smapping` | boolean | `false` | Also run DeepRelaxo R2* mapping. Off by default -- noticeably slower than QSM alone; enable via retro-recon afterward if needed. Requires 2+ echoes regardless. |

Parameter `id`s must match `^[A-Za-z0-9]+$` (no underscores) -- an Open Recon schema constraint.

## Local testing

**Requires VS Code** (with the Dev Containers, Python, and Jupyter extensions).

Put a sample multi-echo GRE DICOM series (magnitude + phase) under `data/dicoms/`
(gitignored -- not part of the repo). [RunQSMRecon.ipynb](RunQSMRecon.ipynb) walks through
converting it, running a reconstruction, and displaying the result.

Open this repo in the devcontainer ("Dev Containers: Reopen in Container"), then use the
**"Start QSM server"** launch config (Run and Debug tab) to start the server on port 9020 --
`bet2` and GPU both work directly inside it, on every host, since it forces
`--platform linux/amd64` and live-mounts this repo over the image's own code (see
[.devcontainer/devcontainer.json](.devcontainer/devcontainer.json)'s own comments for the full
rationale and its one trade-off: on Apple Silicon, the *entire* devcontainer runs under amd64
emulation, not just `bet2` calls -- slower pip installs, Python startup, every terminal command).

There's also a **"Start QSM server (Docker)" task** in `.vscode/tasks.json` that runs the
already-built, tagged `openrecon-qsm:prod` image directly (`docker run`, no devcontainer
rebuild/reopen needed) with the same live bind-mount -- handy for a quick server without
switching VS Code's environment, or for testing the *exact* image that's about to be
[packaged](#packaging-for-scanner-deployment).

To simulate a specific UI parameter value from `client.py` without a real scanner/Open Recon UI: create a `<config>.json` sidecar file (e.g. `qsm.json`) in the working directory --
```json
{"parameters": {"config": "qsm", "reconmode": "masked", "qsmenabled": true, "r2smapping": false}}
```
`client.py` automatically finds and sends it when you pass `-c qsm`.

## Offline batch reconstruction

For research/offline use against archived DICOMs -- **not** the Siemens Open Recon scanner
framework, no Injector/Emitter protocol involved. [docker/offline_recon.dockerfile](docker/offline_recon.dockerfile)
builds a separate image on top of `openrecon-qsm:prod` (same iQSM_Plus/DeepRelaxo/bet2/torch
dependencies) with a different entrypoint: [offline_recon.py](offline_recon.py), a CLI script
that takes a folder of DICOMs in and writes QSM/R2* DICOMs out, with no server/client dance
required on your end -- internally it orchestrates the exact same `dicom2mrd.py` ->
[qsm.py](qsm.py)'s `process_qsm()` (via a local, loopback-only MRD server+client pair) ->
`mrd2dicom.py` pipeline already used everywhere else in this repo, rather than reimplementing
anything.

```bash
# after building openrecon-qsm:prod (see "Building the Docker image" above):
docker build -f docker/offline_recon.dockerfile -t offline-qsm-recon .

docker run --rm --gpus all \
    -v /path/to/dicoms:/input:ro \
    -v /path/to/output:/output \
    offline-qsm-recon --input /input --output /output --mode masked
```

`--mode` is `masked` (default) or `wholehead`, matching the scanner app's "Reconstruction
Mode" UI parameter -- see [UI parameters](#ui-parameters). `--gpus all` is optional (CPU
fallback works, just far slower); drop it entirely on a host with no GPU/NVIDIA Container
Toolkit. Run `docker run --rm offline-qsm-recon --help` for all options.

See [docs/offline-recon-guide.md](docs/offline-recon-guide.md) for the full usage guide --
input/output format, DICOM series numbering, expected timing, and troubleshooting.

## Diagnostics

`qsm.py` and `inference.py` log the source of every acquisition parameter fed into iQSM+ (`voxel_size_mm`, `b0_dir`, `TE(s)`, `b0` -- including *which* MRD header field or fallback path each came from), plus memory usage (host RSS, cgroup usage vs. limit, GPU allocation) at every major pipeline stage, including a background heartbeat during the deep-learning inference call. Added after a production run was silently OOM-killed by the kernel with no other trace in the log -- see git history for the incident and the voxel-size unit bug it led to finding.

## Requirements

Per `qsm_json_ui.json`'s `reconstruction` section: GPU optional but supported (`min_required_gpu_memory: 8192` MB), `min_required_memory: 16384` MB, `min_count_required_cpu_cores: 4`. CPU-only inference works (verified) but is far slower than GPU.

## License

This repository's own code is MIT licensed (inherited from the upstream framework, see [LICENSE](LICENSE)). `vendor/bet2/` is FSL's Brain Extraction Tool, separately licensed -- see [vendor/bet2/README.md](vendor/bet2/README.md). The iQSM+ and DeepRelaxo model weights are separate checkouts ([sunhongfu/iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus), [sunhongfu/DeepRelaxo](https://github.com/sunhongfu/DeepRelaxo)) with their own license terms, not tracked in this repo's git history.
