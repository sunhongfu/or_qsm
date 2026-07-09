# OpenRecon QSM (iQSM+)

A Siemens [Open Recon](https://www.siemens-healthineers.com/magnetic-resonance-imaging/options-and-upgrades/open-recon) app that reconstructs quantitative susceptibility maps (QSM) from multi-echo GRE magnitude/phase images, using [iQSM+](https://github.com/sunhongfu/iQSM_Plus), a deep-learning QSM pipeline. Runs on the scanner as a Docker container, invoked via Siemens' image-to-image (i2i) Open Recon workflow.

**Built on top of [kspaceKelvin/python-ismrmrd-server](https://github.com/kspaceKelvin/python-ismrmrd-server)** (MIT licensed, see [LICENSE](LICENSE)), a reference MRD client/server framework for building modular MRI reconstruction/analysis pipelines. That repo's own README documents the underlying framework in depth (module structure, client/server protocol, generic Docker setup) -- worth reading for background, but not duplicated here.

**Research use only. Not for diagnostic use.**

## Table of Contents
- [How it works](#how-it-works)
- [Repository layout](#repository-layout)
- [Building the Docker image](#building-the-docker-image)
- [Packaging for scanner deployment](#packaging-for-scanner-deployment)
- [UI parameters](#ui-parameters)
- [Local testing](#local-testing)
- [Diagnostics](#diagnostics)
- [Requirements](#requirements)
- [License](#license)

## How it works

1. The scan operator runs a multi-echo 3D GRE sequence with magnitude + phase output, with this app selected on the Open Recon card.
2. Siemens' Emitter functor streams the reconstructed magnitude/phase images to the container over MRD ([qsm.py](qsm.py)'s `process()`).
3. [qsm.py](qsm.py) buffers every image (QSM needs the whole 3D multi-echo volume, not one image at a time -- see the module's own docstring), then:
   - Optionally runs FSL's `bet2` on the first echo to get a brain mask (toggle: [UI parameters](#ui-parameters)).
   - Calls into [iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus) (`run_iqsm_plus()`, cloned locally as a gitignored subfolder of this repo -- see [Building the Docker image](#building-the-docker-image) -- rather than tracked in this repo's own git history) to run the actual deep-learning reconstruction.
   - Quantizes the resulting susceptibility map (ppm) into uint16 DICOM pixel data with a fixed rescale slope/intercept.
4. Both the QSM map (`image_series_index=100`) **and** the original acquisition series are sent back unmodified -- Open Recon only saves/displays images an app explicitly returns, so passing through the originals is what keeps them from being silently discarded.

## Repository layout

- [qsm.py](qsm.py) -- the QSM reconstruction module itself (the `config` this server runs; see `--defaultConfig=qsm` in the Dockerfile `CMD`).
- `iQSM_Plus/` -- **not tracked in this repo** (gitignored); a local clone of [sunhongfu/iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus) that you create yourself before building -- see [Building the Docker image](#building-the-docker-image).
- [vendor/bet2/](vendor/bet2/) -- FSL's `bet2` binary + its ~15 runtime shared libraries, vendored directly (not a full FSL install).
- [docker/qsm.dockerfile](docker/qsm.dockerfile) -- builds the deployable image. `docker/qsm-cuda-conda.dockerfile` is kept as a rollback to the original conda-based build (larger, ~8.7GB vs ~6.3GB) if the slim pip-based one ever misbehaves.
- [docker/build_openrecon_package.py](docker/build_openrecon_package.py) -- packages the built image + `docs.pdf` into the `.zip` Open Recon expects for scanner installation.
- [qsm_json_ui.json](qsm_json_ui.json) -- the Open Recon app manifest: UI parameters, GPU/memory/CPU requirements, versioning. Gets base64-encoded into a Docker image label during packaging.
- [RunQSMRecon.ipynb](RunQSMRecon.ipynb) -- local test/validation workflow: DICOM-to-MRD conversion, running a reconstruction, displaying the QSM map, and a standalone `bet2` smoke test.
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
# server (Docker)" task -- edit its code, restart the task, no rebuild needed, same as
# qsm.py.
git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus

# --platform linux/amd64 is required on Apple Silicon: the base image (python:3.12-slim)
# publishes a native arm64 manifest, so without this flag Docker silently builds for
# arm64 and the CUDA-only torch wheels fail to resolve with a confusing "no matching
# distribution" error. Not needed (but harmless) on a native linux/amd64 machine.
docker build --platform linux/amd64 -f docker/qsm.dockerfile \
    -t openrecon-qsm:prod .
```

Pretrained checkpoints are downloaded from [Hugging Face](https://huggingface.co/sunhongfu/iQSM_Plus) during the build (they're not committed to the iQSM_Plus git repo itself) -- skipped automatically if your local clone already has them (e.g. after running iQSM_Plus's own `run.py --download-checkpoints`).

See `docker/qsm.dockerfile`'s header comment for troubleshooting registry/BuildKit issues.

**Using the devcontainer instead?** Do the `git clone ... iQSM_Plus` step above (and put test
DICOMs under `data/DICOMs_openrecon/`, see [Local testing](#local-testing)) *before* opening the
devcontainer -- it no longer clones/downloads anything for you (see `.devcontainer/devcontainer.json`).

## Packaging for scanner deployment

```bash
python3 docker/build_openrecon_package.py
```

Validates `qsm_json_ui.json` against the Open Recon JSON schema, builds a labeled image (`docker/OpenRecon_qsm.dockerfile`, generated automatically from the JSON config), `docker save`s it, converts the manifest format if the local Docker version exceeds Open Recon's supported maximum, and zips it with `docs.pdf`. Output goes to `~/Desktop/OpenRecon_package/`.

**Versioning**: bump `qsm_json_ui.json`'s `general.version` (and the matching `regulatory_information` fields) before repackaging for any functional change. Commit and push to GitHub whenever the version bumps, so the repo history stays in sync with what's actually been shipped.

## UI parameters

Defined in `qsm_json_ui.json`'s `parameters` array, rendered as the Open Recon card's UI, and delivered back to `qsm.py`'s `process(connection, config, metadata)` as a dict (`config['parameters'][...]`) when the scan runs -- see `server.py`'s `configAdditional` handling for the underlying mechanism.

| id | type | default | purpose |
|---|---|---|---|
| `config` | choice | `qsm` | Which module the server dispatches to. |
| `customconfig` | string | `""` | Override `config` with an arbitrary module name not in the dropdown. |
| `brainextraction` | boolean | `true` | Run `bet2` on the magnitude image before reconstruction, to crop the field of view to brain tissue and suppress non-brain background artifacts. |

Parameter `id`s must match `^[A-Za-z0-9]+$` (no underscores) -- an Open Recon schema constraint.

## Local testing

Put a sample multi-echo GRE DICOM series (magnitude + phase) under `data/DICOMs_openrecon/`
(gitignored -- not part of the repo). [RunQSMRecon.ipynb](RunQSMRecon.ipynb) walks through
converting it, running a reconstruction, and displaying the result. To run the server for it:
- **"Start QSM server (Docker)" task** (Terminal > Run Task) -- runs the actual `openrecon-qsm:prod` image, matching production, including `bet2`. Requires the image already built and Docker Desktop running. Maps to the same port (9020) as the native config, so notebook cells don't need to change either way.

To simulate a specific UI parameter value from `client.py` without a real scanner/Open Recon UI: create a `<config>.json` sidecar file (e.g. `qsm.json`) in the working directory --
```json
{"parameters": {"config": "qsm", "brainextraction": true}}
```
`client.py` automatically finds and sends it when you pass `-c qsm`.

## Diagnostics

`qsm.py` and `inference.py` log the source of every acquisition parameter fed into iQSM+ (`voxel_size_mm`, `b0_dir`, `TE(s)`, `b0` -- including *which* MRD header field or fallback path each came from), plus memory usage (host RSS, cgroup usage vs. limit, GPU allocation) at every major pipeline stage, including a background heartbeat during the deep-learning inference call. Added after a production run was silently OOM-killed by the kernel with no other trace in the log -- see git history for the incident and the voxel-size unit bug it led to finding.

## Requirements

Per `qsm_json_ui.json`'s `reconstruction` section: GPU optional but supported (`min_required_gpu_memory: 8192` MB), `min_required_memory: 16384` MB, `min_count_required_cpu_cores: 4`. CPU-only inference works (verified) but is far slower than GPU.

## License

This repository's own code is MIT licensed (inherited from the upstream framework, see [LICENSE](LICENSE)). `vendor/bet2/` is FSL's Brain Extraction Tool, separately licensed -- see [vendor/bet2/README.md](vendor/bet2/README.md). The iQSM+ model weights are a separate checkout ([sunhongfu/iQSM_Plus](https://github.com/sunhongfu/iQSM_Plus)) with their own license terms, not tracked in this repo's git history.
