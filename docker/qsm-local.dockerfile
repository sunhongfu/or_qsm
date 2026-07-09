# Variant of docker/qsm-cpu.dockerfile that builds from the already-cached local
# devcontainer image instead of pulling kspacekelvin/fire-python fresh from
# Docker Hub. Useful when Docker Hub registry pulls are unavailable/blocked but
# a local devcontainer image (e.g. built by VS Code's Dev Containers extension)
# already has ismrmrd/matplotlib/pydicom installed.
#
# Build from the or_qsm repo folder. Step 1 (one-time): clone iQSM_Plus into this repo
# as a subfolder (gitignored, but included in the Docker build context) -- see
# qsm.dockerfile's header comment for the full rationale:
#   git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus
# Step 2:
#   docker build -f docker/qsm-local.dockerfile \
#       --build-arg BASE_IMAGE=<local-devcontainer-image-tag> \
#       -t openrecon-qsm:local .

ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS openrecon-qsm-local

# Upgrade matplotlib and h5py alongside torch so pip's resolver picks
# numpy-2-compatible versions of everything -- torch pulls in numpy>=2, which
# is incompatible with the numpy<2-era matplotlib==3.8.2 and h5py==3.10.0
# pinned in the base image (h5py fails with a numpy dtype ABI error at import
# time otherwise). Since server.py unconditionally imports invertcontrast.py
# (which imports matplotlib) and ismrmrd (which imports h5py) at startup, a
# broken version of either means the whole server won't run.
RUN pip3 install --no-cache-dir --upgrade torch nibabel scipy matplotlib h5py

# The devcontainer image only builds through the "python-mrd-devcontainer" stage
# (tools installed, no source code baked in -- VS Code mounts the repo live
# instead), so the full repo needs to be copied in here.
RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

# Already arrived via COPY . above (subfolder of this repo) -- see qsm.dockerfile's
# equivalent step for the full rationale on why it's nested here, not a sibling path.
ENV IQSM_PLUS_DIR=/opt/code/python-ismrmrd-server/iQSM_Plus

# In case iQSM_Plus was never cloned (README step skipped) -- see qsm.dockerfile's
# equivalent step for the full rationale.
RUN test -f "$IQSM_PLUS_DIR/inference.py" || \
    { echo "ERROR: iQSM_Plus not found at $IQSM_PLUS_DIR -- see readme.md's 'Building the" \
           "Docker image' section (git clone iQSM_Plus into this repo first)." >&2; \
      exit 1; }

# Checkpoints are expected to already be present in the local iQSM_Plus/ clone (same
# prerequisite as its code) -- see qsm.dockerfile's equivalent step for the full
# rationale. Fail loudly here rather than produce an image that only errors at runtime.
RUN test -f "$IQSM_PLUS_DIR/checkpoints/iQSM_plus.pth" && \
    test -f "$IQSM_PLUS_DIR/checkpoints/LoTLayer_chi.pth" || \
    { echo "ERROR: iQSM_Plus checkpoints not found under $IQSM_PLUS_DIR/checkpoints -- see" \
           "readme.md's 'Building the Docker image' section (download them before building)." >&2; \
      exit 1; }

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=qsm"]
