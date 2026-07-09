# CPU-only variant of docker/qsm.dockerfile, for fast local testing on machines
# without an NVIDIA GPU (e.g. a laptop). For actual scanner deployment, use
# docker/qsm.dockerfile instead, which is CUDA-enabled for the MaRS GPU.
#
# Build from the or_qsm repo folder. Step 1 (one-time): clone iQSM_Plus into this repo
# as a subfolder (gitignored, but included in the Docker build context) -- see
# qsm.dockerfile's header comment for the full rationale:
#   git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus
# Step 2:
#   docker build -f docker/qsm-cpu.dockerfile -t openrecon-qsm:cpu .

FROM kspacekelvin/fire-python:latest AS openrecon-qsm-cpu

# Upgrade matplotlib and h5py alongside torch so pip's resolver picks
# numpy-2-compatible versions of everything -- torch pulls in numpy>=2, which
# is incompatible with the numpy<2-era matplotlib/h5py pinned in the base
# image (h5py fails with a numpy dtype ABI error at import time otherwise).
# Since server.py unconditionally imports invertcontrast.py (which imports
# matplotlib) and ismrmrd (which imports h5py) at startup, a broken version
# of either means the whole server won't run.
RUN pip3 install --no-cache-dir --upgrade torch --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install --no-cache-dir --upgrade nibabel scipy matplotlib h5py

# Copied from the local subfolder (see header comment) rather than requiring
# --build-context. This base image doesn't COPY the whole repo in (unlike
# qsm.dockerfile), so iQSM_Plus needs its own explicit COPY here -- kept at the same
# nested-under-python-ismrmrd-server path as the other variants for consistency, even
# though this particular Dockerfile isn't used by .vscode/tasks.json's live-edit mount.
COPY iQSM_Plus /opt/code/python-ismrmrd-server/iQSM_Plus
ENV IQSM_PLUS_DIR=/opt/code/python-ismrmrd-server/iQSM_Plus

# In case iQSM_Plus was never cloned (README step skipped) -- the COPY above would
# already fail outright on a missing source path, but with a much less actionable Docker
# error, so check explicitly here too for a clear message pointing back to the README.
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

COPY qsm.py /opt/code/python-ismrmrd-server/qsm.py

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=qsm"]
