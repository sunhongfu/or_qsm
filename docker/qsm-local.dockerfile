# Variant of docker/qsm-cpu.dockerfile that builds from the already-cached local
# devcontainer image instead of pulling kspacekelvin/fire-python fresh from
# Docker Hub. Useful when Docker Hub registry pulls are unavailable/blocked but
# a local devcontainer image (e.g. built by VS Code's Dev Containers extension)
# already has ismrmrd/matplotlib/pydicom installed.
#
# Build from the python-ismrmrd-server folder:
#   docker build -f docker/qsm-local.dockerfile \
#       --build-arg BASE_IMAGE=<local-devcontainer-image-tag> \
#       --build-context iqsm_plus=/Users/uqhsun8/Documents/repos/iQSM_Plus \
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

COPY --from=iqsm_plus . /opt/code/iQSM_Plus
ENV IQSM_PLUS_DIR=/opt/code/iQSM_Plus

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=qsm"]
