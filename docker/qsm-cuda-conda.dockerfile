# ROLLBACK COPY of the original conda-based build, kept in case qsm.dockerfile
# (now the leaner python:3.12-slim + pip-installed torch variant, as of the
# v1.0.3 image) turns out not to work in practice. To roll back: rebuild using
# THIS file instead of qsm.dockerfile (same build command, just swap -f), and
# re-tag as openrecon-qsm:prod before re-running docker/build_openrecon_package.py.
# See qsm-slim-build*.log / the size comparison in conversation history for why
# this was replaced: it produced an 8.68GB image vs 6.06GB for qsm.dockerfile,
# ~93% of which was a full Anaconda distribution baked into the base image's
# /opt/conda, most of it not actually used by this project.
#
# Open Recon "image to image" app: iQSM+ deep-learning QSM reconstruction.
#
# This extends docker/pytorch/Dockerfile (which already provides a CUDA-enabled
# PyTorch base + compiled ismrmrd/siemens_to_ismrmrd + the python-ismrmrd-server
# code) with the iQSM+ pipeline and the qsm.py module.
#
# Build from the or_qsm repo folder. Step 1 (one-time): clone iQSM_Plus
# (https://github.com/sunhongfu/iQSM_Plus) into this repo as a subfolder (gitignored,
# but included in the Docker build context) -- see qsm.dockerfile's header comment for
# the full rationale:
#
#   git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus
#
# Step 2:
#   docker build -f docker/qsm-cuda-conda.dockerfile \
#       -t openrecon-qsm:prod .
#
# For a CPU-only build/test (e.g. on a machine without an NVIDIA GPU), swap the
# base image below for "python:3.12.0-slim" and add "RUN pip3 install torch"
# instead of relying on the CUDA-enabled PyTorch base image.
#
# Troubleshooting "context deadline exceeded" / stuck on "load metadata for
# docker.io/...": this means BuildKit is trying to re-check the registry for a
# newer manifest, even if the image is already pulled locally -- if your
# Docker Desktop registry proxy (Settings > Resources > Proxies) is flaky,
# `docker pull <image>` can succeed while `docker build` still hangs on the
# same image. First retry normally; if it keeps failing, force the classic
# builder (which checks the local image store first) as a workaround:
#   DOCKER_BUILDKIT=0 docker build -f docker/qsm.dockerfile ...

# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=America/Chicago

RUN  apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev
RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----
FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-runtime AS python-mrd-cuda-devcontainer

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Add dependencies for siemens_to_ismrmrd
RUN apt-get update && apt-get install --no-install-recommends -y libxslt1.1 libhdf5-103 libboost-program-options1.74.0 libpugixml1v5 git dos2unix nano
RUN mkdir -p /opt/code

# Python MRD library
RUN pip3 install h5py==3.10.0 ismrmrd==1.14.1

RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd-python-tools.git && \
    cd /opt/code/ismrmrd-python-tools && \
    pip3 install --no-cache-dir .

# matplotlib is used by rgb.py and provides various visualization tools including colormaps
# pydicom is used by dicom2mrd.py to parse DICOM data
# nibabel/scipy are used by the iQSM+ pipeline
#
# matplotlib and h5py versions are left unpinned (rather than ==3.8.2 and
# ismrmrd==1.14.1's h5py==3.10.0 pin from docker/Dockerfile) so pip's resolver
# can pick numpy-2-compatible builds matching whatever numpy version the base
# pytorch/pytorch image's torch install requires -- pinned numpy<2-era
# versions of either would break the whole server at startup (server.py
# unconditionally imports invertcontrast.py -> matplotlib, and ismrmrd -> h5py).
RUN pip3 install --no-cache-dir --upgrade matplotlib h5py pydicom==3.0.1 nibabel scipy

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# ----- 3. Copy deployed code, add the iQSM+ pipeline, configure as an Open Recon app -----
# (Not split into a separate stage -- see qsm.dockerfile's equivalent step for why.)
FROM python-mrd-cuda-devcontainer AS openrecon-qsm

RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

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

# bet2 (brain extraction), vendored directly in the repo -- see vendor/bet2/README.md
COPY vendor/bet2 /opt/bet2
ENV BET2_DIR=/opt/bet2
ENV FSLOUTPUTTYPE=NIFTI_GZ

# GPU driver mapping, per Open Recon's GPU container guidance
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "/bin/bash", "-c", "/usr/sbin/ldconfig && exec python3 /opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=qsm" ]
