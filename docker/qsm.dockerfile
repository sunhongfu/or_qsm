# Open Recon "image to image" app: iQSM+ deep-learning QSM reconstruction.
#
# As of the v1.0.3 image, this builds on a plain python:3.12-slim base and
# installs GPU-enabled torch via pip, instead of the ~8.68GB
# pytorch/pytorch:*-cuda11.8-cudnn8-runtime conda-based image previously used
# here -- ~93% of that image's size was a full Anaconda distribution baked
# into /opt/conda, most of it unused by this project. PyTorch's official
# cu118 wheels bundle their own CUDA/cuDNN/cuBLAS runtime libraries as pip
# dependencies (nvidia-cudnn-cu11, nvidia-cublas-cu11, etc.) -- they only need
# the host's *driver* (libcuda.so) at runtime, which OpenRecon's nvidia
# container runtime already injects (confirmed from a real production
# container inspect: HostConfig.Runtime="nvidia", NVIDIA_VISIBLE_DEVICES=all).
# Result: 6.06GB vs 8.68GB for an equivalent build.
#
# Rollback: the previous conda-based version of this file is preserved as
# docker/qsm-cuda-conda.dockerfile -- if this leaner build doesn't work in
# practice (e.g. the pip-installed cu118 wheels behave differently from
# conda's under OpenRecon's actual nvidia runtime), rebuild/repackage with
# that file instead (same commands, just swap -f) and nothing else needs to
# change.
#
# Build from the or_qsm repo folder. Step 1 (one-time): clone iQSM_Plus
# (https://github.com/sunhongfu/iQSM_Plus) directly into this repo as a subfolder --
# it's gitignored, so this doesn't affect or_qsm's own git history, but it IS included
# in the Docker build context, so stage 3's `COPY .` picks it up automatically (no
# --build-context needed):
#
#   git clone https://github.com/sunhongfu/iQSM_Plus.git iQSM_Plus
#
# This also means `iQSM_Plus/` is covered by the live-edit bind-mount in
# .vscode/tasks.json's "Start QSM server (Docker)" task -- edit its code, restart the
# task, no rebuild needed, same as qsm.py.
#
# Step 2: build. --platform linux/amd64 is required explicitly on Apple Silicon hosts:
# python:3.12-slim (unlike the old CUDA base) publishes a native arm64 manifest, so
# without this flag Docker silently builds for arm64 and the CUDA-only torch wheels
# fail to resolve ("no matching distribution") with no obvious reason why:
#
#   docker build --platform linux/amd64 -f docker/qsm.dockerfile \
#       -t openrecon-qsm:prod .
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
FROM python:3.12.0-slim AS mrd_converter
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
FROM python:3.12.0-slim AS python-mrd-cuda-devcontainer

LABEL org.opencontainers.image.description="Python MRD Image Reconstruction and Analysis Server"

# Match the env vars pytorch/pytorch's own CUDA images set (confirmed from a
# real production container inspect) -- the driver library (libcuda.so) is
# mounted into these paths by OpenRecon's nvidia container runtime at
# container start, not baked into the image itself.
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV PYTORCH_VERSION=2.3.0

# GPU-enabled torch, installed via pip instead of relying on a CUDA/conda base
# image. cu118 wheels bundle their own CUDA/cuDNN/cuBLAS runtime libraries.
# Installed before any other numpy-dependent package (h5py, matplotlib, scipy,
# nibabel) below so pip's resolver picks numpy-2-compatible builds of all of
# them from the start -- same ordering rationale as before, where torch (and
# its numpy>=2 requirement) needs to be present before those pip installs run.
RUN pip3 install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

# Add dependencies for siemens_to_ismrmrd. gcc/pkg-config/libhdf5-dev are kept
# (as in docker/Dockerfile) in case h5py needs to build from source on a
# platform without a prebuilt wheel.
RUN apt-get update && apt-get install --no-install-recommends -y libxslt1.1 libhdf5-103 libhdf5-dev pkg-config gcc libboost-program-options1.74.0 libpugixml1v5 git dos2unix nano
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
# Unpinned so pip's resolver picks numpy-2-compatible builds matching the
# torch install above (torch pulls in numpy>=2, incompatible with the
# numpy<2-era h5py==3.10.0 pinned just above -- upgrading afterward fixes it).
RUN pip3 install --no-cache-dir --upgrade matplotlib h5py pydicom==3.0.1 nibabel scipy

# Cleanup files not required after installation
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /root/.cache/pip

# ----- 3. Copy deployed code into the devcontainer for deployment -----
FROM python-mrd-cuda-devcontainer AS python-mrd-runtime

RUN mkdir -p /opt/code/python-ismrmrd-server
COPY . /opt/code/python-ismrmrd-server

RUN find /opt/code/python-ismrmrd-server -name "*.sh" | xargs dos2unix
RUN find /opt/code/python-ismrmrd-server -name "*.sh" -exec chmod +x {} \;

# ----- 4. Add the iQSM+ pipeline and configure this as an Open Recon app -----
FROM python-mrd-runtime AS openrecon-qsm

# iQSM_Plus's code already arrived via stage 3's `COPY . /opt/code/python-ismrmrd-server`
# (it's a subfolder of this repo -- see the header comment for the one-time local clone
# step). Nested under python-ismrmrd-server's own path (rather than a separate
# /opt/code/iQSM_Plus, as in earlier versions of this file) specifically so it's also
# covered by .vscode/tasks.json's live-edit bind-mount, which replaces the whole
# /opt/code/python-ismrmrd-server directory -- a sibling path outside that mount would
# vanish the moment that task runs.
ENV IQSM_PLUS_DIR=/opt/code/python-ismrmrd-server/iQSM_Plus

# Pretrained model checkpoints are hosted on Hugging Face
# (https://huggingface.co/sunhongfu/iQSM_Plus), not part of the git repo -- skip
# downloading if the local clone already has them (e.g. after running iQSM_Plus's own
# `run.py --download-checkpoints`), otherwise fetch via plain urllib (already in the
# Python stdlib, no huggingface_hub dependency needed) mirroring that same script.
RUN mkdir -p "$IQSM_PLUS_DIR/checkpoints" && \
    python3 -c "import os, urllib.request; base = 'https://huggingface.co/sunhongfu/iQSM_Plus/resolve/main'; ckpt_dir = os.environ['IQSM_PLUS_DIR'] + '/checkpoints'; [urllib.request.urlretrieve(f'{base}/{n}', f'{ckpt_dir}/{n}') for n in ['iQSM_plus.pth', 'LoTLayer_chi.pth'] if not os.path.exists(f'{ckpt_dir}/{n}')]"

# bet2 (brain extraction), vendored directly in the repo at vendor/bet2/ (bin + its ~15
# FSL-specific runtime libs, ~118MB) rather than extracted from a Docker Hub image at
# build time -- zero dependency on brainlife/fsl continuing to exist (community-maintained,
# not an official image; Docker Hub tags/images are not guaranteed permanent). See
# vendor/bet2/README.md for provenance, license, and re-extraction instructions.
# FSLOUTPUTTYPE is required by bet2 at runtime (errors out without it); LD_LIBRARY_PATH is
# set per-call in qsm.py rather than globally here, to avoid leaking into torch's own
# library resolution.
COPY vendor/bet2 /opt/bet2
ENV BET2_DIR=/opt/bet2
ENV FSLOUTPUTTYPE=NIFTI_GZ

# GPU driver mapping, per Open Recon's GPU container guidance
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "/bin/bash", "-c", "/usr/sbin/ldconfig && exec python3 /opt/code/python-ismrmrd-server/main.py -v -H=0.0.0.0 -p=9002 -l=/tmp/python-ismrmrd-server.log --defaultConfig=qsm" ]
