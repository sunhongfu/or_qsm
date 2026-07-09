# CPU-only variant of docker/qsm.dockerfile, for fast local testing on machines
# without an NVIDIA GPU (e.g. a laptop). For actual scanner deployment, use
# docker/qsm.dockerfile instead, which is CUDA-enabled for the MaRS GPU.
#
# Build from the python-ismrmrd-server folder:
#   docker build -f docker/qsm-cpu.dockerfile \
#       --build-context iqsm_plus=/Users/uqhsun8/Documents/repos/iQSM_Plus \
#       -t openrecon-qsm:cpu .

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

COPY --from=iqsm_plus . /opt/code/iQSM_Plus
ENV IQSM_PLUS_DIR=/opt/code/iQSM_Plus

COPY qsm.py /opt/code/python-ismrmrd-server/qsm.py

WORKDIR /opt/code/python-ismrmrd-server

CMD [ "python3", "/opt/code/python-ismrmrd-server/main.py", "-v", "-H=0.0.0.0", "-p=9002", "-l=/tmp/python-ismrmrd-server.log", "--defaultConfig=qsm"]
