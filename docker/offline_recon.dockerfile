# Standalone offline batch QSM/R2* reconstruction -- NOT the Siemens Open Recon scanner
# framework. Builds on top of the already-built openrecon-qsm:prod image (same
# iQSM_Plus/DeepRelaxo/bet2/torch dependencies, see docker/qsm.dockerfile), just with a
# different entrypoint: offline_recon.py, a CLI script that runs a folder of input DICOMs
# through the exact same reconstruction pipeline (dicom2mrd.py -> qsm.py's process_qsm()
# via a local loopback-only MRD server+client -> mrd2dicom.py) with no
# scanner/Injector-Emitter protocol involved.
#
# Build (after building openrecon-qsm:prod via docker/qsm.dockerfile):
#   docker build -f docker/offline_recon.dockerfile -t offline-qsm-recon .
#
# Run:
#   docker run --rm --gpus all \
#       -v /path/to/dicoms:/input:ro \
#       -v /path/to/output:/output \
#       offline-qsm-recon --input /input --output /output --mode masked
#
# --gpus all is optional (CPU fallback works, just far slower -- see readme.md). Drop it
# entirely on a host with no GPU/NVIDIA Container Toolkit.
#
# COPY'd explicitly here (rather than relying on openrecon-qsm:prod's own baked-in copy)
# so this image can be rebuilt on its own when only offline_recon.py changes, without
# needing to rebuild the whole multi-GB base image.
FROM openrecon-qsm:prod

COPY offline_recon.py /opt/code/python-ismrmrd-server/offline_recon.py

ENTRYPOINT ["python3", "/opt/code/python-ismrmrd-server/offline_recon.py"]
