"""
Open Recon "image to image" module that runs the iQSM+ deep-learning QSM
pipeline (https://github.com/sunhongfu/iQSM_Plus) on incoming multi-echo
GRE magnitude+phase images.

Unlike the simpler example modules (invertcontrast.py, i2i.py), QSM needs the
*entire* 3D multi-echo volume (all slices, all echoes, magnitude and phase)
before it can run -- there is no way to process images one at a time as they
stream in. So this module buffers every incoming image and only runs the
reconstruction once the connection closes (see the "Streaming/order" note in
process(), below).
"""

import ismrmrd
import os
import sys
import logging
import traceback
import threading
import subprocess
import numpy as np
import nibabel as nib
import mrdhelper
import constants
from time import perf_counter

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# The iQSM+ pipeline (inference.py, models/, checkpoints/) is kept as its own
# repo/checkout, cloned as a gitignored subfolder of this repo rather than tracked in
# it (see readme.md's "Building the Docker image" section) -- so try IQSM_PLUS_DIR
# first (if set, e.g. by the devcontainer's containerEnv), then fall back to where it
# ends up in the Docker image / when running from a plain venv on the host, rather than
# requiring the launch config/environment to be edited per-context.
_IQSM_PLUS_CANDIDATES = [
    os.environ.get("IQSM_PLUS_DIR"),
    "/opt/code/python-ismrmrd-server/iQSM_Plus",        # baked into the Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "iQSM_Plus"),  # local checkout, native/no-docker
]
IQSM_PLUS_DIR = next((p for p in _IQSM_PLUS_CANDIDATES if p and os.path.isdir(p)), None)
if IQSM_PLUS_DIR and IQSM_PLUS_DIR not in sys.path:
    sys.path.insert(0, IQSM_PLUS_DIR)

# bet2 (FSL's Brain Extraction Tool), vendored directly in the repo at vendor/bet2/ (bin +
# its ~15 FSL-specific runtime shared libraries, not a full FSL install -- see
# vendor/bet2/README.md for provenance/license). Same multi-candidate resolution pattern as
# IQSM_PLUS_DIR: baked into the Docker image at /opt/bet2 (see docker/qsm.dockerfile), or
# the repo's own vendor/bet2/ for local dev/notebook testing (see RunQSMRecon.ipynb section 4).
_BET2_CANDIDATES = [
    os.environ.get("BET2_DIR"),
    "/opt/bet2",                                                            # baked into the Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "bet2"),  # local dev/notebook testing
]
BET2_DIR = next((p for p in _BET2_CANDIDATES
                 if p and os.path.isfile(os.path.join(p, "bin", "bet2"))), None)


# ----------------------------------------------------------------------------
# Diagnostics: which device iQSM+ inference actually runs on, and memory usage
# over time. Added after a run was silently OOM-killed (exit 137) by the
# kernel mid-inference -- a hard kernel kill gives the process no chance to
# log or raise anything, so the log just goes dark. These helpers make sure
# *something* gets logged (and flushed to the OpenRecon container log, which
# is what survives the kill) right up until the moment it dies, and make it
# possible to tell from the log alone whether inference used the GPU or fell
# back to (slower, much more host-RAM-hungry) CPU execution.
# ----------------------------------------------------------------------------

def _log_device_info():
    """Log whether CUDA/GPU is visible to this process. A silent CPU fallback
    (e.g. because the installed torch build has no CUDA support, or the driver/
    toolkit versions don't match) looks identical to a working GPU run except
    for using far more host RAM and being far slower -- this makes it visible
    in the log instead of having to guess after the fact."""
    logging.info("NVIDIA_VISIBLE_DEVICES=%s CUDA_VISIBLE_DEVICES=%s",
                 os.environ.get("NVIDIA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
    try:
        import torch
    except ImportError as e:
        logging.warning("Could not import torch to check GPU availability: %s", e)
        return

    cudaAvailable = torch.cuda.is_available()
    logging.info("torch %s -- torch.cuda.is_available() = %s", torch.__version__, cudaAvailable)

    if cudaAvailable:
        try:
            deviceCount = torch.cuda.device_count()
            for i in range(deviceCount):
                props = torch.cuda.get_device_properties(i)
                logging.info("  GPU %d: %s, %.1f GB total memory, capability %d.%d",
                             i, props.name, props.total_memory / (1024 ** 3), props.major, props.minor)
        except Exception:
            logging.warning("Could not enumerate CUDA devices:\n%s", traceback.format_exc())
    else:
        logging.warning("CUDA is NOT available in this process -- iQSM+ inference will run on "
                         "CPU, which is far slower and uses substantially more host RAM for a "
                         "volume this size than the equivalent GPU run.")


def _cgroup_memory_usage_bytes():
    """Current/limit memory usage as enforced by Docker's cgroup, in bytes, or
    (None, None) if unreadable. This is the number OpenRecon's container memory
    limit (and the kernel OOM killer) actually acts on -- it can differ from
    plain process RSS (page cache, shared libs, etc.), so it's the most direct
    signal for "how close are we to being OOM-killed right now"."""
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            usage = int(f.read().strip())
        with open("/sys/fs/cgroup/memory.max") as f:
            raw = f.read().strip()
            limit = None if raw == "max" else int(raw)
        return usage, limit
    except OSError:
        pass
    # cgroup v1
    try:
        with open("/sys/fs/cgroup/memory/memory.usage_in_bytes") as f:
            usage = int(f.read().strip())
        with open("/sys/fs/cgroup/memory/memory.limit_in_bytes") as f:
            limit = int(f.read().strip())
        return usage, limit
    except OSError:
        return None, None


def _log_memory_usage(tag):
    """Log host RSS, cgroup memory usage/limit, and GPU memory, tagged with a
    short label identifying the pipeline stage. Cheap enough to call liberally.
    Intended to leave a trail of breadcrumbs so that if the process is hard-
    killed (kernel OOM killer -- no exception, no traceback, nothing) we can
    still see memory climbing toward the limit beforehand instead of the log
    just stopping with no explanation."""
    try:
        with open("/proc/self/status") as f:
            status = f.read()
        vmrssKb = next((int(line.split()[1]) for line in status.splitlines()
                        if line.startswith("VmRSS:")), None)
    except OSError:
        vmrssKb = None
    rssStr = "%.1f MB" % (vmrssKb / 1024.0) if vmrssKb is not None else "unknown"

    cgroupUsage, cgroupLimit = _cgroup_memory_usage_bytes()
    if cgroupUsage is not None and cgroupLimit:
        cgroupStr = "%.1f/%.1f MB (%.0f%%)" % (
            cgroupUsage / (1024 ** 2), cgroupLimit / (1024 ** 2), 100.0 * cgroupUsage / cgroupLimit)
    elif cgroupUsage is not None:
        cgroupStr = "%.1f MB (no limit found)" % (cgroupUsage / (1024 ** 2))
    else:
        cgroupStr = "unknown"

    gpuStr = "n/a"
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved  = torch.cuda.memory_reserved() / (1024 ** 2)
            gpuStr = "allocated=%.1f MB reserved=%.1f MB" % (allocated, reserved)
    except ImportError:
        pass

    logging.info("[mem] %-28s host RSS=%s | cgroup=%s | GPU %s", tag, rssStr, cgroupStr, gpuStr)


class _MemoryHeartbeat:
    """Logs memory usage on a background thread every `interval_sec` seconds
    while the `with` block runs. Meant to wrap long-running/opaque calls (like
    run_iqsm_plus) whose internals we can't add logging to directly -- without
    this, a run that gets OOM-killed mid-call leaves nothing in the log between
    the last line before the call and the kill event, which is exactly what
    happened on 2026-07-03 (9+ minutes of silence before exit code 137)."""

    def __init__(self, tag, interval_sec=5.0):
        self._tag = tag
        self._interval = interval_sec
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self):
        _log_memory_usage(self._tag + " (start)")
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        self._thread.join(timeout=self._interval + 1.0)
        _log_memory_usage(self._tag + " (end, exc=%s)" % (exc_type.__name__ if exc_type else "none"))

    def _run(self):
        while not self._stop_event.wait(self._interval):
            _log_memory_usage(self._tag)


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # ------------------------------------------------------------------
    # Buffer every incoming image, keyed by (image_type, image_series_index,
    # slice, contrast). image_series_index is included because some ME-GRE
    # protocols emit *two* magnitude series (e.g. raw + B1/intensity-corrected)
    # that otherwise share identical (image_type, slice, contrast) -- without
    # it, one series would silently overwrite the other right here, before
    # process_qsm() ever gets a chance to notice or choose between them.
    #
    # Streaming/order note: images arrive one at a time, in whatever order
    # ICE finishes them (not necessarily grouped or interleaved in any
    # predictable way -- see the DIS2DFunctor discussion). QSM needs the
    # complete 3D+echo volume before the network can run at all, so instead
    # of trying to detect "series complete" mid-stream, we simply accumulate
    # everything and process once when the connection closes (item is None).
    # This is the same fallback pattern used by i2i.py/analyzeflow.py for
    # their "untriggered" groups -- here it's just the primary path.
    # ------------------------------------------------------------------
    buffer = {}
    try:
        for item in connection:
            if isinstance(item, ismrmrd.Acquisition):
                raise Exception("Raw k-space data is not supported by this module")

            elif isinstance(item, ismrmrd.Image):
                buffer[(item.image_type, item.image_series_index, item.slice, item.contrast)] = item

            elif item is None:
                break

            else:
                logging.warning("Unsupported data type %s -- ignoring", type(item).__name__)

        if len(buffer) > 0:
            imagesOut = process_qsm(buffer, connection, config, metadata)
            if imagesOut:
                connection.send_image(imagesOut)
        else:
            logging.warning("No images received -- nothing to process")

    except Exception as e:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())

        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")


def _phase_to_radians(raw, meta):
    """Convert Siemens integer phase pixel values to radians."""
    slope     = mrdhelper.get_meta_value(meta, 'RescaleSlope')
    intercept = mrdhelper.get_meta_value(meta, 'RescaleIntercept')

    if (slope is not None) and (intercept is not None):
        rescaled = raw.astype(np.float64) * float(slope) + float(intercept)
    else:
        # Real-time ICE images may not carry DICOM-style rescale tags. Fall back to
        # Siemens' standard 12-bit convention (0..4095 representing -pi..+pi) and log
        # the raw range so this assumption can be verified against real streamed data.
        logging.warning("No RescaleSlope/RescaleIntercept in image metadata -- assuming "
                         "raw values already follow Siemens' 0..4095 -> -pi..+pi convention. "
                         "Raw range: [%s, %s]", raw.min(), raw.max())
        rescaled = raw.astype(np.float64) - 2048.0

    return (rescaled * (np.pi / 4096.0)).astype(np.float32)


def _get_voxel_size_mm(img, meta, metadata):
    """Voxel size [row/phase-dir, col/read-dir, slice-dir] in mm."""
    pixelSpacing   = mrdhelper.get_meta_value(meta, 'PixelSpacing')
    sliceThickness = mrdhelper.get_meta_value(meta, 'SliceThickness')

    if (pixelSpacing is not None) and (sliceThickness is not None):
        voxel_mm = [float(pixelSpacing[0]), float(pixelSpacing[1]), float(sliceThickness)]
        logging.info("voxel_size_mm source: DICOM PixelSpacing=%s SliceThickness=%s -> %s",
                     pixelSpacing, sliceThickness, voxel_mm)
        return voxel_mm

    # Fallback for real scanner data, where the explicit PixelSpacing/SliceThickness
    # MetaAttributes added by dicom2mrd.py won't be present. Derive from the standard
    # MRD header fields instead (field_of_view and reconSpace matrixSize).
    #
    # field_of_view.x/.y are the *full in-plane* FOV (pixel_spacing * matrix_size --
    # see dicom2mrd.py's CalcFieldOfView), so dividing by matrixSize.x/.y recovers
    # pixel spacing. field_of_view.z, however, is already the slice thickness itself
    # (CalcFieldOfView stores SliceThickness directly, not SliceThickness * nSlices)
    # -- it must NOT be divided by matrixSize.z (the partition/slice count) again.
    # Doing so previously divided the true slice thickness by nSlices (e.g. 3.0mm /
    # 72 slices = 0.0139mm), which fed a bogus near-zero voxel size into iQSM+'s
    # isotropic-interpolation step, causing it to try to upsample the in-plane axes
    # ~216x and exhaust the container's memory limit (see 2026-07-03 OOM incident).
    fov = img.field_of_view
    mtx = metadata.encoding[0].reconSpace.matrixSize
    voxel_mm = [float(fov[1]) / float(mtx.y), float(fov[0]) / float(mtx.x), float(fov[2])]
    logging.info("voxel_size_mm source: no DICOM PixelSpacing/SliceThickness meta -- "
                 "fallback from field_of_view=%s, reconSpace.matrixSize=(%s,%s,%s) -> %s",
                 list(fov), mtx.x, mtx.y, mtx.z, voxel_mm)
    return voxel_mm


def _get_b0_dir(img):
    """
    Approximate B0 direction in image-axis space [row/phase-dir, col/read-dir, slice-dir].
    B0 is always along the patient's superior-inferior (world z, in the same LPS patient
    coordinate system used by read_dir/phase_dir/slice_dir) axis, so project world-z onto
    the image's own axes.
    """
    world_z   = np.array([0.0, 0.0, 1.0])
    read_dir  = np.array(img.read_dir)
    phase_dir = np.array(img.phase_dir)
    slice_dir = np.array(img.slice_dir)
    logging.info("b0_dir source: read_dir=%s phase_dir=%s slice_dir=%s",
                 read_dir.tolist(), phase_dir.tolist(), slice_dir.tolist())

    b0_dir = np.array([np.dot(phase_dir, world_z), np.dot(read_dir, world_z), np.dot(slice_dir, world_z)])
    norm = np.linalg.norm(b0_dir)
    if norm < 1e-6:
        logging.warning("b0_dir projection has near-zero norm (%.2e) -- defaulting to [0,0,1]", norm)
        return [0.0, 0.0, 1.0]
    return (b0_dir / norm).tolist()


def _get_te_seconds(phaseKeys, nEchoes, metadata):
    """Echo times in seconds, ordered by contrast index 0..nEchoes-1."""
    try:
        te_ms = [float(x) for x in metadata.sequenceParameters.TE]
        if len(te_ms) == nEchoes:
            logging.info("TE(s) source: MRD header sequenceParameters.TE=%s ms", te_ms)
            return [t / 1000.0 for t in te_ms]
        logging.warning("Header TE list length (%d) does not match number of echoes (%d) "
                         "-- falling back to per-image EchoTime metadata", len(te_ms), nEchoes)
    except:
        logging.warning("Could not read TE list from MRD header -- falling back to "
                         "per-image EchoTime metadata")

    te_ms = [None] * nEchoes
    for (sl, ct), img in phaseKeys.items():
        if te_ms[ct] is None:
            meta = ismrmrd.Meta.deserialize(img.attribute_string)
            te_val = mrdhelper.get_meta_value(meta, 'EchoTime')
            if te_val is not None:
                te_ms[ct] = float(te_val)

    if any(t is None for t in te_ms):
        raise Exception("Could not determine echo time (TE) for all %d echoes" % nEchoes)

    logging.info("TE(s) source: per-image EchoTime metadata=%s ms", te_ms)
    return [t / 1000.0 for t in te_ms]


def _should_run_brain_extraction(config):
    """Read the 'brainextraction' Open Recon UI boolean parameter (default True).

    server.py passes the full parsed JSON config dict (not just the resolved config
    string) through as this function's `config` argument -- see server.py's
    `configAdditional` / `module.process(connection, configAdditional, metadata)`. Falls
    back to True (matching qsm_json_ui.json's own default) if config isn't in that dict
    shape, e.g. when testing locally via client.py without the JSON config message.
    """
    try:
        return bool(config['parameters'].get('brainextraction', True))
    except (TypeError, KeyError, AttributeError):
        return True


def _run_bet2(mag_nii_path, output_dir, fractional_intensity=0.5):
    """Run FSL's bet2 on a magnitude volume, returning the path to the binary brain mask
    NIfTI, or None if bet2 isn't available or the run fails. Brain extraction is an
    optional preprocessing step (see _should_run_brain_extraction) -- a failure here
    should never abort the whole QSM reconstruction, only skip masking.
    """
    if BET2_DIR is None:
        logging.warning("bet2 not found (checked $BET2_DIR, /opt/bet2, ./vendor/bet2) -- "
                         "skipping brain extraction")
        return None

    bet2Bin = os.path.join(BET2_DIR, "bin", "bet2")
    outPrefix = os.path.join(output_dir, "bet2_out")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = os.path.join(BET2_DIR, "lib")
    env.setdefault("FSLOUTPUTTYPE", "NIFTI_GZ")

    tic = perf_counter()
    try:
        result = subprocess.run(
            [bet2Bin, mag_nii_path, outPrefix, "-m", "-f", str(fractional_intensity)],
            env=env, capture_output=True, text=True, timeout=120,
        )
    except Exception:
        logging.warning("bet2 failed to run -- skipping brain extraction:\n%s", traceback.format_exc())
        return None

    if result.returncode != 0:
        logging.warning("bet2 exited with code %d -- skipping brain extraction. stdout=%s stderr=%s",
                        result.returncode, result.stdout, result.stderr)
        return None

    maskPath = outPrefix + "_mask.nii.gz"
    if not os.path.exists(maskPath):
        logging.warning("bet2 completed but mask file not found at %s -- skipping brain extraction",
                        maskPath)
        return None

    mask = nib.load(maskPath).get_fdata()
    logging.info("bet2 brain extraction completed in %.1f s -> %s (%.1f%% of voxels)",
                 perf_counter() - tic, maskPath, 100.0 * mask.sum() / mask.size)
    return maskPath


def process_qsm(buffer, connection, config, metadata):
    tic = perf_counter()

    _log_device_info()
    _log_memory_usage("process_qsm start")

    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # ------------------------------------------------------------------
    # Some ME-GRE protocols emit *two* magnitude series (e.g. a raw/uncorrected
    # one and a B1/intensity-corrected one) alongside the phase series. Both
    # map to image_type == IMTYPE_MAGNITUDE, so without disambiguation they'd
    # silently collide on the same (slice, contrast) buffer key -- whichever
    # happened to be processed last would win, arbitrarily. Disambiguate on
    # image_series_index instead (which does differ between the two mag
    # series) and pick exactly one, deterministically -- preferring whichever
    # series carries Siemens' 'NORM' ImageType flag (PreScan Normalize /
    # B1-intensity-corrected), since dicom2mrd.py preserves the full ImageType
    # list per image. Falls back to the lowest image_series_index if the NORM
    # flag isn't present/isn't unambiguous (e.g. real-time ICE images that
    # never carry DICOM-style ImageType metadata at all).
    #
    # Note: iQSM+ only uses magnitude for the multi-echo weighted combination
    # of per-echo chi maps (weights = (mag*TE)**2, see inference.py) -- a
    # purely spatial per-voxel multiplicative correction (which is what B1/
    # intensity correction is) cancels out of that weighted average, so which
    # of the two series is picked should not change the QSM result. The
    # QSM_MAGNITUDE_SERIES_INDEX env var can force a specific series if that
    # assumption ever proves wrong for a given protocol.
    # ------------------------------------------------------------------
    allMagSeries = sorted(set(si for (t, si, sl, ct) in buffer.keys() if t == ismrmrd.IMTYPE_MAGNITUDE))

    def _series_has_norm_flag(seriesIndex):
        for (t, si, sl, ct), img in buffer.items():
            if t == ismrmrd.IMTYPE_MAGNITUDE and si == seriesIndex:
                imageType = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(img.attribute_string), 'ImageType')
                return imageType is not None and 'NORM' in imageType
        return False

    forcedSeries = os.environ.get("QSM_MAGNITUDE_SERIES_INDEX")
    if forcedSeries is not None:
        magSeriesIndex = int(forcedSeries)
    elif len(allMagSeries) > 1:
        normSeries = [si for si in allMagSeries if _series_has_norm_flag(si)]
        if len(normSeries) == 1:
            magSeriesIndex = normSeries[0]
            reason = "carries Siemens' 'NORM' ImageType flag (B1/intensity-corrected)"
        else:
            magSeriesIndex = allMagSeries[0]
            reason = "no unambiguous 'NORM' flag found -- defaulting to lowest image_series_index"
        logging.warning("Received magnitude images from %d distinct series (image_series_index=%s) "
                         "-- this protocol likely emits both a raw and a B1/intensity-corrected "
                         "magnitude series. Using image_series_index=%d (%s); set "
                         "QSM_MAGNITUDE_SERIES_INDEX to override.",
                         len(allMagSeries), allMagSeries, magSeriesIndex, reason)
    elif len(allMagSeries) == 1:
        magSeriesIndex = allMagSeries[0]
    else:
        magSeriesIndex = None

    magKeys   = {(sl, ct): img for (t, si, sl, ct), img in buffer.items()
                 if t == ismrmrd.IMTYPE_MAGNITUDE and si == magSeriesIndex}
    phaseKeys = {(sl, ct): img for (t, si, sl, ct), img in buffer.items() if t == ismrmrd.IMTYPE_PHASE}

    if len(phaseKeys) == 0:
        raise Exception("No phase images received -- QSM requires magnitude and phase images")

    nSlices = max(sl for (sl, ct) in phaseKeys.keys()) + 1
    nEchoes = max(ct for (sl, ct) in phaseKeys.keys()) + 1
    logging.info("Buffered %d magnitude and %d phase images -- expecting %d slices x %d echoes",
                 len(magKeys), len(phaseKeys), nSlices, nEchoes)

    sampleImg = next(iter(phaseKeys.values()))
    rows, cols = sampleImg.data.shape[-2:]

    magVol   = np.zeros((rows, cols, nSlices, nEchoes), dtype=np.float32)
    phaseVol = np.zeros((rows, cols, nSlices, nEchoes), dtype=np.float32)

    missing = 0
    for sl in range(nSlices):
        for ct in range(nEchoes):
            pImg = phaseKeys.get((sl, ct))
            if pImg is None:
                missing += 1
                continue
            meta = ismrmrd.Meta.deserialize(pImg.attribute_string)
            phaseVol[:, :, sl, ct] = _phase_to_radians(pImg.data[0, 0, :, :], meta)

            mImg = magKeys.get((sl, ct))
            if mImg is not None:
                magVol[:, :, sl, ct] = mImg.data[0, 0, :, :].astype(np.float32)

    if missing > 0:
        logging.warning("%d of %d expected (slice, echo) phase images were missing -- "
                         "those slices were left as zero", missing, nSlices * nEchoes)

    logging.info("Buffered volumes: magVol shape=%s dtype=%s (%.1f MB), phaseVol shape=%s dtype=%s (%.1f MB)",
                 magVol.shape, magVol.dtype, magVol.nbytes / (1024 ** 2),
                 phaseVol.shape, phaseVol.dtype, phaseVol.nbytes / (1024 ** 2))
    _log_memory_usage("after buffering volumes")

    np.save(os.path.join(debugFolder, "magVol.npy"), magVol)
    np.save(os.path.join(debugFolder, "phaseVol_rad.npy"), phaseVol)

    # ------------------------------------------------------------------
    # Gather the acquisition parameters iQSM+ needs
    # ------------------------------------------------------------------
    metaSample = ismrmrd.Meta.deserialize(sampleImg.attribute_string)
    voxel_mm   = _get_voxel_size_mm(sampleImg, metaSample, metadata)
    b0_dir     = _get_b0_dir(sampleImg)
    te_sec     = _get_te_seconds(phaseKeys, nEchoes, metadata)

    try:
        b0_tesla = float(metadata.acquisitionSystemInformation.systemFieldStrength_T)
        logging.info("b0 source: MRD header acquisitionSystemInformation.systemFieldStrength_T=%.4fT",
                     b0_tesla)
    except:
        logging.warning("Could not read systemFieldStrength_T from header -- assuming 3.0T")
        b0_tesla = 3.0

    # Consolidated view of every acquisition parameter fed into iQSM+, for quick sanity-checking
    # against the individual "source:" lines logged above (which show what each was derived from).
    logging.info("QSM parameters fed to iQSM+: voxel_size_mm=%s, b0_dir=%s, b0=%.2fT, TE(s)=%s, "
                 "n_slices=%d, n_echoes=%d",
                 voxel_mm, b0_dir, b0_tesla, te_sec, nSlices, nEchoes)

    # ------------------------------------------------------------------
    # Run iQSM+ (operates on NIfTI files on disk, not in-memory arrays)
    # ------------------------------------------------------------------
    try:
        from inference import run_iqsm_plus, CheckpointNotFoundError
    except ImportError as e:
        raise Exception("Could not import iQSM+ from IQSM_PLUS_DIR='%s' (%s). Set the "
                         "IQSM_PLUS_DIR environment variable to a valid iQSM_Plus checkout." %
                         (IQSM_PLUS_DIR, e))

    affine = np.diag(voxel_mm + [1.0])

    # ------------------------------------------------------------------
    # Optional brain extraction (FSL's bet2), toggled by the 'Brain Extraction (BET)'
    # Open Recon UI parameter (default on -- see qsm_json_ui.json). bet2 needs a plain
    # 3D volume, not the 4D multi-echo array saved above, so run it on the first echo
    # only -- the resulting mask is reused for every echo by run_iqsm_plus.
    # ------------------------------------------------------------------
    maskPath = None
    if _should_run_brain_extraction(config):
        mag3dPath = os.path.join(debugFolder, "mag_echo0_for_bet2.nii.gz")
        nib.save(nib.Nifti1Image(magVol[..., 0], affine), mag3dPath)
        maskPath = _run_bet2(mag3dPath, debugFolder)
    else:
        logging.info("Brain extraction disabled via 'brainextraction' UI parameter")

    # run_iqsm_plus() processes a single echo per call -- multi-echo combination
    # is handled externally by the caller, exactly as iQSM_Plus's own run.py /
    # app.py do (see inference.py's docstring; this API changed from an internal
    # te_values=list loop to this per-echo form upstream). Model weights are
    # cached globally inside get_model() (keyed by device), so looping here does
    # NOT reload them from disk each iteration -- only the per-echo forward pass
    # itself repeats, which is unavoidable (the network has no cross-echo
    # batching), so this costs nothing extra versus the old internal-loop API.
    #
    # run_iqsm_plus() is opaque to us (lives in the separate iQSM_Plus checkout,
    # not this repo) and was observed to run for 9+ minutes with zero log output
    # before the container was killed by the kernel OOM killer (exit code 137,
    # see 2026-07-03 incident). Since a kernel SIGKILL gives no chance to log or
    # raise an exception from inside that call, _MemoryHeartbeat logs memory on a
    # background thread every few seconds *during* the call, so the last few
    # heartbeats before a future kill will show how memory was trending.
    inferenceStart = perf_counter()
    qsmVolumes = []
    try:
        with _MemoryHeartbeat("during run_iqsm_plus"):
            for echo in range(nEchoes):
                echoPhasePath = os.path.join(debugFolder, "phase_echo%d.nii.gz" % echo)
                echoMagPath   = os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo)
                nib.save(nib.Nifti1Image(phaseVol[..., echo], affine), echoPhasePath)
                nib.save(nib.Nifti1Image(magVol[..., echo],   affine), echoMagPath)

                logging.info("Running iQSM+ on echo %d/%d (TE=%.4f s)", echo + 1, nEchoes, te_sec[echo])
                echoQsmPath = run_iqsm_plus(
                    phase_nii_path=echoPhasePath,
                    te=float(te_sec[echo]),
                    mag_nii_path=echoMagPath,
                    mask_nii_path=maskPath,
                    voxel_size=voxel_mm,
                    b0_dir=b0_dir,
                    b0=b0_tesla,
                    output_dir=os.path.join(debugFolder, "echo%d_output" % echo),
                )
                qsmVolumes.append(nib.load(echoQsmPath).get_fdata(dtype=np.float32))
    except CheckpointNotFoundError as e:
        raise Exception("iQSM+ model checkpoints not found: %s" % e)
    except Exception:
        logging.error("run_iqsm_plus() raised after %.1f s:\n%s",
                      perf_counter() - inferenceStart, traceback.format_exc())
        raise
    logging.info("run_iqsm_plus() completed %d echo(es) in %.1f s", nEchoes, perf_counter() - inferenceStart)

    # Magnitude x TE^2 weighted average across echoes -- mirrors iQSM_Plus's own
    # run.py:_run_multi_echo() combiner exactly.
    qsmStack = np.stack(qsmVolumes, axis=-1)
    teWeights = (magVol * np.array(te_sec, dtype=np.float32).reshape(1, 1, 1, -1)) ** 2
    teWeightsSum = teWeights.sum(axis=-1)
    teWeightsSum[teWeightsSum == 0] = 1.0
    qsmVol = ((teWeights * qsmStack).sum(axis=-1) / teWeightsSum).astype(np.float32)
    np.save(os.path.join(debugFolder, "qsmVol.npy"), qsmVol)

    # ------------------------------------------------------------------
    # Quantize the signed float ppm values into an unsigned 16-bit integer
    # range, with RescaleSlope/RescaleIntercept so a DICOM viewer can recover
    # the true value (real = pixel * slope + intercept). This is required
    # because DICOM's PixelData is fundamentally integer-only -- sending raw
    # float32 values (as MRD itself allows) gets misinterpreted as garbage
    # integers once converted to DICOM.
    #
    # A *fixed* clinical range (rather than one computed per-scan from that
    # scan's own min/max) is used so the same real ppm value always maps to
    # the same raw pixel count across every reconstruction -- otherwise raw
    # pixel values aren't comparable between scans/patients/timepoints unless
    # the rescale is always applied first. +/-4 ppm comfortably covers real
    # brain tissue (rarely beyond +/-1 ppm even for iron-rich deep grey
    # matter) while still clipping only the most extreme unmasked background
    # artifacts. Stored as *unsigned* (rather than signed int16) because
    # mrd2dicom.py always defaults PixelRepresentation to 0 (unsigned)
    # regardless of the source data's sign -- the same convention already
    # used for the input phase images (RescaleSlope=2, RescaleIntercept=-4096
    # in the original scanner DICOMs).
    # ------------------------------------------------------------------
    QSM_DISPLAY_RANGE_PPM = 4.0  # clip/quantize over [-4, +4] ppm
    rescaleIntercept = -QSM_DISPLAY_RANGE_PPM
    rescaleSlope     = (2.0 * QSM_DISPLAY_RANGE_PPM) / 65535.0
    qsmVol_quantized = np.clip(np.round((qsmVol - rescaleIntercept) / rescaleSlope), 0, 65535).astype(np.uint16)

    chi_min = float(qsmVol.min())
    chi_max = float(qsmVol.max())
    n_clipped = int(np.sum((qsmVol < -QSM_DISPLAY_RANGE_PPM) | (qsmVol > QSM_DISPLAY_RANGE_PPM)))
    if n_clipped > 0:
        logging.warning("%d of %d voxels fell outside +/-%.1f ppm and were clipped",
                         n_clipped, qsmVol.size, QSM_DISPLAY_RANGE_PPM)
    logging.info("QSM value range [%.4f, %.4f] ppm -> quantized to uint16 over fixed [-%.1f, %.1f] ppm "
                 "(RescaleSlope=%.8g, RescaleIntercept=%.4f)",
                 chi_min, chi_max, QSM_DISPLAY_RANGE_PPM, QSM_DISPLAY_RANGE_PPM, rescaleSlope, rescaleIntercept)

    toc = perf_counter()
    strProcessTime = "QSM reconstruction time: %.2f s" % (toc - tic)
    logging.info(strProcessTime)
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # ------------------------------------------------------------------
    # Re-slice the 3D susceptibility map back into individual 2D MRD images,
    # matching the granularity ICE expects for its per-slice DICOM pipeline
    # (see the earlier "2D vs 3D streaming" discussion). Geometry/header for
    # each slice is copied from that slice's first-echo magnitude image.
    # ------------------------------------------------------------------
    imagesOut = []
    for sl in range(nSlices):
        templateImg = magKeys.get((sl, 0)) or phaseKeys.get((sl, 0))
        if templateImg is None:
            continue

        qsmImg = ismrmrd.Image.from_array(qsmVol_quantized[:, :, sl], transpose=False)

        oldHeader = templateImg.getHead()
        oldHeader.data_type          = qsmImg.getHead().data_type
        oldHeader.image_type         = ismrmrd.IMTYPE_MAGNITUDE
        oldHeader.image_index        = sl + 1
        oldHeader.image_series_index = 100
        qsmImg.setHead(oldHeader)

        tmpMeta = ismrmrd.Meta.deserialize(templateImg.attribute_string)
        tmpMeta['DataRole']                      = 'Image'
        tmpMeta['ImageProcessingHistory']        = ['PYTHON', 'IQSM_PLUS']
        # QSM is computed/derived data, not the original acquisition -- override the
        # inherited magnitude template's ImageType (['ORIGINAL', 'PRIMARY', 'M', ...])
        # rather than passing it through unchanged.
        tmpMeta['ImageType']                     = ['DERIVED', 'SECONDARY', 'M']
        tmpMeta['SequenceDescriptionAdditional'] = 'QSM'
        tmpMeta['ImageComments']                 = 'QSM (ppm), iQSM+'
        # WindowCenter/WindowWidth are in real-world (rescaled) ppm units per DICOM
        # convention -- VOI windowing is applied after the RescaleSlope/Intercept
        # (Modality LUT) transform, not to the raw quantized pixel values.
        tmpMeta['WindowCenter']                  = '0'
        tmpMeta['WindowWidth']                   = '0.6'
        # DICOM's DS (Decimal String) value representation caps field length at 16
        # characters -- plain str(float) can exceed that (e.g. for small slopes in
        # scientific notation), so format explicitly rather than relying on repr.
        tmpMeta['RescaleSlope']                  = "{:.6e}".format(rescaleSlope)
        tmpMeta['RescaleIntercept']               = "{:.6f}".format(rescaleIntercept)
        tmpMeta['Keep_image_geometry']            = 1

        if tmpMeta.get('ImageRowDir') is None:
            tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]
        if tmpMeta.get('ImageColumnDir') is None:
            tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

        qsmImg.attribute_string = tmpMeta.serialize()
        imagesOut.append(qsmImg)

    # ------------------------------------------------------------------
    # Pass through every originally-received image (all magnitude series,
    # all echoes, phase) unmodified, as their own series alongside the new
    # QSM map. Per Open Recon's documented behavior, only images explicitly
    # returned by the app are saved to DICOM/displayed on the scanner -- the
    # standard ICE-reconstructed images are NOT automatically preserved
    # (see or_sdk/README.md: "only images that are returned by the Open
    # Recon app are saved to DICOMs ... standard Siemens reconstructed
    # images are not automatically saved"). Without this, the original
    # acquisition series would simply be discarded, not left untouched.
    # These are the exact objects received from the Emitter -- only their
    # .data/.attribute_string were read (never mutated) when building the
    # QSM volumes/output above, so returning them here reproduces the same
    # DICOMs the scanner would have produced without Open Recon involved.
    # ------------------------------------------------------------------
    nQsmImages = len(imagesOut)
    imagesOut.extend(buffer.values())

    logging.info("Returning %d QSM image(s) + %d original image(s) = %d total",
                 nQsmImages, len(buffer), len(imagesOut))
    return imagesOut
