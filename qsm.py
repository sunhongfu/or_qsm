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
# first (for manual overrides), then fall back to where it ends up in the Docker image,
# then to a path relative to this file's own location (which is where it lands as a
# local checkout -- devcontainer, native venv, or plain git clone), rather than
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

# DeepRelaxo (https://github.com/sunhongfu/DeepRelaxo) -- R2* mapping, same
# separate-repo/gitignored-subfolder pattern as IQSM_PLUS_DIR above.
_DEEPRELAXO_CANDIDATES = [
    os.environ.get("DEEPRELAXO_DIR"),
    "/opt/code/python-ismrmrd-server/DeepRelaxo",       # baked into the Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepRelaxo"),  # local checkout
]
DEEPRELAXO_DIR = next((p for p in _DEEPRELAXO_CANDIDATES if p and os.path.isdir(p)), None)

# NOT added to sys.path at module level like IQSM_PLUS_DIR above -- both iQSM_Plus and
# DeepRelaxo happen to define their own bare top-level `data_utils.py` module (unrelated
# repos, coincidentally similar layout), and Python caches imports by module name, so
# having both directories on sys.path simultaneously risks one repo's `import data_utils`
# silently resolving to the *other* repo's file depending on sys.path order -- a silent
# wrong-answer bug, not an ImportError, since both define same-named functions with
# different implementations. iQSM_Plus's inference.py doesn't currently import
# data_utils itself, but relying on that staying true forever is fragile. _import_deeprelaxo()
# instead does the sys.path insertion (and a sys.modules cache-bust) right before the one
# place DeepRelaxo is actually imported, guaranteeing DeepRelaxo's own data_utils.py wins
# regardless of import order elsewhere in the process.
def _import_deeprelaxo():
    if DEEPRELAXO_DIR is None:
        raise Exception("DeepRelaxo not found (checked $DEEPRELAXO_DIR, "
                         "/opt/code/python-ismrmrd-server/DeepRelaxo, ./DeepRelaxo)")
    sys.modules.pop("data_utils", None)
    if DEEPRELAXO_DIR in sys.path:
        sys.path.remove(DEEPRELAXO_DIR)
    sys.path.insert(0, DEEPRELAXO_DIR)
    from run_estimator_stage import estimate_r2s
    from run_denoiser_stage import denoise_r2s_map
    return estimate_r2s, denoise_r2s_map


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


def _get_qsm_output_mode(config):
    """Read the 'qsmoutput' Open Recon UI choice parameter (default 'both').

    Returns one of 'both', 'masked', 'wholehead'. server.py passes the full parsed JSON
    config dict (not just the resolved config string) through as this function's `config`
    argument -- see server.py's `configAdditional` / `module.process(connection,
    configAdditional, metadata)`. Falls back to 'both' (matching qsm_json_ui.json's own
    default) if config isn't in that dict shape (e.g. testing locally via client.py
    without the JSON config message) or the value doesn't match a known option -- same
    fail-safe spirit as the old boolean toggle this replaced (see git history), just with
    a wider set of fallback triggers since there are now 3 valid strings instead of 2.
    """
    try:
        value = config['parameters'].get('qsmoutput', 'both')
    except (TypeError, KeyError, AttributeError):
        return 'both'
    value = str(value).strip().lower()
    return value if value in ('both', 'masked', 'wholehead') else 'both'


def _get_r2s_enabled(config):
    """Read the 'r2smapping' Open Recon UI boolean parameter (default True).

    Same string/bool handling as the old 'brainextraction' toggle this doesn't replace
    (see _get_qsm_output_mode's docstring) -- Open Recon's Injector has been observed to
    send boolean UI parameters as the *string* "false", and bool("false") is True in
    Python, so this must not just bool(...) the raw value.
    """
    try:
        value = config['parameters'].get('r2smapping', True)
    except (TypeError, KeyError, AttributeError):
        return True
    if isinstance(value, str):
        return value.strip().lower() not in ('false', '0', '')
    return bool(value)


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
    # Which QSM output(s) to produce, per the 'QSM Output' Open Recon UI choice
    # parameter (default 'both' -- see qsm_json_ui.json). These are genuinely two
    # different network inputs, not one output post-processed into two: iQSM+'s phase
    # input is masked *before* inference (phase = phase * mask, matching the MATLAB
    # reference's Save_Input_iQSMplus.m), so there's no way to derive the brain-extracted
    # result from the whole-head result (or vice versa) after the fact -- 'both' means
    # running the full echo loop and network inference twice, roughly doubling
    # reconstruction time versus either mode alone.
    # ------------------------------------------------------------------
    outputMode = _get_qsm_output_mode(config)
    needMasked = outputMode in ('both', 'masked')
    needWholehead = outputMode in ('both', 'wholehead')

    # bet2 needs a plain 3D volume, not the 4D multi-echo array saved above, so run it on
    # the first echo only -- the resulting mask is reused for every echo below.
    maskPath = None
    if needMasked:
        mag3dPath = os.path.join(debugFolder, "mag_echo0_for_bet2.nii.gz")
        nib.save(nib.Nifti1Image(magVol[..., 0], affine), mag3dPath)
        maskPath = _run_bet2(mag3dPath, debugFolder)
        if maskPath is None:
            logging.warning("Brain-extracted QSM output was requested but bet2 failed/was "
                             "unavailable -- skipping the brain-extracted series")
            needMasked = False
            needWholehead = True  # still return at least one QSM series
    else:
        logging.info("Whole-head-only QSM output selected -- skipping brain extraction")

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
    def _reconstruct_qsm(maskNiiPath, label):
        """Run iQSM+ across all echoes with the given brain mask (None for whole-head,
        unmasked reconstruction) and combine echoes into one susceptibility volume.
        `label` disambiguates debug output/log lines when this runs twice (both-mode).
        """
        inferenceStart = perf_counter()
        qsmVolumes = []
        try:
            with _MemoryHeartbeat("during run_iqsm_plus (%s)" % label):
                for echo in range(nEchoes):
                    echoPhasePath = os.path.join(debugFolder, "phase_echo%d.nii.gz" % echo)
                    echoMagPath   = os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo)
                    nib.save(nib.Nifti1Image(phaseVol[..., echo], affine), echoPhasePath)
                    nib.save(nib.Nifti1Image(magVol[..., echo],   affine), echoMagPath)

                    logging.info("Running iQSM+ (%s) on echo %d/%d (TE=%.4f s)",
                                 label, echo + 1, nEchoes, te_sec[echo])
                    echoQsmPath = run_iqsm_plus(
                        phase_nii_path=echoPhasePath,
                        te=float(te_sec[echo]),
                        mag_nii_path=echoMagPath,
                        mask_nii_path=maskNiiPath,
                        voxel_size=voxel_mm,
                        b0_dir=b0_dir,
                        b0=b0_tesla,
                        output_dir=os.path.join(debugFolder, "%s_echo%d_output" % (label, echo)),
                    )
                    qsmVolumes.append(nib.load(echoQsmPath).get_fdata(dtype=np.float32))
        except CheckpointNotFoundError as e:
            raise Exception("iQSM+ model checkpoints not found: %s" % e)
        except Exception:
            logging.error("run_iqsm_plus() (%s) raised after %.1f s:\n%s",
                          label, perf_counter() - inferenceStart, traceback.format_exc())
            raise
        logging.info("run_iqsm_plus() (%s) completed %d echo(es) in %.1f s",
                     label, nEchoes, perf_counter() - inferenceStart)

        # Magnitude x TE^2 weighted average across echoes -- mirrors iQSM_Plus's own
        # run.py:_run_multi_echo() combiner exactly.
        qsmStack = np.stack(qsmVolumes, axis=-1)
        teWeights = (magVol * np.array(te_sec, dtype=np.float32).reshape(1, 1, 1, -1)) ** 2
        teWeightsSum = teWeights.sum(axis=-1)
        # Guard against dividing by a *near*-zero (not just exactly-zero) denominator --
        # confirmed as the actual cause of a real scanner artifact: at image edges/background
        # (air, weak MR signal), magVol is typically tiny but essentially never exactly 0.0 in
        # real acquired data (thermal noise floor), so the old `teWeightsSum == 0` check let
        # tiny-but-nonzero denominators through, amplifying numerator noise into wildly
        # extreme susceptibility values that then clipped to exactly -4ppm at the display
        # stage -- visible as a hard-edged, incorrect band at the image boundary. Voxels below
        # a small relative threshold (no reliable signal to combine) are set to 0 ppm directly
        # instead of dividing by a near-zero number.
        weightThreshold = float(teWeightsSum.max()) * 1e-3
        reliableVoxels = teWeightsSum > weightThreshold
        qsmVol = np.zeros(teWeightsSum.shape, dtype=np.float32)
        qsmVol[reliableVoxels] = ((teWeights * qsmStack).sum(axis=-1) / teWeightsSum)[reliableVoxels]
        logging.info("Multi-echo combination (%s): %d of %d voxels (%.1f%%) below weight threshold "
                     "(insufficient signal to combine) -- set to 0 ppm instead of divided",
                     label, int((~reliableVoxels).sum()), teWeightsSum.size,
                     100.0 * (~reliableVoxels).sum() / teWeightsSum.size)
        np.save(os.path.join(debugFolder, "qsmVol_%s.npy" % label), qsmVol)
        return qsmVol

    # (label, image_series_index, SequenceDescriptionAdditional, ImageComments, qsmVol).
    # image_series_index=100 is kept for the brain-extracted series specifically (not
    # reassigned to whichever mode happens to run first) for backward compatibility --
    # 100 was the only QSM series index this app ever produced before this change
    # (readme.md and RunQSMRecon.ipynb both already document/rely on "image_100" =
    # the QSM map), and brain extraction defaulted on before 'qsmoutput' existed, so
    # existing tooling that assumes image_series_index=100 keeps working unmodified for
    # anyone still on 'masked' or 'both' mode. 101 is the new whole-head series.
    qsmResults = []
    if needWholehead:
        qsmResults.append(("wholehead", 101, "QSM_WHOLEHEAD",
                            "QSM whole-head (ppb = ppm*1e3), iQSM+",
                            _reconstruct_qsm(None, "wholehead")))
    if needMasked:
        qsmResults.append(("masked", 100, "QSM_MASKED",
                            "QSM brain-extracted (ppb = ppm*1e3), iQSM+",
                            _reconstruct_qsm(maskPath, "masked")))

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
    #
    # QSM_PIXEL_MAX = 4095 (12-bit), not 65535 (16-bit): confirmed on a real
    # scanner export that Open Recon's Injector writes MR-derived images as
    # BitsAllocated=16 but BitsStored=12/HighBit=11 -- a fixed Siemens
    # convention for this image type that we can't change from here. Values
    # we send above 4095 don't get clipped/rescaled by the Injector, they get
    # silently truncated to their low 12 bits, which corrupts almost
    # everything (a near-zero-ppm voxel encodes to raw~32768 = 0x8000 under
    # the old 65535 scheme, whose low 12 bits are 0 -- collapsing most of the
    # image to exactly RescaleIntercept, i.e. -4 ppm, on real hardware).
    # Quantizing into 0..4095 up front means our own values already fit
    # inside whatever the Injector actually preserves.
    # ------------------------------------------------------------------
    QSM_DISPLAY_RANGE_PPM = 4.0  # clip/quantize over [-4, +4] ppm
    QSM_PIXEL_MAX = 4095  # 12-bit -- see rationale above
    rescaleIntercept = -QSM_DISPLAY_RANGE_PPM
    rescaleSlope     = (2.0 * QSM_DISPLAY_RANGE_PPM) / QSM_PIXEL_MAX

    # The DICOM-facing RescaleSlope/Intercept (and WindowCenter/Width below) report
    # values in ppb (ppm * 1000), not ppm directly -- confirmed on the real scanner that
    # its window/level adjustment tool has coarse (effectively integer-only) granularity
    # in whatever real-world unit RescaleSlope/Intercept report, so a genuine ppm-scale
    # window (e.g. WindowWidth=1) only has a handful of distinct adjustable positions
    # across the whole clinically-relevant range -- nowhere near enough for usable
    # mouse-based adjustment (compare magnitude images, whose real-world W/L values are
    # naturally in the hundreds-to-thousands, since RescaleSlope~1 there). This doesn't
    # change any pixel data or the true value any voxel represents, only the *units* a
    # DICOM viewer reports/adjusts it in -- RescaleType is set to PPB below so anything
    # that respects it knows the units changed.
    DICOM_UNIT_SCALE = 1000.0  # ppm -> ppb
    dicomRescaleSlope     = rescaleSlope * DICOM_UNIT_SCALE
    dicomRescaleIntercept = rescaleIntercept * DICOM_UNIT_SCALE

    # ------------------------------------------------------------------
    # Optional R2* mapping (DeepRelaxo, https://github.com/sunhongfu/DeepRelaxo), gated by
    # the 'Compute R2* Mapping' Open Recon UI parameter (default on -- see
    # qsm_json_ui.json). Uses the SAME whole-head/masked/both selection ('qsmoutput') and
    # the SAME bet2 mask already computed for QSM above -- no separate brain-extraction
    # run. DeepRelaxo's estimator stage is a per-voxel Transformer-MLP (confirmed:
    # LayerNorm + Dropout only, no BatchNorm, so it has no cross-voxel/batch-composition
    # dependence) -- a whole-head estimator pass is therefore exactly reusable to derive
    # the masked variant by zeroing background before a second (fast) denoiser-only pass,
    # instead of needing a second full estimator run over the masked-only voxel subset.
    # Only in 'masked'-only mode (no whole-head pass computed anyway) does it fall back to
    # letting the estimator itself restrict to the masked voxels directly, which is faster
    # than computing the whole head only to discard most of it. Reuses the exact
    # mag_echo{N}.nii.gz files already written by _reconstruct_qsm() above (always written
    # at least once, regardless of qsmoutput mode) rather than re-saving them.
    #
    # Failures here are non-fatal to the QSM output -- caught broadly (covers e.g. a
    # missing DeepRelaxo checkout/checkpoints) and logged, same fail-soft spirit as bet2
    # failures above, so a broken R2* dependency never blocks the QSM reconstruction the
    # rest of this module already validated.
    # ------------------------------------------------------------------
    r2sResults = []  # (label, image_series_index, SequenceDescriptionAdditional, ImageComments, r2sVol)
    if _get_r2s_enabled(config):
        try:
            estimate_r2s, denoise_r2s_map = _import_deeprelaxo()
            magnitude_entries = [{"path": os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo)}
                                  for echo in range(nEchoes)]
            te_values_ms = [t * 1000.0 for t in te_sec]

            r2sWholeheadMap = None
            if needWholehead:
                r2sTic = perf_counter()
                # DeepRelaxo's per-voxel estimator has no internal progress logging (unlike
                # iQSM+'s inference.py) -- a whole-head pass processes every voxel (~8.6M for
                # a typical 256x192x176 volume) in ~50000-voxel batches with zero console
                # output in between, which otherwise looks indistinguishable from a hang in
                # the log. Same _MemoryHeartbeat wrapper as the iQSM+ calls above.
                with _MemoryHeartbeat("during DeepRelaxo estimator (wholehead)"):
                    r2sWholeheadMap, _ = estimate_r2s(magnitude_entries=magnitude_entries,
                                                       te_values_ms=te_values_ms, bet_mask_path=None)
                r2sWholeheadMap = r2sWholeheadMap.numpy()
                wholeheadMask = np.ones(r2sWholeheadMap.shape, dtype=bool)
                r2sWholeheadDenoised = denoise_r2s_map(r2sWholeheadMap, wholeheadMask)
                logging.info("DeepRelaxo (wholehead) completed in %.1f s", perf_counter() - r2sTic)
                r2sResults.append(("wholehead", 103, "R2S_WHOLEHEAD",
                                    "R2* whole-head (s^-1), DeepRelaxo", r2sWholeheadDenoised))

            if needMasked and maskPath is not None:
                r2sTic = perf_counter()
                maskArr = nib.load(maskPath).get_fdata() > 0
                if r2sWholeheadMap is not None:
                    r2sMaskedMap = r2sWholeheadMap.copy()
                    r2sMaskedMap[~maskArr] = 0.0
                    logging.info("DeepRelaxo (masked) derived from whole-head estimator pass "
                                 "(shared-estimator optimization)")
                else:
                    with _MemoryHeartbeat("during DeepRelaxo estimator (masked)"):
                        r2sMaskedMap, _ = estimate_r2s(magnitude_entries=magnitude_entries,
                                                        te_values_ms=te_values_ms, bet_mask_path=maskPath)
                    r2sMaskedMap = r2sMaskedMap.numpy()
                r2sMaskedDenoised = denoise_r2s_map(r2sMaskedMap, maskArr)
                logging.info("DeepRelaxo (masked) completed in %.1f s", perf_counter() - r2sTic)
                r2sResults.append(("masked", 102, "R2S_MASKED",
                                    "R2* brain-extracted (s^-1), DeepRelaxo", r2sMaskedDenoised))
        except Exception:
            logging.error("R2* mapping (DeepRelaxo) failed -- continuing without R2* output:\n%s",
                          traceback.format_exc())

    # R2* is non-negative and naturally spans tens-to-~100+ s^-1 in brain tissue, unlike
    # QSM's tiny +/-4ppm range -- no ppb-style unit inflation needed here (see
    # DICOM_UNIT_SCALE above) for a scanner W/L tool to have usable granularity; reporting
    # directly in s^-1 already gives a comparably large integer range to magnitude images.
    # 250 s^-1 was chosen after a real local test run showed values up to ~226 s^-1
    # (whole-head, uncropped background/edge voxels included) -- a 100 s^-1 ceiling
    # clipped ~6% of whole-head voxels and ~1.8% of brain-masked voxels at that range.
    # Tune to your own clinical experience if needed.
    R2S_DISPLAY_RANGE_MAX = 250.0  # clip/quantize over [0, 250] s^-1
    R2S_PIXEL_MAX = 4095  # 12-bit, same Injector constraint as QSM (see rationale above)
    r2sRescaleSlope = R2S_DISPLAY_RANGE_MAX / R2S_PIXEL_MAX
    r2sRescaleIntercept = 0.0

    toc = perf_counter()
    strProcessTime = "QSM reconstruction time: %.2f s" % (toc - tic)
    logging.info(strProcessTime)
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # ------------------------------------------------------------------
    # Re-slice each 3D map (QSM and/or R2*) back into individual 2D MRD images, matching
    # the granularity ICE expects for its per-slice DICOM pipeline (see the earlier "2D vs
    # 3D streaming" discussion). Geometry/header for each slice is copied from that
    # slice's first-echo magnitude image. One full pass here per requested output series
    # (1-4, depending on qsmoutput/r2smapping selection), each as its own DICOM series.
    # ------------------------------------------------------------------
    imagesOut = []

    def _append_dicom_series(vol, seriesIndex, seqDescAdditional, imageComments,
                              seriesRescaleSlope, seriesRescaleIntercept, rescaleType,
                              windowCenter, windowWidth, pixelMax, processingHistory):
        volQuantized = np.clip(np.round((vol - seriesRescaleIntercept) / seriesRescaleSlope),
                                0, pixelMax).astype(np.uint16)
        for sl in range(nSlices):
            templateImg = magKeys.get((sl, 0)) or phaseKeys.get((sl, 0))
            if templateImg is None:
                continue

            img = ismrmrd.Image.from_array(volQuantized[:, :, sl], transpose=False)

            oldHeader = templateImg.getHead()
            oldHeader.data_type          = img.getHead().data_type
            oldHeader.image_type         = ismrmrd.IMTYPE_MAGNITUDE
            oldHeader.image_index        = sl + 1
            oldHeader.image_series_index = seriesIndex
            img.setHead(oldHeader)

            tmpMeta = ismrmrd.Meta.deserialize(templateImg.attribute_string)
            tmpMeta['DataRole']                      = 'Image'
            tmpMeta['ImageProcessingHistory']        = processingHistory
            # Computed/derived data, not the original acquisition -- override the
            # inherited magnitude template's ImageType (['ORIGINAL', 'PRIMARY', 'M', ...])
            # rather than passing it through unchanged.
            tmpMeta['ImageType']                     = ['DERIVED', 'SECONDARY', 'M']
            tmpMeta['SequenceDescriptionAdditional'] = seqDescAdditional
            tmpMeta['ImageComments']                 = imageComments
            # WindowCenter/WindowWidth are in real-world (rescaled) units per DICOM convention
            # -- VOI windowing is applied after the RescaleSlope/Intercept (Modality LUT)
            # transform, not to the raw quantized pixel values. Confirmed on a real scanner
            # export that a fractional WindowWidth ('0.6') doesn't survive Open Recon's
            # Injector -- came back as WindowWidth=0 (no usable default window at all),
            # consistent with the Injector truncating it to an integer rather than applying it
            # as a real-valued DICOM DS -- pass whole numbers only.
            tmpMeta['WindowCenter']                  = "{:.0f}".format(windowCenter)
            tmpMeta['WindowWidth']                   = "{:.0f}".format(windowWidth)
            # DICOM's DS (Decimal String) value representation caps field length at 16
            # characters -- plain str(float) can exceed that (e.g. for small slopes in
            # scientific notation), so format explicitly rather than relying on repr.
            tmpMeta['RescaleSlope']                  = "{:.6e}".format(seriesRescaleSlope)
            tmpMeta['RescaleIntercept']               = "{:.6f}".format(seriesRescaleIntercept)
            tmpMeta['RescaleType']                    = rescaleType
            tmpMeta['Keep_image_geometry']            = 1

            if tmpMeta.get('ImageRowDir') is None:
                tmpMeta['ImageRowDir'] = ["{:.18f}".format(oldHeader.read_dir[0]), "{:.18f}".format(oldHeader.read_dir[1]), "{:.18f}".format(oldHeader.read_dir[2])]
            if tmpMeta.get('ImageColumnDir') is None:
                tmpMeta['ImageColumnDir'] = ["{:.18f}".format(oldHeader.phase_dir[0]), "{:.18f}".format(oldHeader.phase_dir[1]), "{:.18f}".format(oldHeader.phase_dir[2])]

            img.attribute_string = tmpMeta.serialize()
            imagesOut.append(img)

    for label, seriesIndex, seqDescAdditional, imageComments, qsmVol in qsmResults:
        chi_min = float(qsmVol.min())
        chi_max = float(qsmVol.max())
        n_clipped = int(np.sum((qsmVol < -QSM_DISPLAY_RANGE_PPM) | (qsmVol > QSM_DISPLAY_RANGE_PPM)))
        if n_clipped > 0:
            logging.warning("(%s) %d of %d voxels fell outside +/-%.1f ppm and were clipped",
                             label, n_clipped, qsmVol.size, QSM_DISPLAY_RANGE_PPM)
        logging.info("(%s) QSM value range [%.4f, %.4f] ppm -> quantized to uint16 over fixed "
                     "[-%.1f, %.1f] ppm (RescaleSlope=%.8g, RescaleIntercept=%.4f)",
                     label, chi_min, chi_max, QSM_DISPLAY_RANGE_PPM, QSM_DISPLAY_RANGE_PPM,
                     rescaleSlope, rescaleIntercept)
        _append_dicom_series(qsmVol, seriesIndex, seqDescAdditional, imageComments,
                              dicomRescaleSlope, dicomRescaleIntercept, 'PPB',
                              windowCenter=0.0, windowWidth=1.0 * DICOM_UNIT_SCALE,
                              pixelMax=QSM_PIXEL_MAX, processingHistory=['PYTHON', 'IQSM_PLUS'])

    for label, seriesIndex, seqDescAdditional, imageComments, r2sVol in r2sResults:
        r2s_min = float(r2sVol.min())
        r2s_max = float(r2sVol.max())
        n_clipped = int(np.sum(r2sVol > R2S_DISPLAY_RANGE_MAX))
        if n_clipped > 0:
            logging.warning("(%s) %d of %d voxels exceeded %.1f s^-1 and were clipped",
                             label, n_clipped, r2sVol.size, R2S_DISPLAY_RANGE_MAX)
        logging.info("(%s) R2* value range [%.4f, %.4f] s^-1 -> quantized to uint16 over fixed "
                     "[0, %.1f] s^-1 (RescaleSlope=%.8g)",
                     label, r2s_min, r2s_max, R2S_DISPLAY_RANGE_MAX, r2sRescaleSlope)
        _append_dicom_series(r2sVol, seriesIndex, seqDescAdditional, imageComments,
                              r2sRescaleSlope, r2sRescaleIntercept, 'R2S',
                              windowCenter=R2S_DISPLAY_RANGE_MAX / 2, windowWidth=R2S_DISPLAY_RANGE_MAX,
                              pixelMax=R2S_PIXEL_MAX, processingHistory=['PYTHON', 'DEEPRELAXO'])

    # ------------------------------------------------------------------
    # Pass through every originally-received image (all magnitude series,
    # all echoes, phase) unmodified, as their own series alongside the new
    # QSM/R2* maps. Per Open Recon's documented behavior, only images explicitly
    # returned by the app are saved to DICOM/displayed on the scanner -- the
    # standard ICE-reconstructed images are NOT automatically preserved
    # (see or_sdk/README.md: "only images that are returned by the Open
    # Recon app are saved to DICOMs ... standard Siemens reconstructed
    # images are not automatically saved"). Without this, the original
    # acquisition series would simply be discarded, not left untouched.
    # These are the exact objects received from the Emitter -- only their
    # .data/.attribute_string were read (never mutated) when building the
    # QSM/R2* volumes/output above, so returning them here reproduces the same
    # DICOMs the scanner would have produced without Open Recon involved.
    # ------------------------------------------------------------------
    nDerivedImages = len(imagesOut)
    imagesOut.extend(buffer.values())

    logging.info("Returning %d QSM/R2* image(s) (%d QSM series, %d R2* series) + "
                 "%d original image(s) = %d total",
                 nDerivedImages, len(qsmResults), len(r2sResults), len(buffer), len(imagesOut))
    return imagesOut
