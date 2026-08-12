"""
Open Recon "image to image" module that reconstructs QSM (via iQSM+,
https://github.com/sunhongfu/iQSM_Plus) and R2* maps (via DeepRelaxo,
https://github.com/sunhongfu/DeepRelaxo) from incoming multi-echo GRE
magnitude+phase images.

Unlike the simpler example modules (invertcontrast.py, i2i.py), this needs the
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
import subprocess
import numpy as np
import nibabel as nib
import mrdhelper
import constants
from time import perf_counter

# Folder for debug output files
debugFolder = "/tmp/share/debug"

# ==============================================================================
# External dependencies: iQSM_Plus, DeepRelaxo, and bet2 all live outside this
# repo (or as vendored binaries) rather than being tracked in it -- see
# readme.md's "Building the Docker image" section for how each gets there.
# Each is resolved via the same fallback chain: an env var override, then
# where it lands in the Docker image, then a path relative to this file (a
# local checkout -- devcontainer, native venv, or plain git clone) -- so
# nothing here needs to be edited per-environment.
# ==============================================================================

_IQSM_PLUS_CANDIDATES = [
    os.environ.get("IQSM_PLUS_DIR"),
    "/opt/code/python-ismrmrd-server/iQSM_Plus",                              # Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "iQSM_Plus"),    # local checkout
]
IQSM_PLUS_DIR = next((p for p in _IQSM_PLUS_CANDIDATES if p and os.path.isdir(p)), None)
if IQSM_PLUS_DIR and IQSM_PLUS_DIR not in sys.path:
    sys.path.insert(0, IQSM_PLUS_DIR)

# bet2 (FSL's Brain Extraction Tool) is vendored directly at vendor/bet2/ (bin + its ~15
# FSL-specific runtime shared libraries, not a full FSL install -- see
# vendor/bet2/README.md for provenance/license).
_BET2_CANDIDATES = [
    os.environ.get("BET2_DIR"),
    "/opt/bet2",                                                                    # Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor", "bet2"),     # local checkout
]
BET2_DIR = next((p for p in _BET2_CANDIDATES
                 if p and os.path.isfile(os.path.join(p, "bin", "bet2"))), None)

_DEEPRELAXO_CANDIDATES = [
    os.environ.get("DEEPRELAXO_DIR"),
    "/opt/code/python-ismrmrd-server/DeepRelaxo",                              # Docker image
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepRelaxo"),    # local checkout
]
DEEPRELAXO_DIR = next((p for p in _DEEPRELAXO_CANDIDATES if p and os.path.isdir(p)), None)


def _import_deeprelaxo():
    """Import DeepRelaxo's estimate_r2s/denoise_r2s_map, isolated from iQSM_Plus's own
    same-named data_utils.py module. Both repos happen to define a bare top-level
    `data_utils.py` (unrelated repos, coincidentally similar layout) with different
    functions -- Python caches imports by module name, so if both directories were on
    sys.path at once, whichever repo's data_utils.py got imported first would silently
    shadow the other's for the rest of the process (a wrong-answer bug, not an
    ImportError, since both define same-named functions). DEEPRELAXO_DIR is therefore
    deliberately NOT added to sys.path at module load time like IQSM_PLUS_DIR above --
    instead, right before the one place DeepRelaxo is actually imported, this pops any
    already-cached `data_utils` and forces DEEPRELAXO_DIR to the front of sys.path,
    guaranteeing DeepRelaxo's own file wins regardless of import order elsewhere.
    """
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


# ==============================================================================
# Open Recon UI parameters (qsm_json_ui.json)
# ==============================================================================

def _get_ui_param(config, key, default):
    """Safely read one UI parameter from the Open Recon config dict. server.py passes
    the full parsed JSON config dict (not just the resolved config string) through as
    `config` -- see server.py's `configAdditional` / `module.process(connection,
    configAdditional, metadata)`. Falls back to `default` if config isn't in that dict
    shape, e.g. when testing locally via client.py without a JSON config message."""
    try:
        return config['parameters'].get(key, default)
    except (TypeError, KeyError, AttributeError):
        return default


def _get_recon_mode(config):
    """Which mask variant to reconstruct: 'masked' (default, brain-extracted via bet2) or
    'wholehead'. Applies to both QSM and R2* -- there is no longer a way to get both
    variants from a single exam; re-run with the other mode if needed."""
    value = str(_get_ui_param(config, 'reconmode', 'masked')).strip().lower()
    return value if value in ('masked', 'wholehead') else 'masked'


def _get_bool_ui_param(config, key, default):
    """Read a boolean UI parameter. Does not just bool(...) the raw value -- Open Recon's
    Injector has been observed to send boolean UI parameters as the *string* "false", and
    bool("false") is True in Python."""
    value = _get_ui_param(config, key, default)
    if isinstance(value, str):
        return value.strip().lower() not in ('false', '0', '')
    return bool(value)


# ==============================================================================
# Per-image metadata extraction
# ==============================================================================

def _phase_to_radians(raw, meta):
    """Convert Siemens integer phase pixel values to radians."""
    slope     = mrdhelper.get_meta_value(meta, 'RescaleSlope')
    intercept = mrdhelper.get_meta_value(meta, 'RescaleIntercept')

    if (slope is not None) and (intercept is not None):
        rescaled = raw.astype(np.float64) * float(slope) + float(intercept)
    else:
        # Real-time ICE images may not carry DICOM-style rescale tags. Fall back to
        # Siemens' standard 12-bit convention (0..4095 representing -pi..+pi) and log the
        # raw range so this assumption can be verified against real streamed data.
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
    # MetaAttributes added by dicom2mrd.py won't be present. Derive from the standard MRD
    # header fields instead. field_of_view.x/.y are the *full in-plane* FOV (pixel_spacing
    # * matrix_size -- see dicom2mrd.py's CalcFieldOfView), so dividing by matrixSize.x/.y
    # recovers pixel spacing. field_of_view.z, however, is already the slice thickness
    # itself (CalcFieldOfView stores SliceThickness directly, not SliceThickness *
    # nSlices) -- it must NOT be divided by matrixSize.z (the partition/slice count)
    # again, or the resulting near-zero voxel size feeds a bogus isotropic-interpolation
    # target into iQSM+ that tries to upsample the in-plane axes ~200x and exhausts
    # available memory.
    fov = img.field_of_view
    mtx = metadata.encoding[0].reconSpace.matrixSize
    voxel_mm = [float(fov[1]) / float(mtx.y), float(fov[0]) / float(mtx.x), float(fov[2])]
    logging.info("voxel_size_mm source: no DICOM PixelSpacing/SliceThickness meta -- "
                 "fallback from field_of_view=%s, reconSpace.matrixSize=(%s,%s,%s) -> %s",
                 list(fov), mtx.x, mtx.y, mtx.z, voxel_mm)
    return voxel_mm


def _get_b0_dir(img):
    """Approximate B0 direction in image-axis space [row/phase-dir, col/read-dir,
    slice-dir]. B0 is always along the patient's superior-inferior (world z, in the same
    LPS patient coordinate system used by read_dir/phase_dir/slice_dir) axis, so project
    world-z onto the image's own axes."""
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


def _select_magnitude_series(buffer):
    """Pick exactly one magnitude image_series_index from the buffer, or None if there
    are no magnitude images at all. Some ME-GRE protocols emit *two* magnitude series
    (e.g. a raw/uncorrected one and a B1/intensity-corrected one) alongside phase -- both
    map to image_type == IMTYPE_MAGNITUDE, so without disambiguation they'd silently
    collide on the same (slice, contrast) buffer key and whichever was processed last
    would win arbitrarily. Prefers whichever series carries Siemens' 'NORM' ImageType
    flag (PreScan Normalize / B1-intensity-corrected); falls back to the lowest
    image_series_index if that flag isn't present or isn't unambiguous (e.g. real-time
    ICE images that never carry DICOM-style ImageType metadata at all).

    Only affects which magnitude series feeds the multi-echo weighted combination
    (weights = (mag*TE)**2, see _reconstruct_qsm) -- a purely spatial per-voxel
    multiplicative correction like B1 intensity correction cancels out of that weighted
    average, so which series is picked shouldn't change the QSM result. Set
    QSM_MAGNITUDE_SERIES_INDEX to force a specific series if that assumption ever proves
    wrong for a given protocol.
    """
    allMagSeries = sorted(set(si for (t, si, sl, ct) in buffer.keys() if t == ismrmrd.IMTYPE_MAGNITUDE))

    forced = os.environ.get("QSM_MAGNITUDE_SERIES_INDEX")
    if forced is not None:
        return int(forced)
    if len(allMagSeries) <= 1:
        return allMagSeries[0] if allMagSeries else None

    def has_norm_flag(seriesIndex):
        for (t, si, sl, ct), img in buffer.items():
            if t == ismrmrd.IMTYPE_MAGNITUDE and si == seriesIndex:
                imageType = mrdhelper.get_meta_value(ismrmrd.Meta.deserialize(img.attribute_string), 'ImageType')
                # ismrmrd.Meta.deserialize() returns a list when ImageType has multiple
                # components (the normal case, e.g. ['ORIGINAL', 'PRIMARY', 'M', 'NORM', ...])
                # but collapses to a plain string when there's only one -- checking 'NORM' in
                # imageType would then do a *substring* match (e.g. falsely matching 'NORMAL'),
                # so require an exact token match either way.
                if isinstance(imageType, list):
                    return 'NORM' in imageType
                return imageType == 'NORM'
        return False

    normSeries = [si for si in allMagSeries if has_norm_flag(si)]
    if len(normSeries) == 1:
        magSeriesIndex, reason = normSeries[0], "carries Siemens' 'NORM' ImageType flag (B1/intensity-corrected)"
    else:
        magSeriesIndex, reason = allMagSeries[0], "no unambiguous 'NORM' flag found -- defaulting to lowest image_series_index"
    logging.warning("Received magnitude images from %d distinct series (image_series_index=%s) "
                     "-- this protocol likely emits both a raw and a B1/intensity-corrected "
                     "magnitude series. Using image_series_index=%d (%s); set "
                     "QSM_MAGNITUDE_SERIES_INDEX to override.",
                     len(allMagSeries), allMagSeries, magSeriesIndex, reason)
    return magSeriesIndex


def _build_volumes(magKeys, phaseKeys, nSlices, nEchoes):
    """Assemble the 4D (row, col, slice, echo) magnitude and phase volumes from the
    per-image buffers. Phase is converted to radians; missing (slice, echo) phase images
    are left as zero (a warning is logged). Returns (magVol, phaseVol, sampleImg), where
    sampleImg is an arbitrary phase image kept around for its geometry/orientation
    metadata."""
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

    return magVol, phaseVol, sampleImg


def _run_bet2(mag_nii_path, output_dir, fractional_intensity=0.5):
    """Run FSL's bet2 on a magnitude volume, returning the path to the binary brain mask
    NIfTI, or None if bet2 isn't available or the run fails. A failure here should never
    abort the whole reconstruction, only skip masking."""
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


def _fix_passthrough_orientation(img):
    """Correct a real, reproducible orientation bug in the pass-through magnitude/phase
    images: on real scanner hardware, they display rotated 180 degrees relative to the
    QSM/R2* output from the same exam, which is correctly oriented (confirmed against real
    hardware via known subject positioning) -- and confirmed to display correctly when the
    same acquisition is reconstructed *without* Open Recon at all, ruling out anything
    upstream of the Open Recon round-trip (the raw acquisition itself, ICE's
    reconstruction) as the cause.

    Root cause: the Injector's DICOM conversion applies its own orientation handling to
    any image unless the app explicitly opts out via the 'Keep_image_geometry' Meta
    attribute -- documented and used throughout this same python-ismrmrd-server framework
    (see e.g. invertcontrast.py's 'sendOriginal' feature, which does exactly this for its
    own pass-through images: "Ensure Keep_image_geometry is set to not reverse image
    orientation"). qsm.py's _append_dicom_series() already sets this for QSM/R2* (which is
    exactly why they're correctly oriented) but never touched pass-through images'
    attribute_string at all -- so whatever the Emitter originally sent (without this flag)
    went straight back to the Injector unchanged, and the Injector's default reversal
    applied. This is why the underlying pixel data itself was never actually wrong (an
    earlier version of this fix incorrectly rotated the pixel data instead, based on
    reasoning that didn't account for this) -- only the missing flag was.

    Mutates `img` in place (attribute_string only) -- pixel data and every other header
    field are left untouched.
    """
    meta = ismrmrd.Meta.deserialize(img.attribute_string)
    meta['Keep_image_geometry'] = 1
    img.attribute_string = meta.serialize()


# ==============================================================================
# Connection entrypoint
# ==============================================================================

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    try:
        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Buffer every incoming image, keyed by (image_type, image_series_index, slice,
    # contrast). image_series_index is included because some ME-GRE protocols emit *two*
    # magnitude series that otherwise share identical (image_type, slice, contrast) --
    # without it, one series would silently overwrite the other here, before
    # process_qsm() ever gets a chance to notice or choose between them (see
    # _select_magnitude_series).
    #
    # Streaming/order note: images arrive one at a time, in whatever order ICE finishes
    # them -- not necessarily grouped or interleaved predictably. QSM needs the complete
    # 3D+echo volume before the network can run at all, so instead of trying to detect
    # "series complete" mid-stream, this simply accumulates everything and processes once
    # the connection closes (item is None).
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
            # process_qsm() can fail anywhere inside (bet2, iQSM+, DeepRelaxo, quantization,
            # ...) -- isolated in its own try/except so a reconstruction failure still
            # returns the originally-received images rather than nothing at all. Without
            # this, a QSM/R2* crash would silently discard the acquisition's own magnitude/
            # phase images too, since they're normally only sent back as part of
            # process_qsm()'s own return value (see its "Pass through" comment) -- an
            # OpenRecon app failure should degrade to "no QSM/R2* this exam", never to
            # "no images from this exam at all".
            try:
                imagesOut = process_qsm(buffer, connection, config, metadata)
            except Exception:
                logging.error("QSM/R2* reconstruction failed -- returning original images "
                              "only (no derived QSM/R2* output):\n%s", traceback.format_exc())
                connection.send_logging(
                    constants.MRD_LOGGING_ERROR,
                    "QSM/R2* reconstruction failed -- original acquisition images are still "
                    "returned. See server log for details.")
                imagesOut = list(buffer.values())
            if imagesOut:
                connection.send_image(imagesOut)
        else:
            logging.warning("No images received -- nothing to process")

    except Exception:
        logging.error(traceback.format_exc())
        connection.send_logging(constants.MRD_LOGGING_ERROR, traceback.format_exc())
        # Close connection without sending MRD_MESSAGE_CLOSE message to signal failure
        connection.shutdown_close()

    finally:
        try:
            connection.send_close()
        except:
            logging.error("Failed to send close message!")


# ==============================================================================
# Main reconstruction
# ==============================================================================

def process_qsm(buffer, connection, config, metadata):
    tic = perf_counter()

    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    magSeriesIndex = _select_magnitude_series(buffer)
    magKeys   = {(sl, ct): img for (t, si, sl, ct), img in buffer.items()
                 if t == ismrmrd.IMTYPE_MAGNITUDE and si == magSeriesIndex}
    phaseKeys = {(sl, ct): img for (t, si, sl, ct), img in buffer.items() if t == ismrmrd.IMTYPE_PHASE}

    if len(phaseKeys) == 0:
        raise Exception("No phase images received -- QSM requires magnitude and phase images")

    nSlices = max(sl for (sl, ct) in phaseKeys.keys()) + 1
    nEchoes = max(ct for (sl, ct) in phaseKeys.keys()) + 1
    logging.info("Buffered %d magnitude and %d phase images -- expecting %d slices x %d echoes",
                 len(magKeys), len(phaseKeys), nSlices, nEchoes)

    magVol, phaseVol, sampleImg = _build_volumes(magKeys, phaseKeys, nSlices, nEchoes)
    logging.info("Buffered volumes: magVol shape=%s dtype=%s (%.1f MB), phaseVol shape=%s dtype=%s (%.1f MB)",
                 magVol.shape, magVol.dtype, magVol.nbytes / (1024 ** 2),
                 phaseVol.shape, phaseVol.dtype, phaseVol.nbytes / (1024 ** 2))
    np.save(os.path.join(debugFolder, "magVol.npy"), magVol)
    np.save(os.path.join(debugFolder, "phaseVol_rad.npy"), phaseVol)

    # ------------------------------------------------------------------
    # Acquisition parameters
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

    logging.info("Acquisition parameters: voxel_size_mm=%s, b0_dir=%s, b0=%.2fT, TE(s)=%s, "
                 "n_slices=%d, n_echoes=%d",
                 voxel_mm, b0_dir, b0_tesla, te_sec, nSlices, nEchoes)

    try:
        from inference import run_iqsm_plus, CheckpointNotFoundError
    except ImportError as e:
        raise Exception("Could not import iQSM+ from IQSM_PLUS_DIR='%s' (%s). Set the "
                         "IQSM_PLUS_DIR environment variable to a valid iQSM_Plus checkout." %
                         (IQSM_PLUS_DIR, e))

    affine = np.diag(voxel_mm + [1.0])

    # ------------------------------------------------------------------
    # Which mask variant to reconstruct, per the 'Reconstruction Mode' UI parameter
    # (default 'masked'). Applies to both QSM and R2* below -- a single exam only ever
    # produces one variant; re-run with the other mode (e.g. via retro-recon, if your
    # OpenRecon deployment supports it) to get the other one.
    # ------------------------------------------------------------------
    reconMode = _get_recon_mode(config)
    useMask = reconMode == 'masked'

    # QSM defaults on, R2* defaults off -- R2* mapping (DeepRelaxo) takes noticeably
    # longer than QSM (iQSM+) alone, so it's opt-in per exam rather than always-on;
    # enable it via retro-recon afterward if it turns out to be needed.
    qsmEnabled = _get_bool_ui_param(config, 'qsmenabled', True)
    r2sEnabled = _get_bool_ui_param(config, 'r2smapping', False)
    if not qsmEnabled and not r2sEnabled:
        logging.warning("Both QSM and R2* mapping are disabled via UI parameters -- only "
                        "the original acquisition images will be returned")

    maskPath = None
    if useMask:
        # bet2 needs a plain 3D volume, not the 4D multi-echo array -- run it on the
        # first echo only, the resulting mask is reused for every echo below.
        mag3dPath = os.path.join(debugFolder, "mag_echo0_for_bet2.nii.gz")
        nib.save(nib.Nifti1Image(magVol[..., 0], affine), mag3dPath)
        maskPath = _run_bet2(mag3dPath, debugFolder)
        if maskPath is None:
            logging.warning("Brain-masked reconstruction was requested but bet2 failed/was "
                             "unavailable -- falling back to whole-head")
            useMask = False
    else:
        logging.info("Whole-head reconstruction mode selected -- skipping brain extraction")

    # image_series_index=100/102 (QSM/R2*) are reserved for the brain-masked variant
    # specifically -- 100 is the series index this app has always used for QSM, and
    # existing tooling (readme.md, RunQSMRecon.ipynb) assumes it. 101/103 are whole-head.
    if useMask:
        label, qsmSeriesIndex, r2sSeriesIndex = "masked", 100, 102
        qsmSeqDesc, r2sSeqDesc = "QSM_MASKED", "R2S_MASKED"
        qsmComment = "QSM brain-extracted (ppb = ppm*1e3), iQSM+"
        r2sComment = "R2* brain-extracted (s^-1), DeepRelaxo"
    else:
        label, qsmSeriesIndex, r2sSeriesIndex = "wholehead", 101, 103
        qsmSeqDesc, r2sSeqDesc = "QSM_WHOLEHEAD", "R2S_WHOLEHEAD"
        qsmComment = "QSM whole-head (ppb = ppm*1e3), iQSM+"
        r2sComment = "R2* whole-head (s^-1), DeepRelaxo"

    # Written unconditionally (regardless of qsmEnabled/r2sEnabled) since R2* mapping
    # (DeepRelaxo) needs mag_echo{N}.nii.gz even when QSM itself is disabled -- previously
    # these were only written as a side effect inside _reconstruct_qsm().
    for echo in range(nEchoes):
        nib.save(nib.Nifti1Image(phaseVol[..., echo], affine),
                 os.path.join(debugFolder, "phase_echo%d.nii.gz" % echo))
        nib.save(nib.Nifti1Image(magVol[..., echo], affine),
                 os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo))

    def _reconstruct_qsm(maskNiiPath):
        """Run iQSM+ across all echoes with the given brain mask (None for whole-head,
        unmasked reconstruction) and combine echoes into one susceptibility volume (ppm).

        run_iqsm_plus() processes a single echo per call -- multi-echo combination is
        handled externally here, exactly as iQSM_Plus's own run.py/app.py do. Model
        weights are cached globally inside get_model() (keyed by device), so looping here
        does not reload them from disk each iteration.
        """
        inferenceStart = perf_counter()
        qsmVolumes = []
        try:
            for echo in range(nEchoes):
                echoPhasePath = os.path.join(debugFolder, "phase_echo%d.nii.gz" % echo)
                echoMagPath   = os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo)

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

        # TE^2-weighted average across echoes -- a fixed per-echo weight (no magnitude
        # weighting), so every voxel is combined the same way regardless of local signal
        # strength.
        qsmStack = np.stack(qsmVolumes, axis=-1)
        teWeights = np.array(te_sec, dtype=np.float32) ** 2
        qsmVol = (qsmStack * teWeights.reshape(1, 1, 1, -1)).sum(axis=-1) / teWeights.sum()

        # The network's raw per-echo output can still independently contain a NaN/Inf at
        # some voxels (observed in practice at the extreme top/bottom Z-slices of a real
        # scan, plausibly interacting with the network's own boundary-zeroing behavior --
        # see LoTLayer's LG() in iQSM_Plus/models/unet_blocks.py). Left unguarded, that NaN
        # survives all the way to uint16 quantization, where `.astype(np.uint16)` on NaN is
        # undefined behavior in C -- confirmed to silently produce different garbage per
        # platform (0 on macOS/arm64, 32768 on Linux/x86_64) rather than raising, corrupting
        # DICOM slices with no exception or warning anywhere.
        badVoxels = ~np.isfinite(qsmVol)
        nBad = int(badVoxels.sum())
        if nBad > 0:
            logging.warning("Multi-echo combination (%s): %d of %d voxels were NaN/Inf after "
                            "combination (network produced a non-finite value) -- set to 0 ppm",
                            label, nBad, qsmVol.size)
            qsmVol[badVoxels] = 0.0

        np.save(os.path.join(debugFolder, "qsmVol_%s.npy" % label), qsmVol)
        return qsmVol

    # (label, image_series_index, SequenceDescriptionAdditional, ImageComments, qsmVol_ppm)
    qsmResults = []
    if qsmEnabled:
        qsmResults = [(label, qsmSeriesIndex, qsmSeqDesc, qsmComment, _reconstruct_qsm(maskPath))]
    else:
        logging.info("QSM reconstruction skipped (disabled via UI parameter)")

    def _run_r2s_mapping():
        """R2* mapping (DeepRelaxo), using the same mask variant as QSM above and the
        mag_echo{N}.nii.gz files written above (no separate brain-extraction run).
        Opt-in via the 'r2smapping' UI parameter (default off -- it takes noticeably
        longer than QSM alone) and requires at least 2 echoes to fit a decay curve.
        Failures here are non-fatal to the QSM output -- caught broadly (covers e.g. a
        missing DeepRelaxo checkout/checkpoints) and logged, so a broken R2* dependency
        never blocks the QSM reconstruction the rest of this module already validated.
        """
        r2sResults = []  # (label, image_series_index, SequenceDescriptionAdditional, ImageComments, r2sVol)
        if not r2sEnabled:
            logging.info("R2* mapping skipped (disabled via UI parameter)")
            return r2sResults
        if nEchoes < 2:
            logging.info("R2* mapping skipped -- requires at least 2 echoes (got %d)", nEchoes)
            return r2sResults

        try:
            estimate_r2s, denoise_r2s_map = _import_deeprelaxo()
            magnitude_entries = [{"path": os.path.join(debugFolder, "mag_echo%d.nii.gz" % echo)}
                                  for echo in range(nEchoes)]
            te_values_ms = [t * 1000.0 for t in te_sec]

            r2sTic = perf_counter()
            r2sMap, _ = estimate_r2s(magnitude_entries=magnitude_entries, te_values_ms=te_values_ms,
                                      bet_mask_path=maskPath if useMask else None)
            r2sMap = r2sMap.numpy()
            maskArr = (nib.load(maskPath).get_fdata() > 0) if useMask else np.ones(r2sMap.shape, dtype=bool)
            r2sDenoised = denoise_r2s_map(r2sMap, maskArr)
            logging.info("DeepRelaxo (%s) completed in %.1f s", label, perf_counter() - r2sTic)
            r2sResults.append((label, r2sSeriesIndex, r2sSeqDesc, r2sComment, r2sDenoised))
        except Exception:
            logging.error("R2* mapping (DeepRelaxo) failed -- continuing without R2* output:\n%s",
                          traceback.format_exc())
        return r2sResults

    r2sResults = _run_r2s_mapping()

    toc = perf_counter()
    strProcessTime = "QSM reconstruction time: %.2f s" % (toc - tic)
    logging.info(strProcessTime)
    connection.send_logging(constants.MRD_LOGGING_INFO, strProcessTime)

    # ------------------------------------------------------------------
    # Quantize each 3D map into uint16 DICOM pixel data and re-slice into individual 2D
    # MRD images, matching the granularity ICE expects for its per-slice DICOM pipeline.
    # DICOM's PixelData is fundamentally integer-only -- sending raw float32 values (as
    # MRD itself allows) gets misinterpreted as garbage integers once converted to DICOM.
    #
    # A *fixed* clinical range (rather than one computed per-scan from that scan's own
    # min/max) is used so the same real value always maps to the same raw pixel count
    # across every reconstruction. Values are stored *unsigned* because mrd2dicom.py
    # always defaults PixelRepresentation to 0 (unsigned) regardless of the source data's
    # sign. PIXEL_MAX is 4095 (12-bit), not 65535 (16-bit): Open Recon's Injector writes
    # MR-derived images as BitsAllocated=16 but BitsStored=12/HighBit=11 -- a fixed
    # Siemens convention for this image type -- so values above 4095 don't get
    # clipped/rescaled, they get silently truncated to their low 12 bits (a near-zero
    # value can encode to raw~32768 = 0x8000, whose low 12 bits are 0, collapsing most of
    # the image to exactly RescaleIntercept on real hardware). Quantizing into 0..4095 up
    # front means the values already fit inside whatever the Injector actually preserves.
    # ------------------------------------------------------------------
    QSM_DISPLAY_RANGE_PPM = 4.0  # clip/quantize over [-4, +4] ppm
    QSM_PIXEL_MAX = 4095
    qsmQuantSlope     = (2.0 * QSM_DISPLAY_RANGE_PPM) / QSM_PIXEL_MAX
    qsmQuantIntercept = -QSM_DISPLAY_RANGE_PPM

    # The DICOM-facing RescaleSlope/Intercept (and WindowCenter/Width) report values in
    # ppb (ppm * 1000) rather than ppm directly: a scanner's window/level tool has been
    # observed to have coarse (effectively integer-only) granularity in whatever
    # real-world unit RescaleSlope/Intercept report, so a genuine ppm-scale window (e.g.
    # WindowWidth=1) only has a handful of distinct adjustable positions across the whole
    # clinically-relevant range -- nowhere near enough for usable mouse-based adjustment
    # (compare magnitude images, whose real-world W/L values are naturally in the
    # hundreds-to-thousands, since RescaleSlope~1 there). This changes nothing about the
    # underlying pixel data or the true value any voxel represents, only the *units* a
    # DICOM viewer reports/adjusts it in -- RescaleType is set to PPB so anything that
    # respects it knows the units changed.
    #
    # This is why QSM's DICOM tags (dicomSlope/dicomIntercept below) differ from what it
    # quantizes with (qsmQuantSlope/qsmQuantIntercept above) -- _append_dicom_series keeps
    # these as two explicit, separately-named parameter pairs specifically so they can
    # never be silently conflated: the pixel data is always quantized in the volume's own
    # native units, and the DICOM tags separately declare whatever units should be
    # reported, which a compliant viewer combines correctly regardless of the difference.
    QSM_DICOM_UNIT_SCALE = 1000.0  # ppm -> ppb, tags only
    qsmDicomSlope     = qsmQuantSlope * QSM_DICOM_UNIT_SCALE
    qsmDicomIntercept = qsmQuantIntercept * QSM_DICOM_UNIT_SCALE

    # R2* is non-negative and naturally spans tens-to-~100+ s^-1 in brain tissue, unlike
    # QSM's tiny +/-4ppm range, so no unit-inflation trick is needed for a scanner W/L tool
    # to have usable granularity -- reporting directly in s^-1 already gives a comparably
    # large integer range to magnitude images. 250 s^-1 was chosen after a real test run
    # showed values up to ~226 s^-1 (whole-head, uncropped background/edge voxels
    # included); a 100 s^-1 ceiling clipped ~6% of whole-head voxels. Tune to your own
    # clinical experience if needed.
    R2S_DISPLAY_RANGE_MAX = 250.0  # clip/quantize over [0, 250] s^-1
    R2S_PIXEL_MAX = 4095
    r2sQuantSlope     = R2S_DISPLAY_RANGE_MAX / R2S_PIXEL_MAX
    r2sQuantIntercept = 0.0

    imagesOut = []

    def _append_dicom_series(vol, seriesIndex, seqDescAdditional, imageComments,
                              quantSlope, quantIntercept, rescaleType,
                              windowCenter, windowWidth, pixelMax, processingHistory,
                              dicomSlope=None, dicomIntercept=None):
        """Quantize `vol` (in its own natural units) into uint16 using quantSlope/
        quantIntercept, and build one MRD image per slice. dicomSlope/dicomIntercept --
        the values actually written into the DICOM RescaleSlope/RescaleIntercept tags --
        default to quantSlope/quantIntercept (the common case: R2*, where the pixel data
        and the reported units are the same) but can be given separately when the DICOM
        tags intentionally report different units than the data was quantized in (QSM;
        see the ppb comment above quantSlope's definition). Keeping these as two distinct,
        explicitly-named parameter pairs -- rather than one pair implicitly serving both
        purposes -- is deliberate: passing data quantized in one unit against tags
        declared in another is exactly the bug that once destroyed all QSM contrast (every
        voxel quantized to one of two raw codes at the zero-crossing) with no error or
        warning anywhere.
        """
        if dicomSlope is None:
            dicomSlope = quantSlope
        if dicomIntercept is None:
            dicomIntercept = quantIntercept

        # `.astype(np.uint16)` on a NaN/Inf value is undefined behavior in C -- confirmed
        # to silently differ by platform rather than raising, so this guarantees every
        # series (QSM and R2* alike) reaches the cast already finite, regardless of which
        # upstream stage might introduce a NaN -- including DeepRelaxo's denoiser, which
        # (unlike its estimator) has no NaN guard of its own on its output.
        nBad = int((~np.isfinite(vol)).sum())
        if nBad > 0:
            logging.warning("(series %d) %d of %d voxels were NaN/Inf before quantization -- "
                            "set to 0", seriesIndex, nBad, vol.size)
            vol = np.nan_to_num(vol, nan=0.0, posinf=0.0, neginf=0.0)

        volQuantized = np.clip(np.round((vol - quantIntercept) / quantSlope), 0, pixelMax).astype(np.uint16)

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
            # WindowCenter/WindowWidth are in real-world (rescaled) units per DICOM
            # convention -- VOI windowing is applied after the RescaleSlope/Intercept
            # (Modality LUT) transform, not to the raw quantized pixel values. A
            # fractional WindowWidth has been observed not to survive Open Recon's
            # Injector (truncated to an integer rather than applied as a real-valued
            # DICOM DS), so only whole numbers are passed here.
            tmpMeta['WindowCenter']                  = "{:.0f}".format(windowCenter)
            tmpMeta['WindowWidth']                   = "{:.0f}".format(windowWidth)
            # DICOM's DS (Decimal String) value representation caps field length at 16
            # characters -- plain str(float) can exceed that (e.g. for small slopes in
            # scientific notation), so format explicitly rather than relying on repr.
            tmpMeta['RescaleSlope']                  = "{:.6e}".format(dicomSlope)
            tmpMeta['RescaleIntercept']               = "{:.6f}".format(dicomIntercept)
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
                     qsmQuantSlope, qsmQuantIntercept)
        _append_dicom_series(qsmVol, seriesIndex, seqDescAdditional, imageComments,
                              quantSlope=qsmQuantSlope, quantIntercept=qsmQuantIntercept, rescaleType='PPB',
                              windowCenter=0.0, windowWidth=1.0 * QSM_DICOM_UNIT_SCALE,
                              pixelMax=QSM_PIXEL_MAX, processingHistory=['PYTHON', 'IQSM_PLUS'],
                              dicomSlope=qsmDicomSlope, dicomIntercept=qsmDicomIntercept)

    for label, seriesIndex, seqDescAdditional, imageComments, r2sVol in r2sResults:
        r2s_min = float(r2sVol.min())
        r2s_max = float(r2sVol.max())
        n_clipped = int(np.sum(r2sVol > R2S_DISPLAY_RANGE_MAX))
        if n_clipped > 0:
            logging.warning("(%s) %d of %d voxels exceeded %.1f s^-1 and were clipped",
                             label, n_clipped, r2sVol.size, R2S_DISPLAY_RANGE_MAX)
        logging.info("(%s) R2* value range [%.4f, %.4f] s^-1 -> quantized to uint16 over fixed "
                     "[0, %.1f] s^-1 (RescaleSlope=%.8g)",
                     label, r2s_min, r2s_max, R2S_DISPLAY_RANGE_MAX, r2sQuantSlope)
        _append_dicom_series(r2sVol, seriesIndex, seqDescAdditional, imageComments,
                              quantSlope=r2sQuantSlope, quantIntercept=r2sQuantIntercept, rescaleType='R2S',
                              windowCenter=R2S_DISPLAY_RANGE_MAX / 2, windowWidth=R2S_DISPLAY_RANGE_MAX,
                              pixelMax=R2S_PIXEL_MAX, processingHistory=['PYTHON', 'DEEPRELAXO'])

    # Pass through every originally-received image (all magnitude series, all echoes,
    # phase) as their own series alongside the new QSM/R2* maps. Per Open Recon's
    # documented behavior, only images explicitly returned by the app are saved to
    # DICOM/displayed on the scanner -- the standard ICE-reconstructed images are NOT
    # automatically preserved -- so without this, the original acquisition series would
    # simply be discarded.
    nDerivedImages = len(imagesOut)
    for img in buffer.values():
        _fix_passthrough_orientation(img)
    imagesOut.extend(buffer.values())

    logging.info("Returning %d QSM/R2* image(s) (%d QSM series, %d R2* series) + "
                 "%d original image(s) = %d total",
                 nDerivedImages, len(qsmResults), len(r2sResults), len(buffer), len(imagesOut))
    return imagesOut
