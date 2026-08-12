#!/usr/bin/env python3
"""
Standalone, offline batch QSM/R2* reconstruction -- NOT the Siemens Open Recon scanner
framework. Takes a folder of multi-echo GRE DICOMs (magnitude + phase) and writes QSM
(and R2*, if the acquisition has 2+ echoes) DICOMs to an output folder, with no
scanner/MRD-streaming/Injector-Emitter protocol involved.

Internally reuses the exact same reconstruction pipeline as the Open Recon app --
dicom2mrd.py, qsm.py's process_qsm() (via a local, loopback-only MRD server+client pair),
and mrd2dicom.py -- rather than reimplementing anything. This script is purely an
orchestration wrapper for offline/batch use (e.g. research use, testing against archived
DICOMs) outside a scanner. Each stage runs as its own subprocess, exactly as when running
these scripts by hand, so nothing here depends on their internals beyond the CLI already
used throughout this repo's own local-testing workflow.

Usage (inside the container):
    python3 offline_recon.py --input /input --output /output [--mode masked|wholehead]

Typical docker invocation (see docker/offline_recon.dockerfile):
    docker run --rm --gpus all \
        -v /path/to/dicoms:/input:ro \
        -v /path/to/output:/output \
        offline-qsm-recon --input /input --output /output --mode masked
"""
import argparse
import json
import logging
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time

REPO_DIR = os.path.dirname(os.path.abspath(__file__))


def _wait_for_port(host, port, timeout_sec, serverProc, serverLog):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        if serverProc.poll() is not None:
            raise RuntimeError(
                "Reconstruction server exited early (code %d) -- see %s" %
                (serverProc.returncode, serverLog))
        try:
            with socket.create_connection((host, port), timeout=1):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(
        "Reconstruction server did not start listening within %ds -- see %s" %
        (timeout_sec, serverLog))


def main():
    # Without this, print()'s progress messages below sit fully buffered (rather than
    # line-buffered) whenever stdout isn't a tty -- which is always true under `docker
    # run`/`docker logs` -- so they'd only appear once the whole process exits instead of
    # in real time, indistinguishable from a hang.
    sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True,
                         help="Folder of input multi-echo GRE magnitude+phase DICOMs")
    parser.add_argument('--output', required=True,
                         help="Folder to write QSM/R2* DICOMs to (created if missing)")
    parser.add_argument('--mode', choices=['masked', 'wholehead'], default='masked',
                         help="Reconstruction mode -- brain-masked (default, via FSL's "
                              "bet2) or whole-head. R2* is included automatically "
                              "whenever the input has 2+ echoes.")
    parser.add_argument('--bet-threshold', type=float, default=0.4,
                         help="bet2's fractional intensity threshold (0.0-1.0, default "
                              "0.4), only used in --mode masked. Smaller values give a "
                              "LARGER brain outline; larger values give a SMALLER/tighter "
                              "one. Useful for re-running a batch if the default mask came "
                              "out over- or under-inclusive.")
    parser.add_argument('--port', type=int, default=9002,
                         help="Internal loopback port for the local reconstruction "
                              "server (default: 9002) -- not exposed outside the "
                              "container, only used to talk to the code already used by "
                              "the Open Recon app")
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(message)s',
                         level=logging.INFO if args.verbose else logging.WARNING)

    if not os.path.isdir(args.input):
        parser.error("--input directory not found: %s" % args.input)
    if not (0.0 <= args.bet_threshold <= 1.0):
        parser.error("--bet-threshold must be between 0.0 and 1.0 (got %s)" % args.bet_threshold)
    os.makedirs(args.output, exist_ok=True)

    workDir = tempfile.mkdtemp(prefix="offline_recon_")
    inputMrd    = os.path.join(workDir, "input.mrd")
    outputMrd   = os.path.join(workDir, "output.mrd")
    serverLog   = os.path.join(workDir, "server.log")
    clientLog   = os.path.join(workDir, "client.log")
    outGroup    = "offline_recon"

    # client.py's -c/--config selects a config MODULE NAME; any parameters (here,
    # 'reconmode') are read from a JSON file named "<config>.json" in the client's
    # current working directory -- NOT via -C/--config-local, which sends a different
    # message type that only overrides the config name, not its parameters.
    #
    # qsmenabled/r2smapping are set explicitly (true/true) rather than left unset -- on
    # the scanner these default to QSM-on/R2*-off (R2* is noticeably slower, so it's
    # opt-in there), but this offline batch tool's own documented behavior has always
    # been to compute both whenever possible, and its timing/output docs assume that;
    # leaving these unset would have silently adopted the scanner's newer, different
    # defaults here instead.
    with open(os.path.join(workDir, "qsm.json"), 'w') as f:
        json.dump({"parameters": {"config": "qsm", "reconmode": args.mode,
                                   "qsmenabled": True, "r2smapping": True,
                                   "betthreshold": args.bet_threshold}}, f)

    serverProc = None
    try:
        print("Converting DICOMs from %s ..." % args.input)
        subprocess.run(
            [sys.executable, "dicom2mrd.py", args.input, "-o", inputMrd, "-g", "dataset"],
            cwd=REPO_DIR, check=True)

        print("Starting reconstruction server (mode=%s) ..." % args.mode)
        serverProc = subprocess.Popen(
            [sys.executable, "main.py", "-p", str(args.port), "-H", "127.0.0.1",
             "-d", "qsm", "-v", "-l", serverLog],
            cwd=REPO_DIR)
        _wait_for_port("127.0.0.1", args.port, timeout_sec=30, serverProc=serverProc, serverLog=serverLog)

        print("Running reconstruction -- this can take a while (GPU strongly "
              "recommended; see readme.md for expected timing) ...")
        subprocess.run(
            [sys.executable, os.path.join(REPO_DIR, "client.py"), inputMrd,
             "-p", str(args.port), "-c", "qsm",
             "-o", outputMrd, "-G", outGroup, "-v", "-l", clientLog],
            # cwd=workDir (not REPO_DIR) so client.py's "-c qsm" resolves qsm.json
            # against the workDir copy written above, not the repo's own qsm.json.
            cwd=workDir, check=True)
    finally:
        if serverProc is not None:
            serverProc.terminate()
            try:
                serverProc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                serverProc.kill()

    if not os.path.exists(outputMrd):
        raise RuntimeError("Reconstruction did not produce an output file -- see %s / %s" %
                            (serverLog, clientLog))

    print("Converting results to DICOM ...")
    subprocess.run(
        [sys.executable, "mrd2dicom.py", outputMrd, "-g", outGroup, "-o", args.output],
        cwd=REPO_DIR, check=True)

    shutil.rmtree(workDir, ignore_errors=True)
    print("Done. DICOMs written to %s" % args.output)


if __name__ == "__main__":
    main()
