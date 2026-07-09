"""
Package openrecon-qsm:prod into a distributable Open Recon .zip.

This is the macOS/bash-native equivalent of or_sdk/tooling/CreateORDockerImage.ipynb
(that notebook assumes Windows + 7-Zip + a Docker version <25). Covers steps 4-7 of the
packaging checklist: embed qsm_json_ui.json as a Docker label, docker save, convert to a
Docker-24-compatible manifest format if needed, and zip with docs.pdf.

Prerequisite: build the application image first (step 1), separately, whenever qsm.py or
its dependencies change -- this script only re-packages, it does not rebuild the app:
    docker build --platform linux/amd64 -f docker/qsm.dockerfile -t openrecon-qsm:prod .

Usage:
    python3 docker/build_openrecon_package.py
"""

import base64
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration -- edit these for future versions
# ---------------------------------------------------------------------------
REPO_ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_FILE_PATH  = os.path.join(REPO_ROOT, "qsm_json_ui.json")
SCHEMA_PATH     = os.path.join(REPO_ROOT, "OpenReconSchema_1.1.0.json")  # from Siemens' Open Recon SDK
BASE_IMAGE_NAME = "openrecon-qsm:prod"          # built separately via docker/qsm.dockerfile
DOCS_FILE       = os.path.join(REPO_ROOT, "docs.pdf")
OUTPUT_DIR      = os.path.join(REPO_ROOT, "OpenRecon_package")
DOCKER_VERSION_MAX = "25.0.0"                   # Open Recon's documented max supported Docker version


def run(cmd, **kwargs):
    print("+ " + " ".join(cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def validate_json(json_path, schema_path):
    import jsonschema
    jsonData = json.load(open(json_path))
    schemaData = json.load(open(schema_path))
    validator = jsonschema.Draft7Validator(schemaData)
    errors = list(validator.iter_errors(jsonData))
    if errors:
        for e in errors:
            print("SCHEMA ERROR:", e)
        raise SystemExit("qsm_json_ui.json failed schema validation -- aborting")
    print("JSON config is valid against the schema.")
    return jsonData


def compute_names(jsonData):
    version = jsonData["general"]["version"]
    vendor  = re.sub(r"[\W_]+", "", jsonData["general"]["vendor"])
    name    = re.sub(r"[\W_]+", "", jsonData["general"]["name"]["en"])
    image_name    = ("OpenRecon_" + vendor + "_" + name + ":V" + version).lower()
    base_filename = "OpenRecon_" + vendor + "_" + name + "_V" + version
    return image_name, base_filename


def build_labeled_image(jsonData, image_name):
    jsonString  = json.dumps(jsonData, indent=2)
    encodedJson = base64.b64encode(jsonString.encode("utf-8")).decode("utf-8")

    dockerfile_path = os.path.join(REPO_ROOT, "docker", "OpenRecon_qsm.dockerfile")
    with open(dockerfile_path, "w") as f:
        f.write(f"FROM {BASE_IMAGE_NAME}\n")
        f.write(f'LABEL "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0"="{encodedJson}"\n')

    # BuildKit insists on re-checking the registry even for purely local image names,
    # which fails/hangs if Docker Hub is briefly unreachable. The classic builder
    # (DOCKER_BUILDKIT=0) checks the local image store first -- see docker/qsm.dockerfile's
    # troubleshooting note for the same issue on a larger scale.
    env = dict(os.environ, DOCKER_BUILDKIT="0")
    run(["docker", "build", "-f", dockerfile_path, "-t", image_name, REPO_ROOT], env=env)


def docker_version_ok():
    from packaging import version as pkgversion
    out = subprocess.check_output(["docker", "--version"], text=True)
    parsed = out.split()[2].rstrip(",")
    return pkgversion.parse(parsed) < pkgversion.parse(DOCKER_VERSION_MAX)


def save_and_convert(image_name, base_filename, output_dir):
    tar_path = os.path.join(output_dir, base_filename + ".tar")
    run(["docker", "save", "-o", tar_path, image_name])

    if docker_version_ok():
        print("Docker version is compatible -- no conversion needed.")
        return tar_path

    print("Docker version exceeds Open Recon's supported maximum "
          f"({DOCKER_VERSION_MAX}) -- converting via docker:24.0-dind ...")
    oci_path = os.path.join(output_dir, base_filename + "_OciManifest.tar")
    os.rename(tar_path, oci_path)

    subprocess.run(["docker", "rm", "-f", "or_build_dind"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    run(["docker", "run", "--rm", "-d", "--privileged", "--name", "or_build_dind",
         "-v", f"{output_dir}:/share", "docker:24.0-dind"])

    import time
    time.sleep(8)

    try:
        run(["docker", "exec", "or_build_dind", "/bin/sh", "-c",
             f"docker load -i /share/{os.path.basename(oci_path)}"])
        run(["docker", "exec", "or_build_dind", "/bin/sh", "-c",
             f"docker save -o /share/{os.path.basename(tar_path)} {image_name}"])
        # The docker save above runs as root inside the dind container. On native Linux
        # Docker Engine, bind mounts don't translate UIDs (unlike Docker Desktop's Mac/
        # Windows VM layer, which smooths this over) -- so the resulting file on the host
        # comes out root-owned, and the zip step below fails with "Permission denied" for
        # the non-root host user. chown it back. hasattr guard: os.getuid()/getgid() don't
        # exist on Windows, where this isn't needed anyway (same VM-layer smoothing as Mac).
        if hasattr(os, "getuid"):
            run(["docker", "exec", "or_build_dind", "chown",
                 f"{os.getuid()}:{os.getgid()}", f"/share/{os.path.basename(tar_path)}"])
    finally:
        subprocess.run(["docker", "stop", "or_build_dind"])

    os.remove(oci_path)
    return tar_path


def package_zip(tar_path, base_filename, output_dir):
    if not os.path.isfile(DOCS_FILE):
        raise SystemExit(f"Could not find documentation file: {DOCS_FILE}")

    pdf_path = os.path.join(output_dir, base_filename + ".pdf")
    import shutil
    shutil.copy(DOCS_FILE, pdf_path)

    zip_path = os.path.join(output_dir, base_filename + ".zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)

    # -0 (store, no compression) rather than the default Deflate: Docker image layers are
    # already gzip-compressed internally, so Deflate barely shrinks the .tar but costs many
    # minutes of CPU time on a multi-GB file. -X strips extra file attributes. Both "Stored"
    # and "Deflate" satisfy Open Recon's actual requirement (avoid Deflate64); this repo's
    # zip does not implement Deflate64 at all, regardless of the -0/-9 flag used.
    #
    # Run with cwd=output_dir and basenames only, so the zip stores the .tar/.pdf as flat
    # entries at the archive root -- passing absolute paths would nest them under the full
    # host path (e.g. "Users/you/Desktop/.../file.tar"), which Open Recon can't install.
    run(["zip", "-X", "-0", os.path.basename(zip_path),
         os.path.basename(tar_path), os.path.basename(pdf_path)],
        cwd=output_dir)

    os.remove(tar_path)
    os.remove(pdf_path)
    return zip_path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("### Validating JSON config...")
    jsonData = validate_json(JSON_FILE_PATH, SCHEMA_PATH)

    image_name, base_filename = compute_names(jsonData)
    print("Docker image tag:", image_name)
    print("Base filename:", base_filename)

    print("\n### Building labeled Open Recon image...")
    build_labeled_image(jsonData, image_name)

    print("\n### Saving image to .tar (and converting format if needed)...")
    tar_path = save_and_convert(image_name, base_filename, OUTPUT_DIR)

    print("\n### Packaging into .zip...")
    zip_path = package_zip(tar_path, base_filename, OUTPUT_DIR)

    print(f"\nDone! Open Recon package ready at:\n  {zip_path}")


if __name__ == "__main__":
    main()
