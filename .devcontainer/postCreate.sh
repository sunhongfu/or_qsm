#!/bin/bash
# Runs once when the devcontainer is created (VS Code's postCreateCommand).
set -e

git config --global core.autocrlf true
pip install ipykernel torch nibabel scipy

# Runs with cwd = the workspace folder (VS Code's default for postCreateCommand). Clone
# iQSM_Plus in-place as a gitignored subfolder here, rather than to a container-only path
# like /opt/iQSM_Plus -- since the workspace folder is itself the live host bind-mount,
# this is the exact same layout docker/qsm.dockerfile expects (see its header comment),
# so the clone is reusable across both the devcontainer and a plain `docker build`, is
# live-editable from the host, and only needs to happen once (not on every re-create).
export IQSM_PLUS_DIR="$(pwd)/iQSM_Plus"
if [ ! -d "$IQSM_PLUS_DIR/.git" ]; then
    git clone https://github.com/sunhongfu/iQSM_Plus.git "$IQSM_PLUS_DIR"
fi

# Pretrained model checkpoints are hosted on Hugging Face, not committed to the git repo
# -- mirrors iQSM_Plus's own `run.py --download-checkpoints`.
mkdir -p "$IQSM_PLUS_DIR/checkpoints"
python3 -c "
import os, urllib.request
base = 'https://huggingface.co/sunhongfu/iQSM_Plus/resolve/main'
ckpt_dir = os.path.join(os.environ['IQSM_PLUS_DIR'], 'checkpoints')
for name in ['iQSM_plus.pth', 'LoTLayer_chi.pth']:
    local = os.path.join(ckpt_dir, name)
    if os.path.exists(local):
        print(f'{name} already present, skipping.')
    else:
        print(f'Fetching {name} ...')
        urllib.request.urlretrieve(f'{base}/{name}', local)
"
