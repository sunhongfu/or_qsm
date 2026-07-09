#!/bin/bash
# Runs once when the devcontainer is created (VS Code's postCreateCommand).
set -e

git config --global core.autocrlf true
pip install ipykernel torch nibabel scipy

# Clone iQSM_Plus directly into the container rather than requiring a host bind-mount
# (previously: mounts a local /Users/<you>/... path, which only worked on the original
# author's machine). Users of this devcontainer aren't expected to modify iQSM_Plus's
# code, so a fresh clone each time the container is (re)created is fine.
if [ ! -d /opt/iQSM_Plus/.git ]; then
    git clone https://github.com/sunhongfu/iQSM_Plus.git /opt/iQSM_Plus
fi

# Pretrained model checkpoints are hosted on Hugging Face, not committed to the git repo
# -- mirrors iQSM_Plus's own `run.py --download-checkpoints`.
mkdir -p /opt/iQSM_Plus/checkpoints
python3 -c "
import os, urllib.request
base = 'https://huggingface.co/sunhongfu/iQSM_Plus/resolve/main'
for name in ['iQSM_plus.pth', 'LoTLayer_chi.pth']:
    local = f'/opt/iQSM_Plus/checkpoints/{name}'
    if os.path.exists(local):
        print(f'{name} already present, skipping.')
    else:
        print(f'Fetching {name} ...')
        urllib.request.urlretrieve(f'{base}/{name}', local)
"
