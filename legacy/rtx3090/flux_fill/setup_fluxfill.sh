#!/usr/bin/env bash
# FLUX.1-Fill-dev (nunchaku SVDQuant INT4) 전용 환경. 기존 zimage/trellis 를 건드리지 않는다.
set -euo pipefail
ENV=flux-fill
WHL="https://github.com/nunchaku-tech/nunchaku/releases/download/v1.2.1/nunchaku-1.2.1+cu12.8torch2.9-cp311-cp311-linux_x86_64.whl"

source <HOME>/anaconda3/etc/profile.d/conda.sh
conda env list | grep -q "^$ENV " || conda create -y -n $ENV python=3.11
PY=<HOME>/anaconda3/envs/$ENV/bin/python
$PY -m pip install -q --upgrade pip
echo "── torch 2.9 + cu128"
$PY -m pip install -q torch==2.9.* torchvision --index-url https://download.pytorch.org/whl/cu128
echo "── diffusers 스택"
$PY -m pip install -q "diffusers>=0.36" "transformers>=4.51,<5" accelerate sentencepiece protobuf pillow "huggingface_hub[hf_transfer]"
echo "── nunchaku v1.2.1 (GitHub 릴리스 휠 — PyPI 의 동명 패키지가 아니다)"
$PY -m pip install -q "$WHL"
echo "── 확인"
$PY - <<'PYEOF'
import torch, nunchaku
from nunchaku.utils import get_precision
print("torch     ", torch.__version__, "| cuda", torch.version.cuda, "| avail", torch.cuda.is_available())
print("gpu       ", torch.cuda.get_device_name(0), "sm", "".join(map(str, torch.cuda.get_device_capability(0))))
print("nunchaku  ", nunchaku.__version__)
print("precision ", get_precision())
PYEOF
