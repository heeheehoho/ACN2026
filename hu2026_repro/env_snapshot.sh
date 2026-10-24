#!/usr/bin/env bash
# Record the environment the results came from (run once on the GPU machine, show it in the video).
#   bash env_snapshot.sh data /path/to/FundusGAN evidence/env.txt
DATA=${1:-data}
GAN_REPO=${2:-}
OUT=${3:-evidence/env.txt}
mkdir -p "$(dirname "$OUT")"
{
  echo "== date";            date -Is
  echo "== GPU";             nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
  echo "== python";          python --version 2>&1
  echo "== packages";        python -m pip freeze 2>/dev/null | grep -iE "^(torch|torchvision|flwr|numpy|scikit-image|scipy|mamba)"
  python - <<'EOF'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| available", torch.cuda.is_available())
EOF
  if [ -n "$GAN_REPO" ] && [ -d "$GAN_REPO/.git" ]; then
    echo "== FundusGAN commit"; git -C "$GAN_REPO" log -1 --format="%H %cd" 2>&1
  fi
  echo "== dataset counts (images / masks / prepare_log failed / roi_misses_disc)"
  for d in "$DATA"/*/; do
    n=$(basename "$d")
    ni=$(ls "$d/images" 2>/dev/null | wc -l); nm=$(ls "$d/masks" 2>/dev/null | wc -l)
    extra=$(python -c "import json,sys;l=json.load(open(sys.argv[1]));print(len(l['failed']),len(l['roi_misses_disc']))" "$d/prepare_log.json" 2>/dev/null)
    echo "$n  $ni / $nm / ${extra:-no log}"
  done
} | tee "$OUT"
