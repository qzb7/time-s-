#!/usr/bin/env bash
set -euo pipefail

# 单卡：python generate_time.py [options]
# 多卡：加 --nproc-per-node N，用 torchrun 起 N 个进程，按 LOCAL_RANK 分片任务。
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
nproc="${NPROC_PER_NODE:-1}"

passthrough=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc-per-node)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "Missing value for --nproc-per-node" >&2; exit 2
      fi
      nproc="$2"; shift 2 ;;
    -h|--help)
      echo "Usage: bash generate_time.sh [--nproc-per-node N] [generate_time.py options]" >&2
      echo "  --nproc-per-node N  使用卡数（默认 \$NPROC_PER_NODE 或 1）" >&2
      echo "  多卡时用 ASCEND_RT_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES 指定可见卡，" >&2
      echo "  generate_time.py 的 --device 保持默认 auto 即可。" >&2
      exit 0 ;;
    *)
      passthrough+=("$1"); shift ;;
  esac
done

if [[ ! "$nproc" =~ ^[1-9][0-9]*$ ]]; then
  echo "--nproc-per-node must be a positive integer" >&2; exit 2
fi

if [[ "$nproc" == 1 ]]; then
  exec python "$script_dir/generate_time.py" "${passthrough[@]}"
else
  exec python -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
    "$script_dir/generate_time.py" "${passthrough[@]}"
fi
