#!/usr/bin/env bash
set -euo pipefail

# 一键循环测评多个 checkpoint，支持多 NPU。
# 每个 checkpoint 一个独立输出目录，跑完自动打 .loop_complete 标记做断点续跑。
#
# 用法示例（2 卡，从 66000 往前每 1000 步测一次）：
#   ASCEND_RT_VISIBLE_DEVICES=0,1 bash evaluate_time_loop.sh --nproc-per-node 2
#
# 只测实际存在的 checkpoint-* 目录，所以 --end-step 设成 0 也不会空跑。

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# ---- 默认值（与 evaluate_time.py 的固定参数对应）----
time_data="/data/ykj/data/TIME"
cache_dir="/data/ykj/qiyaom_time_eval_cache2"
checkpoint_root="/data/ykj/QiYao-M-internal-run13"
output_root="/data/ykj/time_results/QiYao-M"
model_name="QiYao-M"
batch_size=32
scaler_fallback_min_obs=8

# ---- 步数参数 ----
start_step=66000
step_interval=1000
end_step=0
nproc="${NPROC_PER_NODE:-1}"
dry_run=0

usage() {
  cat >&2 <<'EOF'
Usage: bash evaluate_time_loop.sh [options]

  多卡:  --nproc-per-node N          (torchrun 起 N 进程; 用 ASCEND_RT_VISIBLE_DEVICES/CUDA_VISIBLE_DEVICES 指定可见卡)
  步数:  --start-step S              (默认 66000)
         --step-interval I           (默认 1000)
         --end-step E                (默认 0, 只测实际存在的 checkpoint-S 目录)
         --dry-run                   (只打印计划, 不执行)

  固定参数(均有默认值):
         --time-data DIR             (默认 /data/ykj/data/TIME)
         --cache-dir DIR             (默认 /data/ykj/qiyaom_time_eval_cache2)
         --checkpoint-root DIR       (默认 /data/ykj/QiYao-M-internal-run13)
         --output-root DIR           (默认 /data/ykj/time_results/QiYao-M)
         --model-name NAME           (默认 QiYao-M)
         --batch-size N              (默认 32)
         --scaler-fallback-min-obs N (默认 8)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --nproc-per-node)          nproc="$2"; shift 2 ;;
    --start-step)              start_step="$2"; shift 2 ;;
    --step-interval)           step_interval="$2"; shift 2 ;;
    --end-step)                end_step="$2"; shift 2 ;;
    --time-data)               time_data="$2"; shift 2 ;;
    --cache-dir)               cache_dir="$2"; shift 2 ;;
    --checkpoint-root)         checkpoint_root="$2"; shift 2 ;;
    --output-root)             output_root="$2"; shift 2 ;;
    --model-name)              model_name="$2"; shift 2 ;;
    --batch-size)              batch_size="$2"; shift 2 ;;
    --scaler-fallback-min-obs) scaler_fallback_min_obs="$2"; shift 2 ;;
    --dry-run)                 dry_run=1; shift ;;
    -h|--help)                 usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ ! "$nproc" =~ ^[1-9][0-9]*$ ]]; then
  echo "--nproc-per-node must be a positive integer" >&2; exit 2
fi
if [[ ! "$start_step" =~ ^[0-9]+$ || ! "$step_interval" =~ ^[0-9]+$ || ! "$end_step" =~ ^[0-9]+$ ]]; then
  echo "--start-step/--step-interval/--end-step must be non-negative integers" >&2; exit 2
fi
if [[ "$step_interval" == 0 ]]; then
  echo "--step-interval must be > 0" >&2; exit 2
fi
if [[ ! "$batch_size" =~ ^[1-9][0-9]*$ ]]; then
  echo "--batch-size must be a positive integer" >&2; exit 2
fi

# 收集实际存在的 checkpoint 步数（从大到小）
steps=()
for ((s = start_step; s >= end_step; s -= step_interval)); do
  if [[ -d "$checkpoint_root/checkpoint-$s" ]]; then
    steps+=("$s")
  fi
done

if [[ ${#steps[@]} -eq 0 ]]; then
  echo "No checkpoint-* dirs found under $checkpoint_root (range $start_step..$end_step, step $step_interval)" >&2
  exit 1
fi

echo "=== 计划测评 ${#steps[@]} 个 checkpoint, nproc=$nproc ==="
printf '  checkpoint-%s\n' "${steps[@]}"

if [[ "$dry_run" == 1 ]]; then
  echo "=== dry-run, 不执行 ==="
  exit 0
fi

run_one() {
  local s="$1"
  local out="$output_root/step-$s"
  local common=(
    --time-data "$time_data"
    --cache-dir "$cache_dir"
    --checkpoint "$checkpoint_root/checkpoint-$s"
    --output-dir "$out"
    --model-name "$model_name"
    --batch-size "$batch_size"
    --scaler-fallback-min-obs "$scaler_fallback_min_obs"
    --device auto
  )
  echo ">>> 测评 checkpoint-$s -> $out"
  if [[ "$nproc" == 1 ]]; then
    python "$script_dir/evaluate_time.py" "${common[@]}"
  else
    # 每个 checkpoint 用独立端口，避免顺序跑时 TIME_WAIT 端口残留冲突。
    python -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
      --master_port "$((29500 + (s % 1000)))" \
      "$script_dir/evaluate_time.py" "${common[@]}"
  fi

  # 所有 shard 都 complete 才标记该 checkpoint 完成。
  local all_done=1
  for ((r = 0; r < nproc; r++)); do
    if [[ ! -f "$out/shard${r}_complete.json" ]]; then
      all_done=0
      break
    fi
  done
  if [[ "$all_done" == 1 ]]; then
    touch "$out/.loop_complete"
  fi
}

for s in "${steps[@]}"; do
  out="$output_root/step-$s"
  if [[ -f "$out/.loop_complete" ]]; then
    echo "SKIP checkpoint-$s (已 .loop_complete)"
    continue
  fi
  run_one "$s"
done

echo "=== 全部完成 ==="
