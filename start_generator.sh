#!/usr/bin/env bash
# Prepare on rank zero -> torchrun on all nodes -> merge on rank zero.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python="${GENERATOR_PYTHON:-python}"
config="$root/sources.json"
pretrain_dir=/data/GIFTEvalPretrain
gifteval_dir=/data/GIFTEval
chronos_dir=/data/chronos
time_dir=/data/TIME
pretrain_ratio=40; gifteval_ratio=25; chronos_ratio=20; time_ratio=15
total=1000000; gifteval_test_count=0; time_test_count=0
batch_size=64; group_size=128; clip_batch=16384; workers=16; prefetch=6
sample_pool_size=61440; seed=20260911; merge_workers=10
clip_checkpoint="$root/weights/ViT-B-32.pt"
output="$root/runs/default"
gpus=auto
nodes="${NNODES:-${SLURM_NNODES:-1}}"
node_rank="${NODE_RANK:-${SLURM_NODEID:-0}}"
master_addr="${MASTER_ADDR:-}"
master_port="${MASTER_PORT:-29777}"
launch_mode=foreground; dry_run=0; resume=0
timeout="${GENERATOR_TIMEOUT:-86400}"
die() { echo "Generator error: $*" >&2; exit 2; }

# Optional CLI overrides; otherwise just edit the defaults above.
while (( $# )); do
  case "$1" in
    --help|-h)
      echo 'Usage: bash start_generator.sh [--total N] [--gifteval-test-count N] [--time-test-count N]'
      echo 'Paths: --pretrain-dir --gifteval-dir --chronos-dir --time-dir --output'
      echo 'Training weights: --pretrain-ratio --gifteval-ratio --chronos-ratio --time-ratio'
      echo 'Encoder: --batch-size --group-size --clip-batch --workers --prefetch --sample-pool-size --seed'
      echo 'Launch: --python --gpus auto|N --nodes N --node-rank N --master-addr HOST --master-port N'
      echo 'Other: --config --clip-checkpoint --dry-run --resume --merge-workers; foreground only (use nohup for single-node background)'
      exit 0;;
    --dry-run|--dry_run) dry_run=1; shift; continue;;
    --resume) resume=1; shift; continue;;
  esac
  if [[ "$1" == --*=* ]]; then set -- "${1%%=*}" "${1#*=}" "${@:2}"; fi
  (( $# >= 2 )) || die "Missing value for $1"
  key="${1#--}"; key="${key//-/_}"
  case "$key" in
    python|config|pretrain_dir|gifteval_dir|chronos_dir|time_dir|output|clip_checkpoint|\
    pretrain_ratio|gifteval_ratio|chronos_ratio|time_ratio|total|gifteval_test_count|time_test_count|\
    batch_size|group_size|clip_batch|workers|prefetch|sample_pool_size|seed|merge_workers|\
    gpus|nodes|node_rank|master_addr|master_port|launch_mode) printf -v "$key" '%s' "$2";;
    *) die "Unknown option $1";;
  esac
  shift 2
done
[[ "$launch_mode" == foreground ]] || die 'Use nohup bash start_generator.sh ... & for single-node background'
for n in "$nodes" "$master_port" "$batch_size" "$group_size" "$clip_batch" "$workers" "$prefetch" "$sample_pool_size" "$timeout"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "Expected positive integer: $n"
done
[[ "$node_rank" =~ ^[0-9]+$ ]] && (( node_rank < nodes )) || die 'Invalid node rank'
(( master_port <= 65535 )) || die 'Invalid master port'
[[ "$total" =~ ^[0-9]+$ ]] || die 'total must be nonnegative'
for n in "$gifteval_test_count" "$time_test_count"; do
  [[ "$n" == -1 || "$n" =~ ^[0-9]+$ ]] || die 'Test count must be -1, 0, or positive'
done
[[ "$output" = /* ]] || output="$PWD/$output"

# torchrun sets ranks itself. If the platform already supplied ranks, do not nest it.
coordinator=$((node_rank == 0))
if [[ -n ${RANK:-}${WORLD_SIZE:-}${LOCAL_RANK:-} ]]; then
  [[ ${RANK:-} =~ ^[0-9]+$ && ${WORLD_SIZE:-} =~ ^[1-9][0-9]*$ && ${LOCAL_RANK:-} =~ ^[0-9]+$ ]] || die 'Incomplete rank environment'
  (( RANK < WORLD_SIZE )) || die 'Rank exceeds world size'
  coordinator=$((RANK == 0)); world="$WORLD_SIZE"
  launcher=("$python" -u)
else
  device_count_py='import torch
try:
    import torch_npu
    print(torch_npu.npu.device_count() if torch_npu.npu.is_available() else torch.cuda.device_count())
except Exception:
    print(torch.cuda.device_count())'
  if [[ "$gpus" == auto ]]; then gpus="$("$python" -c "$device_count_py")"; fi
  [[ "$gpus" =~ ^[1-9][0-9]*$ ]] || die 'No accelerator (NPU/CUDA) detected; activate the intended environment'
  if (( ! dry_run )); then
    visible="$("$python" -c "$device_count_py")"
    (( gpus <= visible )) || die "Requested $gpus local GPUs, visible=$visible"
  fi
  if (( nodes > 1 )); then [[ -n "$master_addr" ]] || die 'Multi-node requires MASTER_ADDR'; fi
  master_addr="${master_addr:-127.0.0.1}"
  world=$((nodes*gpus))
  launcher=("$python" -m torch.distributed.run --nnodes "$nodes" --node_rank "$node_rank"
    --nproc_per_node "$gpus" --master_addr "$master_addr" --master_port "$master_port")
fi
(( world <= 100000 )) || die 'Too many workers for the block size'
block_size=$((100000/world*world))
configure=("$python" "$root/configure.py" --config "$config" --output "$output" --clip-checkpoint "$clip_checkpoint"
  --total "$total" --gifteval-test-count "$gifteval_test_count" --time-test-count "$time_test_count" --seed "$seed"
  --pretrain-dir "$pretrain_dir" --gifteval-dir "$gifteval_dir" --chronos-dir "$chronos_dir" --time-dir "$time_dir"
  --pretrain-ratio "$pretrain_ratio" --gifteval-ratio "$gifteval_ratio" --chronos-ratio "$chronos_ratio" --time-ratio "$time_ratio")
encoder=("$root/encoder/extract.py" --batch_size "$batch_size" --sample_group_size "$group_size"
  --clip_checkpoint "$clip_checkpoint" --clip_batch_size "$clip_batch" --workers "$workers"
  --prefetch_groups "$prefetch" --sample_pool_size "$sample_pool_size" --seed "$seed"
  --block_size "$block_size" --zarr_write_batch_size 64)
if (( dry_run )); then
  "${configure[@]}" --dry-run
  printf 'Encoder launch: '; printf '%q ' "${launcher[@]}" "${encoder[@]}"; printf '\n'
  if (( total > 0 )); then echo "Pipeline: prepare.py -> gift210_mix_loader -> merged ($total training rows)"; fi
  if (( gifteval_test_count != 0 || time_test_count != 0 )); then
    echo "Test phase: prepare_test_inputs.py -> test_input_loader -> test_inputs/merged; --gifteval-test-count $gifteval_test_count --time-test-count $time_test_count; TEST HISTORY ONLY"
  fi
  exit 0
fi
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export FOUR_SOURCE_CACHE="$output/cache" FOUR_SOURCE_TOTAL="$total"
if (( coordinator )); then
  if (( resume )) && [[ -f "$output/launch.json" ]]; then
    find "$output" -maxdepth 1 -name 'FAILED-*' -type f -delete 2>/dev/null || true
    echo "Resume mode: reusing $output/sources.resolved.json + launch.json"
  else
    "${configure[@]}"
  fi
fi

# Shared filesystem markers are needed for multi-node preparation/merge.
wait_file() {
  local began=$SECONDS
  until [[ -f "$1" ]]; do
    if find "$output" -maxdepth 1 -name 'FAILED-*' -type f 2>/dev/null | grep -q .; then die 'A node failed; see FAILED-*'; fi
    (( SECONDS-began < timeout )) || die "Timed out waiting for $1"
    sleep 2
  done
}
wait_file "$output/launch.json"
phase() {
  local name="$1" loader="$2" count="$3" ready="$4" base="$output" began status start_block
  shift 4
  if [[ "$name" == test_inputs ]]; then base="$output/test_inputs"; fi
  if [[ -f "$base/merged/COMPLETE.json" ]]; then
    echo "Phase $name already merged; skipping ($base/merged/COMPLETE.json)"
    return 0
  fi
  start_block=0
  if (( resume )); then
    start_block="$("$python" "$root/resume_scan.py" --shards "$base/shards" --world "$world" --block-size "$block_size" --clean)"
    echo "Resume $name: start_block=$start_block"
  fi
  if (( coordinator )); then
    if [[ -f "$ready" ]]; then
      echo "Reuse $ready (skip prepare)"
    else
      if "$@"; then :; else
        status=$?; printf 'Preparation failed: %s\n' "$name" > "$output/FAILED-$name-prepare"; return "$status"
      fi
    fi
  fi
  wait_file "$ready"
  if [[ "$name" == test_inputs ]]; then count="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["total"])' "$ready")"; fi
  export GENERATOR_DATA_SPLIT="$name" ZARR_RANK_ROOT="$base/shards"
  export TEST_INPUT_PLAN="$output/test_inputs/plan.json" TEST_INPUT_SHARDS="$base/shards"
  mkdir -p "$base/shards"
  echo "Generate $name: global rows=$count, workers=$world"
  if "${launcher[@]}" "${encoder[@]}" --loader_module "$loader" --output_dir "$base/shards" --num_samples "$count" --start_block "$start_block"; then :; else
    status=$?; printf 'Encoder failed: %s\n' "$name" > "$output/FAILED-$name-node$node_rank-rank${RANK:-launcher}"; return "$status"
  fi
  if (( coordinator )); then
    began=$SECONDS
    until [[ $(find "$base/shards" -maxdepth 1 -name 'DONE-rank-*.json' | wc -l | tr -d ' ') == "$world" ]]; do
      if find "$output" -maxdepth 1 -name 'FAILED-*' -type f | grep -q .; then die 'Encoder failed on another node'; fi
      (( SECONDS-began < timeout )) || die "Timed out waiting for $world completed ranks"
      sleep 2
    done
    if "$python" "$root/merge.py" "$base/shards" "$base/merged" --expected "$count" --workers "$merge_workers"; then :; else
      status=$?; printf 'Merge failed: %s\n' "$name" > "$output/FAILED-$name-merge"; return "$status"
    fi
    echo "Complete: $base/merged/COMPLETE.json"
  fi
}
if (( total > 0 )); then
  phase training gift210_mix_loader "$total" "$output/cache/READY.json" \
    "$python" "$root/prepare.py" "$output/sources.resolved.json" "$output/cache"
fi
if (( gifteval_test_count != 0 || time_test_count != 0 )); then
  phase test_inputs test_input_loader 0 "$output/test_inputs/plan.json" \
    "$python" "$root/prepare_test_inputs.py" --gifteval-dir "$gifteval_dir" --time-dir "$time_dir" \
    --gifteval-test-count "$gifteval_test_count" --time-test-count "$time_test_count" \
    --seed "$seed" --output "$output/test_inputs/plan.json"
fi
