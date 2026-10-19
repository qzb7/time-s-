#!/usr/bin/env bash
# 从 checkpoint-10000 到 checkpoint-50000，每隔 1000 步评测一次 endogenous 模型。
# 每个 checkpoint 一个独立输出目录（protocol 哈希不同，不能混在同一个目录）。
# 用法：bash eval_checkpoints.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# -------------------- 可改配置 --------------------
RUN_DIR="/data/ykj/QiYao-M-internal-run6"          # internal 训练输出目录（checkpoint-* 所在）
START_STEP=10000                                    # 起始步
END_STEP=50000                                      # 结束步（含）
STEP_INTERVAL=1000                                  # 步长
GIFT_EVAL_DIR="/data/ykj/data/gift_eval/dataset"   # 原始 GIFT-Eval 数据
CACHE_DIR="/data/ykj/output7"                      # patch 图文缓存
OUTPUT_BASE="/data/ykj"                            # 每个 checkpoint 输出到 ${OUTPUT_BASE}/tests${step}
BATCH_SIZE=32
# ---------------------------------------------------

[[ -f run_eval.sh ]] || { echo "找不到 run_eval.sh，请在评测目录下运行" >&2; exit 2; }

for ((step=START_STEP; step<=END_STEP; step+=STEP_INTERVAL)); do
    ckpt="${RUN_DIR}/checkpoint-${step}"
    out="${OUTPUT_BASE}/tests${step}"

    if [[ ! -d "$ckpt" ]]; then
        echo "[skip] 缺失 checkpoint: $ckpt" >&2
        continue
    fi
    if [[ -f "$out/all_results.csv" ]]; then
        echo "[skip] 已存在结果: $out/all_results.csv" >&2
        continue
    fi

    echo "==================== step=${step} -> ${out} ===================="
    bash run_eval.sh endogenous \
        --gift-eval-dir "$GIFT_EVAL_DIR" \
        --cache-dir "$CACHE_DIR" \
        --output "$out" \
        --batch-size "$BATCH_SIZE" \
        --checkpoint "$ckpt" \
        --no-shared-head \
        --model-name "QiYao-M-${step}"
done

echo "全部完成（缺失或已完成的 checkpoint 已跳过）。"
