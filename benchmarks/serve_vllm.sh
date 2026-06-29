#!/usr/bin/env bash
#   bash benchmarks/serve_vllm.sh
#   python -m benchmarks.llm_eval --backend vllm --model "$MODEL"
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-4096}"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "${GPU_MEM:-0.45}" \
  --dtype auto \
  --disable-log-requests
