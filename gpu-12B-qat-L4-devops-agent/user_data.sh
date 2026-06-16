#!/bin/bash
exec > >(tee -a /var/log/custom-data.log) 2>&1
echo "=== Starting Startup Script ==="

# Wait for docker service to be active
until systemctl is-active --quiet docker; do
  echo "Waiting for Docker..."
  sleep 2
done

# Start the vLLM container
docker run -d --name vllm-server \
  --gpus all \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  vllm/vllm-openai:nightly \
  --model google/gemma-4-12B-it-qat-w4a16-ct \
  --quantization compressed-tensors \
  --dtype bfloat16 \
  --max-model-len 32768 \
  --disable-chunked-mm-input \
  --gpu-memory-utilization 0.95 \
  --kv-cache-dtype fp8 \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 4096 \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --async-scheduling \
  --limit-mm-per-prompt '{}' \
  --host 0.0.0.0 \
  --port 8080

echo "=== Startup Script Completed ==="
