#!/bin/bash
exec > >(tee -a /var/log/custom-data.log) 2>&1
echo "=== Starting Startup Script ==="

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl wget apt-transport-https ca-certificates gnupg lsb-release

# Install Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt-get update -y
apt-get install -y docker-ce docker-ce-cli containerd.io
systemctl start docker
systemctl enable docker

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update -y
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

# Start vLLM container
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
