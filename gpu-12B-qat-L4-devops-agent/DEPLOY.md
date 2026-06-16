# Deployment Guide: Self-Hosted vLLM on Azure VM (Gemma 4 12B-it QAT)

This document summarizes the deployment state, configuration, and architecture for the self-hosted vLLM inference server running on Azure Virtual Machines.

---

## 📦 Model Artifacts
The model is served using the official quantized w4a16 compressed-tensors (QAT) checkpoint:
*   **Source:** Hugging Face (`google/gemma-4-12B-it-qat-w4a16-ct`)
*   **Alternative Storage:** Azure Blob Storage (`https://<account>.blob.core.windows.net/models/gemma-4-12B-it-qat-w4a16-ct/`)
*   **Format:** Hugging Face Transformers (Safetensors with compressed-tensors)

---

## 🚀 Azure Inference Stack (Standard_NV36ads_A10_v5)
The inference server is hosted on an Azure Virtual Machine optimized for graphics-intensive and compute workloads.

*   **Instance Type:** `Standard_NV36ads_A10_v5`
*   **GPU Accelerator:** 1x NVIDIA A10 (24 GiB VRAM)
*   **vCPUs / RAM:** 36 vCPUs / 440 GiB RAM
*   **Operating System:** Azure Linux 4.0 (Fedora-based Public Preview)
*   **Image URN:** `microsoftazurelinux:azurelinux-4:4:latest`
*   **Container Port:** `8080` (mapped to Host Port `8080`)
*   **Network Security Group:** Allows inbound TCP on `8080` and `22`

### Azure CLI Launch Command
```bash
# 1. Create a Resource Group
az group create --name gpu-12b-qat-l4-devops-agent-rg --location eastus

# 2. Deploy the VM with Azure Linux 4.0 and custom data script
az vm create \
  --resource-group gpu-12b-qat-l4-devops-agent-rg \
  --name gpu-12b-qat-l4-devops-agent-vm \
  --image microsoftazurelinux:azurelinux-4:4:latest \
  --size Standard_NV36ads_A10_v5 \
  --admin-username azureuser \
  --generate-ssh-keys \
  --custom-data user_data.sh
```

---

## 🛠 Deployment & Startup Script (`user_data.sh`)

The instance is deployed with a `custom-data` script that automatically provisions `moby-engine` (Docker), configures the NVIDIA container runtime on Azure Linux 4.0 (Fedora-based), and launches the vLLM engine inside a container with optimized parameters.

### UserData script (`user_data.sh`):
```bash
#!/bin/bash
# Install container engine (moby-engine) on Azure Linux 4.0
dnf install -y moby-engine
systemctl start docker
systemctl enable docker

# Configure NVIDIA repositories for Fedora (upstream base of Azure Linux 4.0)
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
dnf clean all
dnf install -y cuda-drivers nvidia-container-toolkit
systemctl restart docker

# Run vLLM Docker container with optimized parameters
docker run -d --name vllm-server \
  --gpus all \
  --ipc=host \
  --restart always \
  -p 8080:8080 \
  -e HF_TOKEN="<your-hf-token>" \
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
```

### Key Parameters Explained
*   `--dtype bfloat16`: Prevents numerical overflow/underflow and matches Gemma 4's native precision.
*   `--quantization compressed-tensors`: Required to deserialize the QAT INT4 model weights.
*   `--gpu-memory-utilization 0.95`: Allocates 95% of GPU memory to vLLM's cache.
*   `--kv-cache-dtype fp8`: Cuts KV cache footprint in half, drastically improving concurrency.
*   `--tool-call-parser gemma4` & `--reasoning-parser gemma4`: Critical for correct parsing of tool calls.

---

## 🔗 Integration with SRE Agent
To connect the MCP SRE Agent to the newly deployed Azure VM endpoint:

1. Discover the public IP of your VM (e.g. `13.82.4.5`).
2. Export the endpoint URL in your terminal environment:
   ```bash
   export VLLM_BASE_URL="http://13.82.4.5:8080"
   export MODEL_NAME="google/gemma-4-12B-it-qat-w4a16-ct"
   ```
3. Start the agent:
   ```bash
   make run
   ```
