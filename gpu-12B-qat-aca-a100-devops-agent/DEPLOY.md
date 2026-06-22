# Deployment Guide: Self-Hosted vLLM on Azure Container Apps (Gemma 4 12B-it QAT)

This document summarizes the deployment state, configuration, and architecture for the self-hosted vLLM inference server running on Azure Container Apps (ACA) with GPU acceleration.

---

## 📦 Model Artifacts
The model is served using the official quantized w4a16 compressed-tensors (QAT) checkpoint:
*   **Source:** Hugging Face (`google/gemma-4-12B-it-qat-w4a16-ct`)
*   **Alternative Storage:** Azure Blob Storage (`https://<account>.blob.core.windows.net/models/gemma-4-12B-it-qat-w4a16-ct/`)
*   **Format:** Hugging Face Transformers (Safetensors with compressed-tensors)

---

## 🚀 Azure inference Stack (Azure Container Apps GPU)
The inference server is hosted on Azure Container Apps utilizing serverless GPU workload profiles.

*   **Workload Profile Type:** `Consumption-GPU-NC24-A100`
*   **GPU Accelerator:** 1x NVIDIA A100 PCIe (80 GB VRAM)
*   **vCPUs / RAM:** 24 vCPUs / 220 GiB RAM
*   **Container Port:** `8080` (External Ingress enabled)
*   **Container Image:** `vllm/vllm-openai:nightly`

---

## 🛠 Deployment & Setup Commands

Deploy the container app with GPU profile using the following CLI commands:

```bash
# 1. Create a Resource Group
az group create --name gpu-12b-qat-l4-devops-agent-rg --location eastus

# 2. Create the Container Apps Environment
az containerapp env create \
  --name gpu-12b-qat-l4-devops-agent-env \
  --resource-group gpu-12b-qat-l4-devops-agent-rg \
  --location eastus

# 3. Add the GPU Workload Profile
az containerapp env workload-profile add \
  --name gpu-12b-qat-l4-devops-agent-env \
  --resource-group gpu-12b-qat-l4-devops-agent-rg \
  --workload-profile-name gpu-profile \
  --workload-profile-type Consumption-GPU-NC24-A100

# 4. Deploy the Container App with vLLM
az containerapp create \
  --name gpu-12b-qat-l4-devops-agent-app \
  --resource-group gpu-12b-qat-l4-devops-agent-rg \
  --environment gpu-12b-qat-l4-devops-agent-env \
  --workload-profile-name gpu-profile \
  --image vllm/vllm-openai:nightly \
  --cpu 24.0 \
  --memory 220.0Gi \
  --ingress external \
  --target-port 8080 \
  --env-vars HF_TOKEN="<your-hf-token>" \
  --args \
    "--model" "google/gemma-4-12B-it-qat-w4a16-ct" \
    "--quantization" "compressed-tensors" \
    "--dtype" "bfloat16" \
    "--max-model-len" "32768" \
    "--disable-chunked-mm-input" \
    "--gpu-memory-utilization" "0.95" \
    "--kv-cache-dtype" "fp8" \
    "--tensor-parallel-size" "1" \
    "--max-num-seqs" "8" \
    "--enable-chunked-prefill" \
    "--max-num-batched-tokens" "4096" \
    "--enable-auto-tool-choice" \
    "--tool-call-parser" "gemma4" \
    "--reasoning-parser" "gemma4" \
    "--async-scheduling" \
    "--limit-mm-per-prompt" "{}" \
    "--host" "0.0.0.0" \
    "--port" "8080"
```

### Key Parameters Explained
*   `--dtype bfloat16`: Prevents numerical overflow/underflow and matches Gemma 4's native precision.
*   `--quantization compressed-tensors`: Required to deserialize the QAT INT4 model weights.
*   `--gpu-memory-utilization 0.95`: Allocates 95% of GPU memory to vLLM's cache.
*   `--kv-cache-dtype fp8`: Cuts KV cache footprint in half, drastically improving concurrency.
*   `--tool-call-parser gemma4` & `--reasoning-parser gemma4`: Critical for correct parsing of tool calls.

---

## 🔗 Integration with SRE Agent
To connect the MCP SRE Agent to the newly deployed Azure Container App endpoint:

1. Discover the FQDN of your Container App:
   ```bash
   az containerapp show \
     --resource-group gpu-12b-qat-l4-devops-agent-rg \
     --name gpu-12b-qat-l4-devops-agent-app \
     --query properties.configuration.ingress.fqdn \
     -o tsv
   ```
   *(e.g., `gpu-12b-qat-l4-devops-agent-app.eastus.azurecontainerapps.io`)*

2. Export the endpoint URL in your terminal environment:
   ```bash
   export VLLM_BASE_URL="https://gpu-12b-qat-l4-devops-agent-app.eastus.azurecontainerapps.io"
   export MODEL_NAME="google/gemma-4-12B-it-qat-w4a16-ct"
   ```

3. Start the agent:
   ```bash
   make run
   ```
