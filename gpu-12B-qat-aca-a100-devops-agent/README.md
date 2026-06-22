# Self-Hosted vLLM DevOps Agent (MCP Server) - Azure Only

This project provides an automated DevOps/SRE assistant that leverages **Gemma models self-hosted via vLLM on Azure Container Apps (ACA)**. It bridges Azure Monitor Log Analytics with a private inference endpoint to analyze infrastructure issues and suggest remediations.

## 🚀 Deployment Requirements

To deploy and run this project, you need to address two main components: the **Inference Stack** (vLLM on Azure Container Apps with GPU) and the **MCP Server** itself.

### 1. Infrastructure Requirements (The Inference Stack)
The MCP server expects a running vLLM instance. Your Azure Container App deployment for the model needs:
*   **Hardware:** NVIDIA A100 GPU (1 unit).
*   **Workload Profile Type:** `Consumption-GPU-NC24-A100`.
*   **Compute:** Minimum 24 vCPUs and 220.0GiB RAM.
*   **Storage:** An Azure Blob Storage container containing the Gemma model weights (e.g., container `models` under storage account `AZURE_STORAGE_ACCOUNT`).
*   **Key Storage:** Azure Key Vault for storing secret tokens.

### 2. Software & API Dependencies
The agent relies on several Azure CLI tools and Python libraries:
*   **Libraries:** `mcp`, `fastmcp`, `huggingface_hub`, `openai`, `httpx`, and `requests`.
*   **Tools:** Azure CLI (`az`) authenticated to your Azure Subscription.
*   **Permissions:** The environment running the agent needs Azure CLI access to read Key Vault secrets, query container app deployment status, and read Log Analytics workspaces.

### 3. Environment Variables
You can configure the following variables for the MCP server:
*   `AZURE_LOCATION`: The target Azure region (defaults to `westus`).
*   `AZURE_KEYVAULT_NAME`: Key Vault name used to store secrets (defaults to `vllm-devops-kv`).
*   `AZURE_STORAGE_ACCOUNT`: Storage Account name where model weights are located.
*   `AZURE_LOG_ANALYTICS_WORKSPACE_ID`: Your Azure Monitor Log Analytics Workspace ID (required for `analyze_cloud_logging`).
*   `VLLM_BASE_URL`: The URL of your Azure Container App. **If omitted, the agent will attempt to auto-discover it using `az`.**
*   `MODEL_NAME`: The model identifier used by vLLM (defaults to `google/gemma-4-12B-it-qat-w4a16-ct`).

## 🛠 Usage & Setup

### Step 1: Prepare Model Weights
Use the built-in tool `get_huggingface_model_copy_instructions` to move Gemma weights to your Azure Blob Storage container.

### Step 2: Deploy vLLM to Azure Container Apps
Run the `get_vllm_gpu_deployment_config` tool within the MCP server to generate the exact Azure CLI commands for deployment, or use the provided [Makefile](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/Makefile):
```bash
make deploy
```

### Step 3: Run the MCP Server
Install dependencies and run the server:
```bash
make install
# Optional: export VLLM_BASE_URL="your-vllm-url"
make run
```

## 🛠 Available Tools

The following tools are available via the MCP server:

### 🐳 Infrastructure & Deployment
*   **[deploy_vllm](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Deploys vLLM to Azure Container Apps with A100 GPU.
*   **[destroy_vllm](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Deletes the Container App deployment.
*   **[status_vllm](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Checks the status of the Container App.
*   **[update_vllm_scaling](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Updates instances scaling profile.
*   **[get_vllm_deployment_config](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Generates the Azure VM deployment configuration.
*   **[get_vllm_gpu_deployment_config](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Generates Azure Container Apps CLI commands for A100 GPU.
*   **[check_gpu_quotas](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Checks GPU VM core family quotas for an Azure region.
*   **[get_vllm_endpoint](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Returns the current active vLLM endpoint URL.

### 📦 Model Management
*   **[list_bucket_models](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Lists model weights in Azure Blob Storage.
*   **[save_hf_token](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Securely saves a Hugging Face API token to Azure Key Vault.
*   **[get_huggingface_model_copy_instructions](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Guide to download model from Hugging Face and upload to Azure Storage.
*   **[get_huggingfacehub_download_path](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Resolves local cache path using huggingface_hub.

### 📊 Monitoring & Status
*   **[get_system_status](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Provides a high-level status dashboard of the Azure service and health.
*   **[get_endpoint](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Verifies connectivity and returns the active service URL.
*   **[get_model_details](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Retrieves detailed model metadata and engine state from `/v1/models`.
*   **[verify_model_health](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Deep health check by querying the model with a simple prompt and measuring latency.

### 📈 Performance & Benchmarking
*   **[run_benchmark](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Runs performance/concurrency benchmark sweeps against the vLLM GPU endpoint.

### 💬 Interaction & Diagnostics
*   **[query_gemma4](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Primary tool to query the self-hosted model with standard chat message format.
*   **[query_gemma4_with_stats](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Queries the model and returns streaming performance statistics (TTFT, throughput).
*   **[query_vllm](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Direct text completions querying tool.
*   **[analyze_cloud_logging](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Fetches logs from Azure Monitor Log Analytics and analyzes them.
*   **[analyze_gpu_logs](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Fetches Azure VM logs and uses Gemma 4 to analyze them for SRE/DevOps errors.
*   **[suggest_sre_remediation](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Suggests remediation plans for SRE errors using the model.
*   **[get_help](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/server.py)**: Provides help text and summarizes the configuration options and all available SRE/DevOps tools.

## 📦 Resources
The server exposes the following MCP resources:
*   **`config://vllm-deployment-template`**: A YAML template for Azure VM GPU deployment.

## 📊 Performance Benchmarks (Standard vs. QAT)

The self-hosted **Gemma 4 12B QAT** model has been benchmarked on a single **NVIDIA A100 GPU** to measure concurrency limits:
* **High Concurrency Stability**: The QAT INT4 model maintains a **100% request success rate** up to **512 concurrent users** (with context windows up to 2048 tokens).
* Detailed matrix results and SRE insights are available in [benchmark_report_summary.md](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/benchmark_report_summary.md).

## 🌟 Grand Demo
A standalone demo script is included to showcase the agent's capabilities:
```bash
python demo_launcher.py
```
This script simulates log analysis, remediation suggestions, and infrastructure configuration generation.

## 🛠 Makefile Helpers
The included [Makefile](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/Makefile) provides several shortcuts:
*   `make install`: Installs Python dependencies.
*   `make run`: Starts the MCP server.
*   `make deploy`: Deploys vLLM to Azure Container Apps.
*   `make destroy`: Removes the Azure Resource Group.
*   `make status`: Checks the status of the Container App.
*   `make query PROMPT="your prompt"`: Queries the vLLM model directly via `curl`.
*   `make test`: Runs the test suite in [test_agent.py](file:///home/xbill/gemma4-tips-azure/gpu-12B-qat-aca-a100-devops-agent/test_agent.py).
