import asyncio
import csv
import json
import logging
import os
import statistics
import subprocess
import sys
import time
from typing import Optional

import httpx
from google.cloud import aiplatform, secretmanager, storage
from google.cloud import logging as cloud_logging
from mcp.server.fastmcp import FastMCP
from openai import AsyncOpenAI

# Setup logging to stderr ONLY to avoid interfering with MCP stdio communication
logging.basicConfig(
    stream=sys.stderr, level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("vllm-devops-agent")
logger.info("Initializing DevOps Agent MCP Server...")

# Initialize FastMCP server
mcp = FastMCP("Self-Hosted vLLM DevOps Agent")

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "aisprint-491218")
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-east4")
BUCKET_NAME = f"{PROJECT_ID}-bucket"

# Azure Configuration
AZURE_LOCATION = os.getenv("AZURE_LOCATION", "eastus")
AZURE_KEYVAULT_NAME = os.getenv("AZURE_KEYVAULT_NAME", "vllm-devops-kv")
AZURE_STORAGE_ACCOUNT = os.getenv("AZURE_STORAGE_ACCOUNT", "vllmmodelsstore")

# AWS Configuration (Fallbacks for template outputs)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "vllm-weights-bucket")

# The URL of the self-hosted vLLM service on Cloud Run or Azure VM
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-12B-it-qat-w4a16-ct")
HF_SECRET_ID = "hf-token"


async def get_secret(secret_id: str = HF_SECRET_ID) -> Optional[str]:
    """Retrieves a secret from Azure Key Vault, GCP Secret Manager, or environment variables."""
    # 1. Check environment variable
    val = os.getenv("HF_TOKEN") or os.getenv("HF_API_KEY")
    if val:
        return val

    # 2. Check Azure Key Vault
    try:
        cmd = [
            "az",
            "keyvault",
            "secret",
            "show",
            "--name",
            secret_id,
            "--vault-name",
            AZURE_KEYVAULT_NAME,
            "--query",
            "value",
            "-o",
            "tsv",
        ]
        process = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=10)
        if process.returncode == 0 and process.stdout.strip():
            return process.stdout.strip()
    except Exception as e:
        logger.debug(f"Azure Key Vault secret retrieval failed: {e}")

    # 3. Fallback to GCP Secret Manager
    try:
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{PROJECT_ID}/secrets/{secret_id}/versions/latest"
        response = await asyncio.to_thread(client.access_secret_version, request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        logger.debug(f"GCP Secret Manager failed: {e}")

    return None


@mcp.tool()
async def save_hf_token(token: str) -> str:
    """Securely saves a Hugging Face API token to Azure Key Vault or GCP Secret Manager."""
    saved_azure = False
    saved_gcp = False

    try:
        # Check if KV exists first or create/set
        cmd = [
            "az",
            "keyvault",
            "secret",
            "set",
            "--vault-name",
            AZURE_KEYVAULT_NAME,
            "--name",
            HF_SECRET_ID,
            "--value",
            token,
        ]
        process = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            saved_azure = True
        else:
            logger.warning(f"Azure Key Vault set failed: {process.stderr.strip()}")
    except Exception as e:
        logger.warning(f"Azure Key Vault call failed: {e}")

    try:
        client = secretmanager.SecretManagerServiceClient()
        secret_parent = f"projects/{PROJECT_ID}/secrets/{HF_SECRET_ID}"
        try:
            await asyncio.to_thread(client.get_secret, request={"name": secret_parent})
        except Exception:
            await asyncio.to_thread(
                client.create_secret,
                request={
                    "parent": f"projects/{PROJECT_ID}",
                    "secret_id": HF_SECRET_ID,
                    "secret": {"replication": {"automatic": {}}},
                },
            )
        await asyncio.to_thread(
            client.add_secret_version,
            request={"parent": secret_parent, "payload": {"data": token.encode("UTF-8")}},
        )
        saved_gcp = True
    except Exception as e:
        logger.warning(f"GCP Secret Manager failed: {e}")

    if saved_azure and saved_gcp:
        return "✅ Token saved to both Azure Key Vault and GCP Secret Manager."
    elif saved_azure:
        return "✅ Token saved to Azure Key Vault."
    elif saved_gcp:
        return "✅ Token saved to GCP Secret Manager."
    else:
        return "❌ Failed to save token to Secret Manager (both Azure and GCP failed)."


DEFAULT_SERVICE_NAME = "gpu-12b-qat-l4-devops-agent"


def discover_vllm_url(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """Attempts to automatically discover the Azure VM public IP or Cloud Run service URL."""
    if VLLM_BASE_URL:
        logger.info(f"Using provided VLLM_BASE_URL: {VLLM_BASE_URL}")
        return VLLM_BASE_URL

    # 1. Azure VM Discovery
    logger.info(f"Attempting to discover Azure VM vLLM URL for: {service_name}")
    try:
        cmd = [
            "az",
            "vm",
            "list",
            "--show-details",
            "--query",
            f"[?tags.Name=='{service_name}' || name=='{service_name}-vm' || name=='{service_name}'].{{ip:publicIps, power:powerState}}",
            "-o",
            "json",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            res = json.loads(process.stdout.strip())
            if res:
                for vm in res:
                    ip = vm.get("ip")
                    power = vm.get("power", "")
                    if ip and "running" in power.lower():
                        url = f"http://{ip}:8080"
                        logger.info(f"📡 Automatically discovered Azure VM vLLM at: {url}")
                        return url
    except Exception as e:
        logger.warning(f"⚠️ Error during Azure VM vLLM discovery: {str(e)}")

    # 2. GCP Cloud Run Discovery
    logger.info(f"Attempting to discover GCP Cloud Run URL for service: {service_name}")
    try:
        cmd = [
            "gcloud",
            "run",
            "services",
            "describe",
            service_name,
            f"--project={PROJECT_ID}",
            "--region",
            LOCATION,
            "--format",
            "value(status.url)",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            url = process.stdout.strip()
            if url:
                logger.info(f"📡 Automatically discovered GCP vLLM at: {url}")
                return url
            else:
                logger.warning("⚠️ gcloud returned empty URL for service.")
        else:
            logger.warning(
                f"⚠️ gcloud failed to discover Cloud Run URL (code {process.returncode}): {process.stderr.strip()}"
            )
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Discovery timed out after 15 seconds.")
    except Exception as e:
        logger.warning(f"⚠️ Error during vLLM discovery: {str(e)}")

    logger.error("❌ Failed to discover service URL.")
    return None


# Resolve base URL at runtime
_ACTIVE_VLLM_URL = None


def get_vllm_url() -> str:
    """Returns the cached vLLM URL or discovers it if needed."""
    global _ACTIVE_VLLM_URL
    if not _ACTIVE_VLLM_URL:
        _ACTIVE_VLLM_URL = discover_vllm_url()

    if not _ACTIVE_VLLM_URL:
        raise Exception(
            "Could not determine vLLM service URL. Ensure you are authenticated and the service/instance exists."
        )

    return _ACTIVE_VLLM_URL


def get_auth_token() -> str:
    """For Azure VM, returns empty string. For GCP, gets Google Cloud Identity Token."""
    # If discovered URL has GCP Cloud Run signature, retrieve identity token
    vllm_url = get_vllm_url()
    if "run.app" not in vllm_url:
        return ""
    try:
        return (
            subprocess.check_output(
                ["gcloud", "auth", "print-identity-token"],
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception as e:
        logger.warning(f"⚠️ Could not obtain identity token: {str(e)}")
        return ""


async def get_vllm_client() -> AsyncOpenAI:
    """Initializes and returns an AsyncOpenAI client for the Cloud Run vLLM service."""
    vllm_url = get_vllm_url()
    token = get_auth_token()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return AsyncOpenAI(
        base_url=f"{vllm_url}/v1",
        api_key=token or "not-needed",
        default_headers=headers,
    )


async def get_active_model_name(client: AsyncOpenAI) -> str:
    """Queries the vLLM endpoint to find the active model name, or falls back to configuration."""
    try:
        models_response = await client.models.list()
        if models_response.data:
            return models_response.data[0].id
    except Exception as e:
        logger.warning(f"⚠️ Failed to dynamically query active model from vLLM: {e}")

    # Fallback
    if "/" not in MODEL_NAME:
        return f"/mnt/models/{MODEL_NAME}"
    return MODEL_NAME


# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=LOCATION)


@mcp.resource("config://vllm-deployment-template")
def get_deployment_template() -> str:
    """Returns a base template for Azure VM GPU vLLM deployment."""
    return """
# Azure VM vLLM Deployment Template
# Required Instance: Standard_NV36ads_A10_v5 (1x NVIDIA A10 GPU, 24GB VRAM)
# Recommended Image: microsoftazurelinux:azurelinux-4:4:latest (Azure Linux 4.0 Preview)

VM_Size: Standard_NV36ads_A10_v5
ImageURN: microsoftazurelinux:azurelinux-4:4:latest
Ports:
  - Container Port: 8080
  - Host Port: 8080

Docker Run Command:
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN=$HF_TOKEN \\
  vllm/vllm-openai:nightly \\
  --model google/gemma-4-12B-it-qat-w4a16-ct \\
  --quantization compressed-tensors \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{}' \\
  --host 0.0.0.0 \\
  --port 8080
"""


@mcp.tool()
def get_vllm_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> Optional[str]:
    """
    Returns the current active vLLM endpoint URL.

    Args:
        service_name: The service name or instance Name tag to describe (defaults to 'gpu-12b-qat-l4-devops-agent').
    """
    if service_name == DEFAULT_SERVICE_NAME:
        return get_vllm_url()
    return discover_vllm_url(service_name)


@mcp.tool()
def list_vertex_models() -> str:
    """
    Uses the Vertex AI SDK (part of ADK ecosystem) to list models in the project registry.
    """
    try:
        models = aiplatform.Model.list()
        if not models:
            return "No models found in Vertex AI Model Registry."

        model_list = [f"- {m.display_name} (ID: {m.name})" for m in models]
        return "### Vertex AI Model Registry\n" + "\n".join(model_list)
    except Exception as e:
        return f"Error listing models from Vertex AI: {str(e)}"


@mcp.tool()
def list_bucket_models(bucket_name: Optional[str] = None) -> str:
    """
    Lists the contents of an S3 bucket or GCS bucket to check for uploaded model files.

    Args:
        bucket_name: The S3 or GCS bucket name. Defaults to AWS_BUCKET_NAME or BUCKET_NAME depending on provider.
    """
    if not bucket_name:
        bucket_name = AZURE_STORAGE_ACCOUNT

    is_azure = "blob.core.windows.net" in bucket_name or not bucket_name.startswith("gs://")
    clean_bucket = bucket_name.replace("https://", "").split(".blob")[0].replace("gs://", "")

    if is_azure:
        try:
            container = "models"
            cmd = [
                "az",
                "storage",
                "blob",
                "list",
                "--container-name",
                container,
                "--account-name",
                clean_bucket,
                "--query",
                "[].{name:name, size:properties.contentLength}",
                "-o",
                "json",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if process.returncode == 0:
                res = json.loads(process.stdout.strip())
                if not res:
                    return f"The Azure Blob container '{container}' in account '{clean_bucket}' is empty or does not exist."

                file_list = [
                    f"- https://{clean_bucket}.blob.core.windows.net/{container}/{obj['name']} ({obj['size'] / 1024 / 1024:.2f} MB)"
                    for obj in res[:50]
                ]
                summary = f"### Contents of Azure Blob Container: {clean_bucket}/{container}\n"
                summary += "\n".join(file_list)

                if len(res) > 50:
                    summary += f"\n\n(Showing 50 of {len(res)} items)"

                return summary
        except Exception as e:
            logger.warning(f"Error listing Azure Blob container '{clean_bucket}/{container}': {e}")
            # Fall through to GCS if Azure failed

    try:
        storage_client = storage.Client(project=PROJECT_ID)
        bucket = storage_client.bucket(clean_bucket)
        blobs = list(bucket.list_blobs(max_results=100))

        if not blobs:
            return f"The GCS bucket '{clean_bucket}' is empty."

        file_list = [f"- gs://{clean_bucket}/{b.name} ({b.size / 1024 / 1024:.2f} MB)" for b in blobs[:50]]
        summary = f"### Contents of GCS Bucket: {clean_bucket}\n"
        summary += "\n".join(file_list)

        if len(blobs) > 50:
            summary += f"\n\n(Showing 50 of {len(blobs)} items)"

        return summary
    except Exception as e:
        return f"Error listing objects in bucket '{clean_bucket}' (tried S3 and GCS): {str(e)}"


@mcp.tool()
async def analyze_cloud_logging(filter_query: str, limit: int = 5) -> str:
    """
    Fetches and summarizes error logs from AWS CloudWatch or Google Cloud Logging.

    Args:
        filter_query: Query filter. For CloudWatch, this is the log group name. For GCS, a standard log query.
        limit: Number of recent logs to fetch.
    """
    combined_logs = ""

    # 1. Try Azure Monitor Log Analytics
    workspace_id = os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID")
    if workspace_id:
        try:
            kusto_query = filter_query
            if "ContainerLog" not in kusto_query and "AppTraces" not in kusto_query:
                kusto_query = f"ContainerLogV2 | where LogMessage has '{filter_query}' | order by TimeGenerated desc | take {limit}"

            cmd = [
                "az",
                "monitor",
                "log-analytics",
                "query",
                "-w",
                workspace_id,
                "--analytics-query",
                kusto_query,
                "-o",
                "json",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            if process.returncode == 0:
                res = json.loads(process.stdout.strip())
                log_texts = []
                for entry in res:
                    time_gen = entry.get("TimeGenerated", "")
                    msg = entry.get("LogMessage", entry.get("Message", str(entry)))
                    log_texts.append(f"Timestamp: {time_gen} | Message: {msg}")
                combined_logs = "\n---\n".join(log_texts)
        except Exception as e:
            logger.warning(f"Failed to fetch Azure Monitor logs: {e}")

    # 2. GCP Fallback if no logs found
    if not combined_logs:
        try:
            logging_client = cloud_logging.Client(project=PROJECT_ID)
            entries = list(
                logging_client.list_entries(filter_=filter_query, order_by=cloud_logging.DESCENDING, page_size=limit)
            )
            if entries:
                log_texts = [
                    f"Timestamp: {e.timestamp} | Severity: {e.severity} | Message: {str(e.payload)[:1000] if isinstance(e.payload, str) else json.dumps(e.payload)[:1000]}"
                    for e in entries
                ]
                combined_logs = "\n---\n".join(log_texts)
        except Exception as e:
            logger.warning(f"Failed to fetch Google Cloud logs: {e}")

    if not combined_logs:
        return "No matching logs found."

    try:
        if len(combined_logs) > 12000:
            combined_logs = combined_logs[:12000] + "... (truncated)"

        prompt = f"Analyze the following DevOps logs and provide a high-level summary of the critical issues and potential root causes:\n\n{combined_logs}\n\nSummary:"

        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Log Analysis (Self-Hosted vLLM)\n\n{response_text}"

    except Exception as e:
        return f"Error analyzing logs via self-hosted vLLM: {str(e)}"


@mcp.tool()
async def suggest_sre_remediation(error_message: str) -> str:
    """
    Proposes remediation steps for a specific SRE incident using self-hosted vLLM.

    Args:
        error_message: The error or incident description to remediate.
    """
    prompt = f"As an expert SRE, suggest a 3-step remediation plan for the following error:\n\nError: {error_message}\n\nRemediation Plan:"

    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=512,
            temperature=0.2,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### Remediation Plan\n\n{response_text}"
    except Exception as e:
        return f"Error fetching remediation plan: {str(e)}"


@mcp.tool()
async def query_vllm(prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
    """
    Directly queries the self-hosted vLLM model with a custom prompt.

    Args:
        prompt: The text prompt to send to the model.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 for deterministic).
    """
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        response_text = chat_completion.choices[0].message.content or ""
        return f"### vLLM Response\n\n{response_text}"
    except Exception as e:
        return f"Error querying vLLM: {str(e)}"


@mcp.tool()
def get_vllm_deployment_config(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    location: str = "eastus",
    gpu_memory_utilization: float = 0.95,
) -> str:
    """
    Generates the Azure CLI command and custom data script to deploy vLLM to an Azure VM (Standard_NV36ads_A10_v5) running Azure Linux 4.0.

    Args:
        service_name: The name or Name tag for the VM resource.
        model_path: Hugging Face repo ID or Azure Blob URI of the model.
        location: Azure region to deploy to (default: 'eastus').
        gpu_memory_utilization: The fraction of GPU VRAM to use for KV cache (default: 0.95).
    """
    quant_arg = (
        "--quantization compressed-tensors" if any(q in model_path.lower() for q in ["qat", "w4a16", "ct"]) else ""
    )

    user_data = f"""#!/bin/bash
# Install container engine (moby-engine) on Azure Linux 4.0 (Fedora-based)
dnf install -y moby-engine
systemctl start docker
systemctl enable docker

# Configure NVIDIA repositories for Fedora (upstream base of Azure Linux 4.0)
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
dnf clean all
dnf install -y cuda-drivers nvidia-container-toolkit
systemctl restart docker

# Run vLLM Docker container with optimized parameters
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="$(az keyvault secret show --name hf-token --vault-name {AZURE_KEYVAULT_NAME} --query value -o tsv 2>/dev/null || echo '')" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  {quant_arg} \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization {gpu_memory_utilization} \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""

    az_cmd = (
        f"az group create --name {service_name}-rg --location {location}\n"
        f"az vm create \\\n"
        f"  --resource-group {service_name}-rg \\\n"
        f"  --name {service_name}-vm \\\n"
        f"  --image microsoftazurelinux:azurelinux-4:4:latest \\\n"
        f"  --size Standard_NV36ads_A10_v5 \\\n"
        f"  --admin-username azureuser \\\n"
        f"  --generate-ssh-keys \\\n"
        f"  --custom-data user_data.sh"
    )

    return (
        f"### 🚀 Azure VM Standard_NV36ads_A10_v5 (NVIDIA A10) vLLM Azure Linux 4.0 Deployment Config\n\n"
        f"#### 1. Custom Data Script (`user_data.sh`):\n"
        f"```bash\n{user_data}\n```\n\n"
        f"#### 2. Azure CLI Run Commands:\n"
        f"```bash\n{az_cmd}\n```\n\n"
        f"#### 3. Prerequisites:\n"
        f'- Save your HF Token in Azure Key Vault: `az keyvault secret set --vault-name {AZURE_KEYVAULT_NAME} --name {HF_SECRET_ID} --value "your-token"`\n'
        f"- Ensure VM port `8080` is open in Network Security Group."
    )


@mcp.tool()
async def deploy_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    key_name: str = "unused",
    subnet_id: Optional[str] = None,
) -> str:
    """
    Deploys vLLM to Azure VM Standard_NV36ads_A10_v5 (NVIDIA A10) running Azure Linux 4.0.

    Args:
        service_name: Name or Name tag for the VM resource.
        model_path: Hugging Face repo ID or Blob URI.
        key_name: Not used for Azure (default SSH keys generated).
        subnet_id: Optional custom subnet ID/name.
    """
    hf_token = await get_secret() or ""
    user_data = f"""#!/bin/bash
dnf install -y moby-engine
systemctl start docker
systemctl enable docker
dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
dnf clean all
dnf install -y cuda-drivers nvidia-container-toolkit
systemctl restart docker
docker run -d --name vllm-server \\
  --gpus all \\
  --ipc=host \\
  --restart always \\
  -p 8080:8080 \\
  -e HF_TOKEN="{hf_token}" \\
  vllm/vllm-openai:nightly \\
  --model {model_path} \\
  --quantization compressed-tensors \\
  --dtype bfloat16 \\
  --max-model-len 32768 \\
  --disable-chunked-mm-input \\
  --gpu-memory-utilization 0.95 \\
  --kv-cache-dtype fp8 \\
  --tensor-parallel-size 1 \\
  --max-num-seqs 8 \\
  --enable-chunked-prefill \\
  --max-num-batched-tokens 4096 \\
  --enable-auto-tool-choice \\
  --tool-call-parser gemma4 \\
  --reasoning-parser gemma4 \\
  --async-scheduling \\
  --limit-mm-per-prompt '{{}}' \\
  --host 0.0.0.0 \\
  --port 8080
"""
    tmp_path = "/tmp/azure_user_data.sh"
    with open(tmp_path, "w") as f:
        f.write(user_data)

    try:
        rg_name = f"{service_name}-rg"
        cmd_rg = ["az", "group", "create", "--name", rg_name, "--location", AZURE_LOCATION]
        proc_rg = await asyncio.to_thread(subprocess.run, cmd_rg, capture_output=True, text=True, timeout=30)
        if proc_rg.returncode != 0:
            return f"Failed to create resource group:\n{proc_rg.stderr.strip()}"

        vm_name = f"{service_name}-vm"
        cmd_vm = [
            "az",
            "vm",
            "create",
            "--resource-group",
            rg_name,
            "--name",
            vm_name,
            "--image",
            "microsoftazurelinux:azurelinux-4:4:latest",
            "--size",
            "Standard_NV36ads_A10_v5",
            "--admin-username",
            "azureuser",
            "--generate-ssh-keys",
            "--custom-data",
            tmp_path,
        ]
        if subnet_id:
            cmd_vm.extend(["--subnet", subnet_id])

        proc_vm = await asyncio.to_thread(subprocess.run, cmd_vm, capture_output=True, text=True, timeout=300)
        if proc_vm.returncode != 0:
            return f"Failed to deploy Azure VM:\n{proc_vm.stderr.strip()}"

        cmd_nsg = ["az", "vm", "open-port", "--resource-group", rg_name, "--name", vm_name, "--port", "8080"]
        await asyncio.to_thread(subprocess.run, cmd_nsg, capture_output=True, text=True, timeout=30)

        cmd_ip = [
            "az",
            "vm",
            "list-ip-addresses",
            "--resource-group",
            rg_name,
            "--name",
            vm_name,
            "--query",
            "[0].virtualMachine.network.publicIpAddresses[0].ipAddress",
            "-o",
            "tsv",
        ]
        proc_ip = await asyncio.to_thread(subprocess.run, cmd_ip, capture_output=True, text=True, timeout=15)
        ip = proc_ip.stdout.strip() if proc_ip.returncode == 0 else "None"

        return (
            f"🚀 Successfully requested Azure VM Standard_NV36ads_A10_v5 Deployment for service '{service_name}'.\n"
            f"Resource Group: `{rg_name}`\n"
            f"VM Name: `{vm_name}`\n"
            f"Public IP: `{ip}`\n"
            f"Please wait a few minutes for the instance to initialize and pull the vLLM docker image."
        )
    except Exception as e:
        return f"Failed to deploy Azure VM:\nError: {str(e)}"
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@mcp.tool()
async def start_azure_vm(
    service_name: str = DEFAULT_SERVICE_NAME,
    model_path: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    key_name: str = "unused",
    subnet_id: Optional[str] = None,
    instance_type: str = "Standard_NV36ads_A10_v5",
    market_type: str = "on-demand",
    instance_id: Optional[str] = None,
) -> str:
    """
    Starts an existing stopped Azure VM, or provisions a new one with NVIDIA A10 GPU if none exists.

    Args:
        service_name: Tag/Name prefix for the Azure resources.
        model_path: Hugging Face repo ID or Blob URI.
        key_name: Not used for Azure (default SSH keys generated).
        subnet_id: Optional custom subnet ID/name.
        instance_type: Azure VM size (default: 'Standard_NV36ads_A10_v5').
        market_type: Market type (Azure spot or on-demand).
        instance_id: Not used for Azure (defaults to service_name mapping).
    """
    rg_name = f"{service_name}-rg"
    vm_name = f"{service_name}-vm"

    # Check if VM already exists and its powerState
    try:
        cmd_show = ["az", "vm", "show", "-g", rg_name, "-n", vm_name, "-d", "--query", "powerState", "-o", "tsv"]
        proc_show = await asyncio.to_thread(subprocess.run, cmd_show, capture_output=True, text=True, timeout=15)
        if proc_show.returncode == 0 and proc_show.stdout.strip():
            state = proc_show.stdout.strip()
            if "stopped" in state.lower() or "deallocated" in state.lower():
                cmd_start = ["az", "vm", "start", "-g", rg_name, "-n", vm_name]
                await asyncio.to_thread(subprocess.run, cmd_start, capture_output=True, text=True, timeout=60)
                return f"🚀 Successfully requested start for existing stopped Azure VM: {vm_name} in group {rg_name}"
            elif "running" in state.lower():
                return f"ℹ️ Azure VM {vm_name} is already running."
    except Exception as e:
        logger.info(f"Checking existing Azure VM returned: {e}")

    # Otherwise, deploy new one
    return await deploy_vllm(service_name=service_name, model_path=model_path, key_name=key_name, subnet_id=subnet_id)


@mcp.tool()
async def destroy_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Cleans up the vLLM Docker container on the Azure VM matching the service name,
    without deleting/deallocating the VM itself.

    Args:
        service_name: Name tag of the VM resources to clean up.
    """
    rg_name = f"{service_name}-rg"
    vm_name = f"{service_name}-vm"

    try:
        cmd = [
            "az",
            "vm",
            "run-command",
            "invoke",
            "-g",
            rg_name,
            "-n",
            vm_name,
            "--command-id",
            "RunShellScript",
            "--scripts",
            "docker stop vllm-server || true",
            "docker rm vllm-server || true",
        ]
        process = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=60)
        if process.returncode == 0:
            return f"🧹 Successfully requested cleanup of the 'vllm-server' Docker container on Azure VM '{vm_name}' (VM remains running)."
        else:
            return f"Failed to clean up container on Azure VM '{vm_name}':\n{process.stderr.strip()}"
    except Exception as e:
        return f"Failed to clean up container for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def stop_azure_vm(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Stops (deallocates) the Azure VM by service name prefix.

    Args:
        service_name: Name prefix of the VM resources to stop (optional).
        instance_id: Not used for Azure (defaults to service_name mapping).
    """
    target_name = service_name or DEFAULT_SERVICE_NAME
    rg_name = f"{target_name}-rg"
    vm_name = f"{target_name}-vm"

    try:
        # deallocate VM to stop billing for compute resources
        cmd = ["az", "vm", "deallocate", "-g", rg_name, "-n", vm_name, "--no-wait"]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return f"🛑 Successfully requested stopping (deallocation) for Azure VM: {vm_name} in group {rg_name}"
    except Exception as e:
        return f"Failed to stop Azure VM {vm_name}:\nError: {str(e)}"


@mcp.tool()
def status_vllm(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Checks the status of the Azure VM matching the service name.

    Args:
        service_name: Name prefix of the VM to check.
    """
    rg_name = f"{service_name}-rg"
    vm_name = f"{service_name}-vm"

    try:
        cmd = [
            "az",
            "vm",
            "show",
            "-g",
            rg_name,
            "-n",
            vm_name,
            "-d",
            "--query",
            "{Name:name, State:powerState, IP:publicIps, Size:hardwareProfile.vmSize}",
            "-o",
            "json",
        ]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0:
            vm = json.loads(process.stdout.strip())
            info = (
                f"- **VM Name**: `{vm['Name']}`\n"
                f"  - **Size**: `{vm['Size']}`\n"
                f"  - **State**: `{vm['State']}`\n"
                f"  - **Public IP**: `{vm.get('IP', 'None')}`\n"
            )
            return f"### Azure VM Status for service prefix '{service_name}':\n\n" + info
        else:
            return f"No Azure VM found matching service name prefix '{service_name}' (Resource Group: {rg_name})."
    except Exception as e:
        return f"Failed to get status for service '{service_name}':\nError: {str(e)}"


@mcp.tool()
def status_azure_vm(
    service_name: Optional[str] = None,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of Azure VM by service name prefix.

    Args:
        service_name: Name prefix of the VM to check (optional).
        instance_id: Not used for Azure.
    """
    target_name = service_name or DEFAULT_SERVICE_NAME
    return status_vllm(target_name)


@mcp.tool()
async def check_vllm(
    service_name: str = DEFAULT_SERVICE_NAME,
    instance_id: Optional[str] = None,
) -> str:
    """
    Checks the status of the vLLM container and engine running on the Azure VM.

    Args:
        service_name: Name prefix of the VM to check.
        instance_id: Not used for Azure.
    """
    rg_name = f"{service_name}-rg"
    vm_name = f"{service_name}-vm"

    try:
        cmd_show = [
            "az",
            "vm",
            "show",
            "-g",
            rg_name,
            "-n",
            vm_name,
            "-d",
            "--query",
            "{State:powerState, IP:publicIps}",
            "-o",
            "json",
        ]
        proc_show = await asyncio.to_thread(subprocess.run, cmd_show, capture_output=True, text=True, timeout=15)
        if proc_show.returncode != 0:
            return f"No active Azure VM found matching service tag '{service_name}'."

        vm = json.loads(proc_show.stdout.strip())
        state = vm["State"]
        ip = vm.get("IP")

        report = f"### 🖥️ VM: `{vm_name}` ({state})\n"
        if "running" not in state.lower():
            report += f"❌ VM is not running (Current State: `{state}`). Skipping container checks.\n"
            return report

        if not ip:
            report += "⚠️ No Public IP associated with this running VM.\n"
            return report

        # 2. Check Docker Container status via VM Run Command
        docker_status = "Unknown"
        try:
            cmd_run = [
                "az",
                "vm",
                "run-command",
                "invoke",
                "-g",
                rg_name,
                "-n",
                vm_name,
                "--command-id",
                "RunShellScript",
                "--scripts",
                "docker inspect -f '{{.State.Status}}' vllm-server 2>&1",
            ]
            proc_run = await asyncio.to_thread(subprocess.run, cmd_run, capture_output=True, text=True, timeout=30)
            if proc_run.returncode == 0:
                res = json.loads(proc_run.stdout.strip())
                if res.get("value") and len(res["value"]) > 0:
                    docker_status = res["value"][0].get("message", "").strip()
            else:
                docker_status = f"Failed (CLI error: {proc_run.stderr.strip()})"
        except Exception as e:
            docker_status = f"Error querying VM: {str(e)}"

        report += f"- **Docker Container (`vllm-server`)**: `{docker_status}`\n"

        # 3. Check vLLM HTTP health endpoint
        http_status = "Unreachable"
        try:
            async with httpx.AsyncClient(timeout=3) as http_client:
                res = await http_client.get(f"http://{ip}:8080/health")
                if res.status_code == 200:
                    http_status = "Healthy ✅"
                else:
                    http_status = f"Unhealthy (HTTP Code: {res.status_code}) ❌"
        except Exception as e:
            http_status = f"Unreachable (Error: {e}) ❌"

        report += f"- **vLLM API Endpoint (`http://{ip}:8080/health`)**: `{http_status}`\n"
        return report
    except Exception as e:
        return f"Failed to describe Azure VM status:\nError: {str(e)}"


@mcp.tool()
def update_vllm_scaling(instance_type: str, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Updates the Azure VM size (scaling vertically) for the vLLM VM service.

    Args:
        instance_type: The new Azure VM size (e.g. 'Standard_NV72ads_A10_v5').
        service_name: The Name prefix of the VM to scale.
    """
    rg_name = f"{service_name}-rg"
    vm_name = f"{service_name}-vm"

    try:
        cmd_show = [
            "az",
            "vm",
            "show",
            "-g",
            rg_name,
            "-n",
            vm_name,
            "--query",
            "{Size:hardwareProfile.vmSize, State:powerState}",
            "-o",
            "json",
        ]
        process = subprocess.run(cmd_show, capture_output=True, text=True, timeout=15)
        if process.returncode != 0:
            return f"No active Azure VM found matching service tag '{service_name}'."

        vm = json.loads(process.stdout.strip())
        current_size = vm["Size"]

        # Apply Resize
        cmd_resize = ["az", "vm", "resize", "-g", rg_name, "-n", vm_name, "--size", instance_type, "--no-wait"]
        subprocess.run(cmd_resize, capture_output=True, text=True, timeout=15)
        return (
            f"⚡ Successfully requested scale-up of Azure VM `{vm_name}` from `{current_size}` to `{instance_type}`.\n"
            f"The VM will automatically apply the changes and restart if needed."
        )
    except Exception as e:
        return f"Failed to scale Azure VM:\nError: {str(e)}"


@mcp.tool()
def get_vllm_gpu_deployment_config(
    cluster_name: str = "gpu-cluster",
    model_name: str = "google/gemma-4-12B-it-qat-w4a16-ct",
) -> str:
    """
    Generates a GKE (Google Kubernetes Engine) deployment manifest and node pool command for GPU (NVIDIA L4).

    Args:
        cluster_name: The name of the GKE cluster.
        model_name: The model identifier (e.g., 'google/gemma-4-12B-it-qat-w4a16-ct').
    """
    manifest = f"""
### 🌀 vLLM on GKE GPU (GCP Deployment)

To deploy vLLM on GKE GPUs, use the following GKE node pool creation command and Kubernetes manifest. This configuration targets a single **NVIDIA L4 GPU** (g2-standard-4).

#### 1. Create a GPU Node Pool via Google Cloud CLI
```bash
gcloud container node-pools create gpu-l4 \\
    --cluster={cluster_name} \\
    --location={LOCATION} \\
    --machine-type=g2-standard-4 \\
    --accelerator=type=nvidia-l4,count=1 \\
    --num-nodes=1
```

#### 2. Kubernetes Manifest (vllm-gpu.yaml)
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-gpu
  template:
    metadata:
      labels:
        app: vllm-gpu
    spec:
      nodeSelector:
        cloud.google.com/gke-gpu: "true"
      containers:
      - name: vllm-gpu
        image: vllm/vllm-openai:nightly
        resources:
          limits:
            nvidia.com/gpu: "1"
          requests:
            nvidia.com/gpu: "1"
        command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
        args:
        - "--model={model_name}"
        - "--gpu-memory-utilization=0.95"
        - "--max-model-len=32768"
        ports:
        - containerPort: 8080
        volumeMounts:
        - name: dshm
          mountPath: /dev/shm
      volumes:
      - name: dshm
        emptyDir:
          medium: Memory
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-gpu-service
spec:
  selector:
    app: vllm-gpu
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8080
  type: ClusterIP
```

#### 3. Deployment Steps
1. Save the YAML above to `vllm-gpu.yaml`.
2. Apply it: `kubectl apply -f vllm-gpu.yaml`.
3. Ensure the NVIDIA GPU driver installer is running on your GKE cluster if not using Container-Optimized OS (COS) with auto-installation enabled.
"""
    return manifest


@mcp.tool()
def get_vertex_ai_model_copy_instructions(model_name: str = "gemma-4-12B-it-qat-w4a16-ct") -> str:
    """
    Provides instructions and commands to transfer Gemma model artifacts from Vertex AI Model Garden to your GCS bucket.
    """
    instructions = f"""
### 🚀 Transferring {model_name} from Vertex AI Model Garden

To use vLLM with Cloud Storage FUSE without Hugging Face, follow these steps:

1. **Accept Terms:** Go to the Vertex AI Model Garden page for Gemma (https://console.cloud.google.com/vertex-ai/publishers/google/model-garden/335) and click 'Accept' on the license agreement.
2. **Download via Signed URL:** After accepting, the console provides a 'Download' button or a signed URL.
3. **Transfer to GCS:**
   If you have the artifacts locally after downloading from the console, use:
   `gcloud storage cp -r ./model_artifacts/* gs://{BUCKET_NAME}/{model_name}/`

4. **Alternative (Direct GCS Copy):**
   Google occasionally provides a managed GCS path for verified projects. If accessible, you can use:
   `gcloud storage cp -r gs://vertex-ai-models/gemma/{model_name}/* gs://{BUCKET_NAME}/{model_name}/`

Once the artifacts are in your bucket, use `get_vllm_deployment_config` to generate your Cloud Run deployment command.
"""
    return instructions


@mcp.tool()
async def get_huggingfacehub_download_path(
    repo_id: str = "google/gemma-4-12B-it-qat-w4a16-ct",
) -> str:
    """
    Returns the local cache path for a Hugging Face model using huggingface_hub.
    Note: This may trigger a download if the model is not already in the cache.
    """
    try:
        from huggingface_hub import snapshot_download

        token = await get_secret() or os.getenv("HF_TOKEN")
        # Run synchronous snapshot_download in a separate thread to avoid blocking the async event loop
        path = await asyncio.to_thread(snapshot_download, repo_id=repo_id, token=token)
        return f"Model '{repo_id}' is available at: {path}"
    except Exception as e:
        return f"Error resolving huggingface_hub path: {str(e)}"


@mcp.tool()
def get_huggingface_model_copy_instructions(
    repo_id: str = "google/gemma-4-12B-it-qat-w4a16-ct",
    bucket_name: Optional[str] = None,
) -> str:
    """
    Provides instructions and commands to transfer Gemma model weights from Hugging Face to your S3 or GCS bucket.

    Args:
        repo_id: The Hugging Face repo ID (e.g., 'google/gemma-4-12B-it-qat-w4a16-ct').
        bucket_name: The target bucket name (defaults to AWS_BUCKET_NAME or BUCKET_NAME).
    """
    if not bucket_name:
        bucket_name = AWS_BUCKET_NAME if os.getenv("AWS_ACCESS_KEY_ID") else BUCKET_NAME

    model_name = repo_id.split("/")[-1]
    is_aws = os.getenv("AWS_ACCESS_KEY_ID") is not None

    if is_aws:
        instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to AWS S3

To use Hugging Face weights with vLLM on EC2 via S3, follow these steps:

#### Option A: Download directly inside EC2 instance (Recommended)
You don't need to copy weights to S3 if you specify the Hugging Face Repo ID directly in `deploy_vllm`.
vLLM will download it automatically from Hugging Face when starting, using your `HF_TOKEN`.

#### Option B: Upload to S3 Bucket
If you prefer to host weights privately in S3:

1. **Download Model locally:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to S3:**
   The command above outputs the local path. Use it to copy the artifacts:
   `aws s3 cp --recursive /path/to/downloaded/model/ s3://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`deploy_vllm(model_path="s3://{bucket_name}/{model_name}/")`
"""
    else:
        instructions = f"""
### 📦 Transferring {model_name} from Hugging Face to GCS

To use Hugging Face weights with vLLM on Cloud Run via GCS FUSE, follow these steps:

#### Option A: Using `huggingface_hub` Python Library (Recommended)
`huggingface_hub` simplifies the download process and can be run directly from python:

1. **Download Model:**
   `python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('{repo_id}'))"`

2. **Upload to GCS:**
   The command above outputs the local path. Use it to copy the artifacts:
   `gcloud storage cp -r /path/to/downloaded/model/* gs://{bucket_name}/{model_name}/`

#### Option B: Using `huggingface-cli`
1. **Setup Hugging Face CLI:**
   `pip install huggingface_hub`
   `huggingface-cli login`

2. **Download Model Artifacts:**
   `huggingface-cli download {repo_id} --local-dir ./{model_name}`

3. **Upload to GCS Bucket:**
   `gcloud storage cp -r ./{model_name}/* gs://{bucket_name}/{model_name}/`

Once uploaded, you can deploy using:
`get_vllm_deployment_config(model_path="{model_name}")`
"""
    return instructions


@mcp.tool()
def check_gpu_quotas(region: Optional[str] = None) -> str:
    """
    Checks GPU quotas for a specific Azure region, AWS region, or Google Cloud region.

    Args:
        region: The region to check (defaults to AZURE_LOCATION/LOCATION).
    """
    if not region:
        region = AZURE_LOCATION

    # Try Azure VM usage
    is_azure = not region.startswith("gs://") and ("-" not in region or region == AZURE_LOCATION)
    if is_azure:
        try:
            cmd = [
                "az",
                "vm",
                "list-usage",
                "--location",
                region,
                "--query",
                "[?limit != `0` && (contains(localName, 'NV') || contains(localName, 'GPU') || contains(localName, 'Cores'))]",
                "-o",
                "json",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            if process.returncode == 0:
                res = json.loads(process.stdout.strip())
                report = f"### 📊 Azure VM GPU/Core Quotas for region `{region}`\n\n"
                for item in res:
                    report += f"- **{item.get('localName')}**:\n"
                    report += f"  - Limit: `{item.get('limit')}`\n"
                    report += f"  - Current Usage: `{item.get('currentValue')}`\n"
                return report
        except Exception as e:
            logger.warning(f"Failed to check Azure VM quotas in `{region}`: {e}")

    # Fallback/alternative: check AWS or GCP
    # If it looks like an AWS region or AWS credentials exist, try AWS Service Quotas
    is_aws = "-" in region and not region.startswith("us-east4")
    if is_aws or os.getenv("AWS_ACCESS_KEY_ID"):
        try:
            import boto3

            sq = boto3.client("service-quotas", region_name=region)
            quotas_to_check = [
                {"QuotaCode": "L-DB2E81BA", "Name": "Running On-Demand G and VT instances"},
                {"QuotaCode": "L-3819A6DF", "Name": "All G and VT Spot Instance Requests"},
                {"QuotaCode": "L-417A2355", "Name": "Running On-Demand P instances"},
            ]
            report = f"### 📊 AWS EC2 GPU Quotas for region `{region}`\n\n"
            for q in quotas_to_check:
                try:
                    res = sq.get_service_quota(ServiceCode="ec2", QuotaCode=q["QuotaCode"])
                    quota = res["Quota"]
                    report += f"- **{q['Name']}** ({quota['QuotaName']}):\n"
                    report += f"  - Limit: `{quota['Value']}` (vCPUs)\n"
                    report += f"  - Adjustable: `{quota['Adjustable']}`\n"
                except Exception as e:
                    report += f"- **{q['Name']}** ({q['QuotaCode']}): Could not fetch quota ({str(e)})\n"
            return report
        except Exception as e:
            logger.warning(f"Failed to check AWS EC2 quotas in `{region}`: {e}")

    # Fallback to GCP
    cmd = [
        "gcloud",
        "compute",
        "regions",
        "describe",
        region,
        f"--project={PROJECT_ID}",
        "--format=json(quotas)",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        quotas = data.get("quotas", [])

        # Filter for GPU quotas
        gpu_quotas = []
        for q in quotas:
            metric = q.get("metric", "")
            if "GPU" in metric or "ACCELERATOR" in metric:
                gpu_quotas.append(f"- **{metric}**:\n  - Limit: `{q.get('limit')}`\n  - Usage: `{q.get('usage')}`")

        if not gpu_quotas:
            return f"No GPU/Accelerator quotas found in GCP region `{region}`."

        return f"### 📊 GCP GPU Quotas for region `{region}`\n\n" + "\n".join(gpu_quotas)

    except subprocess.CalledProcessError as e:
        return f"Failed to retrieve GPU quotas for region `{region}`:\nError: {e.stderr}\nOutput: {e.stdout}"
    except Exception as e:
        return f"Error checking GPU quotas: {str(e)}"


@mcp.tool()
async def verify_model_health() -> str:
    """Runs a deep health check with latency reporting on the Cloud Run GPU-hosted model."""
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        start_time = time.monotonic()
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": "Hello, is the model working?"}],
            model=model_name,
            max_tokens=200,
        )
        end_time = time.monotonic()
        latency = end_time - start_time
        response_content = chat_completion.choices[0].message.content

        if response_content:
            return (
                f"✅ Model health check PASSED.\n"
                f"Model: {model_name}\n"
                f"Response: '{response_content[:50]}...'\n"
                f"Latency: {latency:.2f} seconds."
            )
        else:
            return "❌ Model health check FAILED: Empty response."
    except Exception as e:
        return f"❌ Model health check FAILED: {e}"


@mcp.tool()
async def query_gemma4(prompt: str) -> str:
    """Queries the self-hosted Gemma 4 model on Cloud Run."""
    logger.info(f"Querying Cloud Run model with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)
        chat_completion = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
        )
        response = chat_completion.choices[0].message.content or "No response from model."
        logger.info(f"Model response: '{response[:100]}...'")
        return response
    except Exception as e:
        logger.error(f"Error querying model: {e}")
        return f"❌ An error occurred while querying the model: {e}"


@mcp.tool()
async def query_gemma4_with_stats(prompt: str) -> str:
    """
    Queries the self-hosted Gemma 4 model on Cloud Run and returns detailed performance statistics.

    This tool provides:
    - The full model response.
    - Time to First Token (TTFT).
    - Total generation time.
    - Tokens per second.
    """
    logger.info(f"Querying model with stats with prompt: '{prompt[:50]}...'")
    try:
        client = await get_vllm_client()
        model_name = await get_active_model_name(client)

        start_time = time.monotonic()
        ttft = None
        response_content = ""
        total_tokens = 0

        stream = await client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=model_name,
            stream=True,
        )

        async for chunk in stream:
            if ttft is None:
                ttft = time.monotonic() - start_time

            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response_content += content
                total_tokens += 1  # Rough token count

        end_time = time.monotonic()
        total_time = end_time - start_time

        if not response_content:
            return "❌ Model returned an empty response."

        tokens_per_second = total_tokens / (total_time - ttft) if ttft and total_time > ttft else 0

        stats_report = (
            f"### 📊 Performance Stats\n"
            f"- **Model:** `{model_name}`\n"
            f"- **Time to First Token (TTFT):** `{ttft:.3f}s`\n"
            f"- **Total Generation Time:** `{total_time:.3f}s`\n"
            f"- **Tokens per Second:** `{tokens_per_second:.2f} tokens/s`\n"
            f"- **Total Tokens (approx.):** `{total_tokens}`\n"
            f"\n### 💬 Model Response\n"
            f"{response_content}"
        )

        logger.info(f"Model response with stats: TTFT={ttft:.3f}s, TotalTime={total_time:.3f}s")
        return stats_report

    except Exception as e:
        logger.error(f"Error querying model with stats: {e}")
        return f"❌ An error occurred while querying the model with stats: {e}"


@mcp.tool()
async def get_model_details() -> str:
    """Retrieves detailed information about the running Cloud Run model, engine, and versions."""
    report = ""
    try:
        vllm_url = get_vllm_url()
        report += f"### 🧩 Model Details ({vllm_url})\n\n"
        client = await get_vllm_client()

        # 1. Get Model Details from /v1/models
        try:
            models_res = await client.models.list()
            report += "**Model Information (`/v1/models`):**\n"
            models_list = [{"id": m.id, "object": m.object, "owned_by": m.owned_by} for m in models_res.data]
            report += f"```json\n{json.dumps(models_list, indent=2)}\n```\n"
        except Exception as e:
            report += f"❌ Error fetching model details via client: {e}\n\n"

        # 2. Get Health Status
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        async with httpx.AsyncClient(timeout=10) as http_client:
            try:
                res = await http_client.get(f"{vllm_url}/health", headers=headers)
                if res.status_code == 200:
                    report += "**Health Status (`/health`):**\n- Status: `Healthy` ✅\n\n"
                else:
                    report += f"**Health Status (`/health`):**\n- Status: `Unhealthy` (Code: {res.status_code}) ❌\n\n"
            except Exception as e:
                report += f"**Health Status (`/health`):**\n- Status: `Unreachable` (Error: {e}) ❌\n\n"
    except Exception as e:
        report += f"❌ Error retrieving system URL or auth token: {e}"

    return report


@mcp.tool()
async def get_system_status(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Provides a high-level dashboard of Azure VM system status and vLLM health, or GCP Cloud Run.

    Args:
        service_name: The name prefix of the Azure VM resource or Cloud Run service name.
    """
    health = "🔴 Offline"
    url = None
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                health = f"🟢 Online ({url})"
            else:
                health = f"🔴 Offline (Status {res.status_code}) ({url})"
    except Exception as e:
        logger.warning(f"Health check failed: {e}")

    # Check Azure VM
    vm_status = "🔴 Unknown"
    is_azure = True
    try:
        rg_name = f"{service_name}-rg"
        vm_name = f"{service_name}-vm"
        cmd = ["az", "vm", "show", "-g", rg_name, "-n", vm_name, "-d", "--query", "powerState", "-o", "tsv"]
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if process.returncode == 0 and process.stdout.strip():
            state = process.stdout.strip()
            if "running" in state.lower():
                vm_status = f"🟢 Running ({vm_name})"
            else:
                vm_status = f"🔴 {state.capitalize()} ({vm_name})"
        else:
            vm_status = "🔴 VM Not Found / Offline"
    except Exception as e:
        vm_status = f"🔴 Azure Error: {str(e)}"

    # If Azure failed or not found, try GCP fallback
    if "Error" in vm_status or "Not Found" in vm_status:
        try:
            cmd = [
                "gcloud",
                "run",
                "services",
                "describe",
                service_name,
                f"--project={PROJECT_ID}",
                f"--region={LOCATION}",
                "--format=json",
            ]
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if process.returncode == 0:
                data = json.loads(process.stdout)
                conditions = data.get("status", {}).get("conditions", [])
                ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
                if ready_cond and ready_cond.get("status") == "True":
                    vm_status = "🟢 GCP Ready"
                elif ready_cond:
                    vm_status = f"🔴 GCP Not Ready ({ready_cond.get('status')})"
                else:
                    vm_status = "🔴 GCP Not Ready (No Ready condition)"
            else:
                vm_status = f"🔴 GCP Error checking service ({process.stderr.strip()})"
        except Exception as e:
            vm_status = f"🔴 GCP Error: {str(e)}"

    if "🟢" in health:
        next_step = "Use `query_gemma4` to interact with the model."
    else:
        provider = "Azure VM" if is_azure else "Cloud Run service"
        next_step = f"Call `deploy_vllm` to provision/start the {provider} `{service_name}`."

    return (
        f"### 🌀 GPU vLLM System Status\n"
        f"- **vLLM Health:** {health}\n"
        f"- **Hosting Status:** {vm_status}\n"
        f"**👉 Next Step:** {next_step}"
    )


@mcp.tool()
async def get_endpoint(service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Returns the active vLLM service URL if available.

    Args:
        service_name: The name of the service or instance Name tag to query.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(f"{url}/health", headers=headers)
            if res.status_code == 200:
                return f"🟢 vLLM is Online at: {url}"
            else:
                return f"🔴 vLLM is configured at {url} but returned status {res.status_code}."
    except Exception as e:
        return f"🔴 vLLM endpoint check failed: {e}. Try deploying/starting it with `deploy_vllm`."


@mcp.tool()
async def run_benchmark(
    model: Optional[str] = None,
    num_prompts: int = 20,
    random_output_len: int = 128,
    max_concurrency: int = 8,
) -> str:
    """
    Runs a performance/concurrency benchmark sweep against the Cloud Run vLLM GPU endpoint.

    Args:
        model: Model name to request (defaults to the active model from /v1/models).
        num_prompts: Number of requests to send per concurrency level.
        random_output_len: Max tokens to generate per request.
        max_concurrency: Maximum concurrency level to sweep up to (powers of 2, e.g. 1, 2, 4, 8).
    """
    from datetime import datetime

    try:
        url = get_vllm_url()
        token = get_auth_token()
    except Exception as e:
        return f"❌ Cannot run benchmark: {e}"

    # Get active model name if not provided
    client = await get_vllm_client()
    if not model:
        model = await get_active_model_name(client)

    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    base_url = f"{url.rstrip('/')}/v1/completions"
    prompt = "Explain the importance of Site Reliability Engineering for large scale AI deployments."

    concurrencies = []
    c = 1
    while c <= max_concurrency:
        concurrencies.append(c)
        c *= 2
    if max_concurrency not in concurrencies:
        concurrencies.append(max_concurrency)

    results = []

    async def send_request(http_client: httpx.AsyncClient, sem: asyncio.Semaphore) -> dict:
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": random_output_len,
            "temperature": 0.0,
            "stream": False,
        }
        async with sem:
            start_time = time.perf_counter()
            try:
                response = await http_client.post(base_url, json=payload, headers=headers, timeout=120)
                end_time = time.perf_counter()
                if response.status_code == 200:
                    latency = end_time - start_time
                    data = response.json()
                    tokens = data.get("usage", {}).get("completion_tokens", random_output_len)
                    return {"success": True, "latency": latency, "tokens": tokens}
                else:
                    return {"success": False, "error": f"Status {response.status_code}"}
            except Exception as e:
                return {"success": False, "error": str(e)}

    # Warmup
    logger.info("Warming up model for benchmark...")
    async with httpx.AsyncClient() as http_client:
        await send_request(http_client, asyncio.Semaphore(1))

    logger.info(f"Starting GPU benchmark sweep against {url} with model {model}...")
    for concurrency in concurrencies:
        logger.info(f"Running sweep with concurrency={concurrency}...")
        sem = asyncio.Semaphore(concurrency)

        async with httpx.AsyncClient() as http_client:
            start_batch = time.perf_counter()
            tasks = [send_request(http_client, sem) for _ in range(num_prompts)]
            batch_results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_batch

        successes = [r for r in batch_results if r["success"]]
        latencies = [r["latency"] for r in successes]

        if not latencies:
            results.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "concurrency": concurrency,
                    "total_requests": num_prompts,
                    "success_rate": 0.0,
                    "avg_latency": 0.0,
                    "p95_latency": 0.0,
                    "req_per_sec": 0.0,
                    "tokens_per_sec": 0.0,
                }
            )
            continue

        avg_lat = statistics.mean(latencies)
        p95_lat = sorted(latencies)[int(len(latencies) * 0.95)]
        throughput = len(successes) / total_time
        tokens_per_sec = sum(r["tokens"] for r in successes) / total_time

        results.append(
            {
                "timestamp": datetime.now().isoformat(),
                "concurrency": concurrency,
                "total_requests": num_prompts,
                "success_rate": len(successes) / num_prompts,
                "avg_latency": avg_lat,
                "p95_latency": p95_lat,
                "req_per_sec": throughput,
                "tokens_per_sec": tokens_per_sec,
            }
        )

    # Save to CSV
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.csv")
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "concurrency",
                "total_requests",
                "success_rate",
                "avg_latency",
                "p95_latency",
                "req_per_sec",
                "tokens_per_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    summary_str = f"### 📊 GPU Benchmark Results (Model: `{model}`)\n\n"
    summary_str += "| Concurrency | Success Rate | Req/s | Tokens/s | Avg Latency | P95 Latency |\n"
    summary_str += "|---:|---:|---:|---:|---:|---:|\n"
    for r in results:
        summary_str += f"| {r['concurrency']} | {r['success_rate']:.1%} | {r['req_per_sec']:.2f} | {r['tokens_per_sec']:.2f} | {r['avg_latency']:.2f}s | {r['p95_latency']:.2f}s |\n"
    summary_str += f"\n\nResults saved to `{output_file}`"
    return summary_str


async def fetch_azure_vm_logs(instance_id: str, limit: int = 50) -> str:
    """Fetches docker logs from the running Azure VM via VM Run Command."""
    # instance_id behaves as service_name here to map to correct RG/VM
    rg_name = f"{instance_id}-rg"
    vm_name = f"{instance_id}-vm"
    try:
        cmd = [
            "az",
            "vm",
            "run-command",
            "invoke",
            "-g",
            rg_name,
            "-n",
            vm_name,
            "--command-id",
            "RunShellScript",
            "--scripts",
            f"docker logs --tail {limit} vllm-server 2>&1",
        ]
        process = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=30)
        if process.returncode == 0:
            res = json.loads(process.stdout.strip())
            if res.get("value") and len(res["value"]) > 0:
                return res["value"][0].get("message", "").strip()
        return f"Failed to fetch logs: {process.stderr.strip()}"
    except Exception as e:
        return f"Failed to fetch logs via VM run command: {str(e)}"


@mcp.tool()
async def analyze_gpu_logs(limit: int = 15, service_name: str = DEFAULT_SERVICE_NAME) -> str:
    """
    Fetches vLLM logs for the specified Azure VM and uses Gemma 4 to analyze them for errors.

    Args:
        limit: Number of log entries to fetch.
        service_name: Name of the VM service/group prefix.
    """
    try:
        rg_name = f"{service_name}-rg"
        vm_name = f"{service_name}-vm"
        cmd = ["az", "vm", "show", "-g", rg_name, "-n", vm_name, "-d", "--query", "powerState", "-o", "tsv"]
        process = await asyncio.to_thread(subprocess.run, cmd, capture_output=True, text=True, timeout=15)
        if process.returncode == 0 and "running" in process.stdout.strip().lower():
            logger.info(f"Fetching Azure VM logs for VM {vm_name} in group {rg_name}...")
            raw_logs = await fetch_azure_vm_logs(service_name, limit)
            # Prepare prompt for Gemma
            prompt = f"Analyze the following vLLM docker container logs and provide a high-level summary of critical issues:\n\n{raw_logs}\n\nSummary:"
            client = await get_vllm_client()
            model_name = await get_active_model_name(client)
            chat_completion = await client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=model_name,
                max_tokens=512,
                temperature=0.2,
            )
            response_text = chat_completion.choices[0].message.content or ""
            return f"### Azure VM Log Analysis (Self-Hosted vLLM)\n\n{response_text}"
    except Exception as e:
        logger.warning(f"Failed to fetch/analyze Azure VM logs: {e}")

    # Fallback to GCP
    filter_query = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{service_name}"'
    return await analyze_cloud_logging(filter_query, limit)


@mcp.tool()
async def get_help() -> str:
    """Provides help text and summarizes the configuration options and all available SRE/DevOps tools for this Azure/GCP MCP server."""
    return (
        "### 🛠️ Azure/GCP Gemma 4 SRE Agent Help & Configuration\n\n"
        "You can configure this MCP server using the following environment variables:\n\n"
        "**Azure Configuration:**\n"
        f"- **`AZURE_LOCATION`**: The Azure region for VM deployment.\n"
        f"  - *Current Value:* `{AZURE_LOCATION}`\n"
        f"- **`AZURE_KEYVAULT_NAME`**: Key Vault name used to store secrets.\n"
        f"  - *Current Value:* `{AZURE_KEYVAULT_NAME}`\n"
        f"- **`AZURE_STORAGE_ACCOUNT`**: Storage Account used to store model weights.\n"
        f"  - *Current Value:* `{AZURE_STORAGE_ACCOUNT}`\n\n"
        "**GCP Configuration:**\n"
        f"- **`GOOGLE_CLOUD_PROJECT`**: Your GCP Project ID.\n"
        f"  - *Current Value:* `{PROJECT_ID}`\n"
        f"- **`GOOGLE_CLOUD_LOCATION`**: The GCP Region for deployment.\n"
        f"  - *Current Value:* `{LOCATION}`\n"
        f"- **`BUCKET_NAME`**: GCS Bucket used to store model weights.\n"
        f"  - *Current Value:* `{BUCKET_NAME}`\n\n"
        "**General serving:**\n"
        f"- **`MODEL_NAME`**: Default Hugging Face repository or path.\n"
        f"  - *Current Value:* `{MODEL_NAME}`\n"
        f"- **`VLLM_BASE_URL`**: The explicit URL of your vLLM service. (If not set, it is auto-discovered via VM tags or Cloud Run)\n"
        f"  - *Current Value:* `{VLLM_BASE_URL or 'Not set (auto-discovering)'}`\n\n"
        "### ℹ️ Active Mode Summary\n"
        f"The server is running in **Azure VM / Cloud Run** mode.\n\n"
        "### 🧰 Available MCP Tools\n\n"
        "Below is a summary of the tools exposed by this SRE/DevOps agent:\n\n"
        "#### 🐳 Infrastructure & Deployment\n"
        "- **`start_azure_vm`**: Starts an existing stopped Azure VM, or provisions a new one (with NVIDIA A10 GPU) if none exists.\n"
        "- **`status_azure_vm`**: Checks the state, size, and public IP details of Azure VMs.\n"
        "- **`stop_azure_vm`**: Safely deallocates active Azure VMs to halt billing.\n"
        "- **`check_vllm`**: Checks the status of the vLLM container and engine running on the Azure VM.\n"
        "- **`deploy_vllm`**: Deploys vLLM to Azure VM Standard_NV36ads_A10_v5 (Ubuntu 22.04 LTS).\n"
        "- **`destroy_vllm`**: Cleans up the vLLM Docker container on the Azure VM without deleting it.\n"
        "- **`status_vllm`**: Checks the status of the Azure VM or Cloud Run vLLM service.\n"
        "- **`update_vllm_scaling`**: Scales Azure VM sizes vertically.\n"
        "- **`get_vllm_deployment_config`**: Generates the Azure VM deployment command and user data (Ubuntu 22.04 LTS).\n"
        "- **`get_vllm_gpu_deployment_config`**: Generates a GKE deployment manifest for GPU (NVIDIA L4).\n"
        "- **`get_vllm_endpoint`**: Returns the current active vLLM endpoint URL.\n"
        "- **`check_gpu_quotas`**: Checks GPU VM core family quotas for an Azure region.\n\n"
        "#### 📊 Model Management\n"
        "- **`list_vertex_models`**: Lists models in the Vertex AI Registry.\n"
        "- **`list_bucket_models`**: Lists model weights in GCS or Azure Blob Storage.\n"
        "- **`save_hf_token`**: Securely saves a Hugging Face API token to Azure Key Vault or GCP Secret Manager.\n"
        "- **`get_vertex_ai_model_copy_instructions`**: Instructions to copy model from Vertex AI Model Garden to GCS.\n"
        "- **`get_huggingface_model_copy_instructions`**: Instructions to download model from Hugging Face and upload to Azure Blob/GCS.\n"
        "- **`get_huggingfacehub_download_path`**: Resolves local cache path using huggingface_hub.\n\n"
        "#### 📊 Monitoring & Status\n"
        "- **`get_metrics`**: Fetches raw Prometheus metrics from the running vLLM service's /metrics endpoint.\n"
        "- **`get_system_status`**: Provides a high-level status dashboard of the service and health.\n"
        "- **`get_endpoint`**: Verifies connectivity and returns the active service URL.\n"
        "- **`get_model_details`**: Retrieves detailed model metadata and engine state from `/v1/models`.\n"
        "- **`verify_model_health`**: Deep health check by querying the model with a simple prompt and measuring latency.\n\n"
        "#### 📈 Performance & Benchmarking\n"
        "- **`run_benchmark`**: Runs performance/concurrency benchmark sweeps against the vLLM GPU endpoint.\n\n"
        "#### 💬 Interaction & Diagnostics\n"
        "- **`query_gemma4`**: Primary tool to query the self-hosted model with standard chat message format.\n"
        "- **`query_gemma4_with_stats`**: Queries the model and returns streaming performance statistics (TTFT, throughput).\n"
        "- **`query_vllm`**: Direct text completions querying tool.\n"
        "- **`analyze_cloud_logging`**: Fetches logs from Azure Monitor Log Analytics or GCP Logging and analyzes them using the model.\n"
        "- **`analyze_gpu_logs`**: Fetches service logs and uses Gemma 4 to analyze them for SRE/DevOps errors.\n"
        "- **`suggest_sre_remediation`**: Suggests remediation plans for SRE errors using the model.\n"
    )


@mcp.tool()
async def get_metrics() -> str:
    """
    Fetches the Prometheus metrics from the active vLLM service.
    """
    try:
        url = get_vllm_url()
        token = get_auth_token()
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(f"{url}/metrics", headers=headers)
            if res.status_code == 200:
                return res.text
            else:
                return f"🔴 Failed to retrieve metrics. Status code: {res.status_code}\n\nResponse:\n{res.text[:1000]}"
    except Exception as e:
        return f"🔴 Error fetching metrics: {e}"


if __name__ == "__main__":
    mcp.run()
