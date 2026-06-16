# Configuration - Update these or override via environment variables
AZURE_LOCATION ?= eastus
AZURE_KEYVAULT_NAME ?= vllm-devops-kv
SERVICE_NAME ?= gpu-12b-qat-l4-devops-agent
MODEL_PATH ?= google/gemma-4-12B-it-qat-w4a16-ct

install:
	pip install -r requirements.txt

run:
	python server.py

test:
	python test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Deploy the vLLM inference stack to Azure VM with Azure Linux 4.0
deploy:
	@echo "🚀 Creating Azure Resource Group $(SERVICE_NAME)-rg in $(AZURE_LOCATION)..."
	az group create --name $(SERVICE_NAME)-rg --location $(AZURE_LOCATION)
	@echo "🚀 Deploying Azure VM $(SERVICE_NAME)-vm with Azure Linux 4.0..."
	@if [ ! -f user_data.sh ]; then \
		echo '#!/bin/bash' > user_data.sh; \
		echo 'dnf install -y moby-engine' >> user_data.sh; \
		echo 'systemctl start docker' >> user_data.sh; \
		echo 'systemctl enable docker' >> user_data.sh; \
		echo 'dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo' >> user_data.sh; \
		echo 'dnf clean all' >> user_data.sh; \
		echo 'dnf install -y cuda-drivers nvidia-container-toolkit' >> user_data.sh; \
		echo 'systemctl restart docker' >> user_data.sh; \
		echo 'docker run -d --name vllm-server --gpus all --ipc=host --restart always -p 8080:8080 -e HF_TOKEN="$$(az keyvault secret show --name hf-token --vault-name $(AZURE_KEYVAULT_NAME) --query value -o tsv 2>/dev/null || echo \'\')" vllm/vllm-openai:nightly --model $(MODEL_PATH) --quantization compressed-tensors --dtype bfloat16 --max-model-len 32768 --disable-chunked-mm-input --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --tensor-parallel-size 1 --max-num-seqs 8 --enable-chunked-prefill --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --async-scheduling --limit-mm-per-prompt '\''{}'\'' --host 0.0.0.0 --port 8080' >> user_data.sh; \
	fi
	az vm create \
		--resource-group $(SERVICE_NAME)-rg \
		--name $(SERVICE_NAME)-vm \
		--image microsoftazurelinux:azurelinux-4:4:latest \
		--size Standard_NV36ads_A10_v5 \
		--admin-username azureuser \
		--generate-ssh-keys \
		--custom-data user_data.sh

# Destroy the vLLM inference stack on Azure
destroy:
	@echo "🗑️ Destroying Azure Resource Group $(SERVICE_NAME)-rg..."
	az group delete --name $(SERVICE_NAME)-rg --yes --no-wait

# Check status of the Azure VM
status:
	@echo "🔍 Checking status of Azure VM $(SERVICE_NAME)-vm..."
	az vm show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-vm --show-details --query "{Name:name, ProvisioningState:provisioningState, PowerState:powerState, PublicIps:publicIps}" -o json

# Get the public IP of the deployed Azure VM
endpoint:
	@az vm show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-vm --show-details --query "publicIps" -o tsv

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model on the Azure VM
query:
	@echo "🔍 Querying $(SERVICE_NAME)-vm with prompt: \"$(PROMPT)\"..."
	@IP=$$(az vm show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-vm --show-details --query "publicIps" -o tsv); \
	curl -s -X POST "http://$$IP:8080/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test clean deploy destroy status endpoint query
