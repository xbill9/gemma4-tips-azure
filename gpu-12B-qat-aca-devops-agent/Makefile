# Configuration - Update these or override via environment variables
AZURE_LOCATION ?= eastus
AZURE_KEYVAULT_NAME ?= vllm-devops-kv
SERVICE_NAME ?= gpu-12b-qat-l4-devops-agent
MODEL_PATH ?= google/gemma-4-12B-it-qat-w4a16-ct

install:
	pip install -r requirements.txt

run:
	python3 server.py

test:
	python3 test_agent.py

lint:
	ruff check .
	ruff format --check .
	mypy .

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +

# Deploy the vLLM inference stack to Azure Container Apps (ACA) with GPU
deploy:
	@echo "🚀 Creating Azure Resource Group $(SERVICE_NAME)-rg in $(AZURE_LOCATION)..."
	az group create --name $(SERVICE_NAME)-rg --location $(AZURE_LOCATION)
	@echo "🚀 Creating Container Apps Environment $(SERVICE_NAME)-env..."
	az containerapp env create \
		--name $(SERVICE_NAME)-env \
		--resource-group $(SERVICE_NAME)-rg \
		--location $(AZURE_LOCATION)
	@echo "🚀 Adding GPU Workload Profile (Consumption-GPU-NC8as-T4) to Environment..."
	az containerapp env workload-profile add \
		--name $(SERVICE_NAME)-env \
		--resource-group $(SERVICE_NAME)-rg \
		--workload-profile-name gpu-profile \
		--workload-profile-type Consumption-GPU-NC8as-T4
	@echo "🚀 Deploying vLLM to Azure Container Apps..."
	az containerapp create \
		--name $(SERVICE_NAME)-app \
		--resource-group $(SERVICE_NAME)-rg \
		--environment $(SERVICE_NAME)-env \
		--workload-profile-name gpu-profile \
		--image vllm/vllm-openai:nightly \
		--cpu 8.0 \
		--memory 56.0Gi \
		--ingress external \
		--target-port 8080 \
		--env-vars HF_TOKEN="$$(az keyvault secret show --name hf-token --vault-name $(AZURE_KEYVAULT_NAME) --query value -o tsv 2>/dev/null || echo '')" \
		--args "--model $(MODEL_PATH) --quantization compressed-tensors --dtype bfloat16 --max-model-len 32768 --disable-chunked-mm-input --gpu-memory-utilization 0.95 --kv-cache-dtype fp8 --tensor-parallel-size 1 --max-num-seqs 8 --enable-chunked-prefill --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4 --reasoning-parser gemma4 --async-scheduling --limit-mm-per-prompt '{}' --host 0.0.0.0 --port 8080"

# Destroy the vLLM inference stack on Azure
destroy:
	@echo "🗑️ Destroying Azure Resource Group $(SERVICE_NAME)-rg..."
	az group delete --name $(SERVICE_NAME)-rg --yes --no-wait

# Check status of the Container App
status:
	@echo "🔍 Checking status of Azure Container App $(SERVICE_NAME)-app..."
	az containerapp show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-app -o json

# Get the endpoint FQDN of the deployed Container App
endpoint:
	@az containerapp show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-app --query properties.configuration.ingress.fqdn -o tsv

PROMPT ?= "What is Site Reliability Engineering?"

# Query the vLLM model on the Container App
query:
	@echo "🔍 Querying $(SERVICE_NAME)-app with prompt: \"$(PROMPT)\"..."
	@FQDN=$$(az containerapp show --resource-group $(SERVICE_NAME)-rg --name $(SERVICE_NAME)-app --query properties.configuration.ingress.fqdn -o tsv); \
	curl -s -X POST "https://$$FQDN/v1/completions" \
		-H "Content-Type: application/json" \
		-d '{"model": "$(MODEL_PATH)", "prompt": "$(PROMPT)", "max_tokens": 128, "temperature": 0.2}' | python3 -m json.tool

.PHONY: install run test clean deploy destroy status endpoint query
