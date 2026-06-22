#!/bin/bash

# Environment configuration for GPU Azure VM (vLLM Serving)

cat <<EOF > .env
MODEL_NAME=google/gemma-4-12B-it-qat-w4a16-ct
AZURE_LOCATION=eastus
AZURE_KEYVAULT_NAME=vllm-devops-kv
AZURE_STORAGE_ACCOUNT=vllmmodelsstore
VM_SIZE=Standard_NV36ads_A10_v5
EOF

echo "Sourcing Env"
source .env

echo "Current Environment:"
cat .env
