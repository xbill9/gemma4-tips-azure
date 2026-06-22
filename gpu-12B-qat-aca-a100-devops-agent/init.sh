#!/bin/bash
# Azure-only setup initialization script

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Logging in to Azure..."
az login
