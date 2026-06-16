import json
import os
import unittest
from unittest.mock import MagicMock, patch

# Clean up AWS environment variables
if "AWS_ACCESS_KEY_ID" in os.environ:
    del os.environ["AWS_ACCESS_KEY_ID"]
if "AWS_SECRET_ACCESS_KEY" in os.environ:
    del os.environ["AWS_SECRET_ACCESS_KEY"]

from server import mcp


class TestDevOpsAgent(unittest.IsolatedAsyncioTestCase):
    async def test_tools_registered(self):
        """Verify that the expected tools are registered with FastMCP."""
        tools = [t.name for t in await mcp.list_tools()]
        self.assertIn("analyze_cloud_logging", tools)
        self.assertIn("suggest_sre_remediation", tools)
        self.assertIn("get_vllm_deployment_config", tools)
        self.assertIn("get_vertex_ai_model_copy_instructions", tools)
        self.assertIn("get_huggingface_model_copy_instructions", tools)
        self.assertIn("get_huggingfacehub_download_path", tools)
        self.assertIn("save_hf_token", tools)
        self.assertIn("list_vertex_models", tools)
        self.assertIn("list_bucket_models", tools)
        self.assertIn("deploy_vllm", tools)
        self.assertIn("destroy_vllm", tools)
        self.assertIn("status_vllm", tools)
        self.assertIn("update_vllm_scaling", tools)
        self.assertIn("check_gpu_quotas", tools)
        self.assertIn("verify_model_health", tools)
        self.assertIn("query_gemma4", tools)
        self.assertIn("query_gemma4_with_stats", tools)
        self.assertIn("get_model_details", tools)
        self.assertIn("get_help", tools)

    @patch("subprocess.run")
    def test_update_vllm_scaling(self, mock_run):
        """Test the update_vllm_scaling tool with mock Azure CLI VM show and resize."""
        from server import update_vllm_scaling

        # Mock VM show
        mock_show_res = MagicMock()
        mock_show_res.returncode = 0
        mock_show_res.stdout = json.dumps({"Size": "Standard_NV36ads_A10_v5"})

        # Mock VM resize
        mock_resize_res = MagicMock()
        mock_resize_res.returncode = 0

        mock_run.side_effect = [mock_show_res, mock_resize_res]

        result = update_vllm_scaling(instance_type="Standard_NV72ads_A10_v5", service_name="test-service")

        # Verify calls
        self.assertEqual(mock_run.call_count, 2)
        args_show, kwargs_show = mock_run.call_args_list[0]
        self.assertIn("show", args_show[0])
        args_resize, kwargs_resize = mock_run.call_args_list[1]
        self.assertIn("resize", args_resize[0])
        self.assertIn("Standard_NV72ads_A10_v5", args_resize[0])

        self.assertIn(
            "Successfully requested scale-up of Azure VM `test-service-vm` from `Standard_NV36ads_A10_v5` to `Standard_NV72ads_A10_v5`",
            result,
        )

    @patch("subprocess.run")
    @patch("server.get_secret")
    async def test_deploy_vllm(self, mock_get_secret, mock_run):
        """Test the deploy_vllm tool with mock Azure CLI commands."""
        from server import deploy_vllm

        mock_get_secret.return_value = "mock-hf-token"

        # Mock response for resource group creation, VM creation, open-port, and IP retrieval
        mock_rg_res = MagicMock()
        mock_rg_res.returncode = 0

        mock_vm_res = MagicMock()
        mock_vm_res.returncode = 0

        mock_nsg_res = MagicMock()
        mock_nsg_res.returncode = 0

        mock_ip_res = MagicMock()
        mock_ip_res.returncode = 0
        mock_ip_res.stdout = "13.82.4.5\n"

        mock_run.side_effect = [mock_rg_res, mock_vm_res, mock_nsg_res, mock_ip_res]

        result = await deploy_vllm(
            service_name="test-service",
            model_path="google/gemma-4-12B-it-qat-w4a16-ct",
        )

        self.assertIn(
            "Successfully requested Azure VM Standard_NV36ads_A10_v5 Deployment for service 'test-service'", result
        )
        self.assertIn("Public IP: `13.82.4.5`", result)
        self.assertEqual(mock_run.call_count, 4)

        # Inspect the VM create call
        vm_create_args = mock_run.call_args_list[1][0][0]
        self.assertIn("create", vm_create_args)
        self.assertIn("Standard_NV36ads_A10_v5", vm_create_args)
        self.assertIn("microsoftazurelinux:azurelinux-4:4:latest", vm_create_args)

    @patch("subprocess.run")
    async def test_destroy_vllm(self, mock_run):
        """Test the destroy_vllm tool with mock Azure CLI run-command."""
        from server import destroy_vllm

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_run.return_value = mock_res

        result = await destroy_vllm(service_name="test-service")

        self.assertIn(
            "Successfully requested cleanup of the 'vllm-server' Docker container on Azure VM 'test-service-vm'", result
        )
        mock_run.assert_called_once()
        cmd_args = mock_run.call_args[0][0]
        self.assertIn("run-command", cmd_args)
        self.assertIn("invoke", cmd_args)
        self.assertIn("docker stop vllm-server || true", cmd_args)

    @patch("subprocess.run")
    def test_status_vllm(self, mock_run):
        """Test status_vllm tool with mock Azure CLI VM show details."""
        from server import status_vllm

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = json.dumps(
            {"Name": "test-service-vm", "Size": "Standard_NV36ads_A10_v5", "State": "VM running", "IP": "13.82.4.5"}
        )
        mock_run.return_value = mock_res

        result = status_vllm(service_name="test-service")
        self.assertIn("Azure VM Status for service prefix 'test-service'", result)
        self.assertIn("test-service-vm", result)
        self.assertIn("VM running", result)
        self.assertIn("Standard_NV36ads_A10_v5", result)

    async def test_resources_registered(self):
        """Verify that the expected resources are registered with FastMCP."""
        resources = [str(r.uri) for r in await mcp.list_resources()]
        self.assertIn("config://vllm-deployment-template", resources)

    def test_get_huggingface_model_copy_instructions(self):
        """Test the output of the Hugging Face model copy instructions tool."""
        from server import get_huggingface_model_copy_instructions

        instructions = get_huggingface_model_copy_instructions("test/slug", "test-bucket")
        self.assertIn("test/slug", instructions)
        self.assertIn("test-bucket", instructions)
        self.assertIn("slug", instructions)
        self.assertIn("snapshot_download('test/slug')", instructions)

    def test_get_vertex_ai_model_copy_instructions(self):
        """Test the output of the Vertex AI model copy instructions tool."""
        from server import get_vertex_ai_model_copy_instructions

        instructions = get_vertex_ai_model_copy_instructions("gemma-4-12B-it-qat-w4a16-ct")
        self.assertIn("gemma-4-12B-it-qat-w4a16-ct", instructions)
        self.assertIn("Vertex AI Model Garden", instructions)
        self.assertIn("gcloud storage cp", instructions)

    @patch("subprocess.run")
    def test_list_bucket_models_mock(self, mock_run):
        """Test list_bucket_models lists Azure Storage Blobs."""
        from server import list_bucket_models

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = json.dumps([{"name": "gemma-4-12B-it-qat-w4a16-ct/config.json", "size": 5242880}])
        mock_run.return_value = mock_res

        result = list_bucket_models("vllmmodelsstore")
        self.assertIn("vllmmodelsstore", result)
        self.assertIn("gemma-4-12B-it-qat-w4a16-ct/config.json", result)
        self.assertIn("5.00 MB", result)

    @patch("subprocess.run")
    @patch("server.secretmanager.SecretManagerServiceClient")
    async def test_save_hf_token(self, mock_gcp_client_class, mock_run):
        """Test save_hf_token tool saves token to Azure Key Vault and GCP Secret Manager."""
        from server import save_hf_token

        mock_azure_res = MagicMock()
        mock_azure_res.returncode = 0
        mock_run.return_value = mock_azure_res

        mock_gcp_client = MagicMock()
        mock_gcp_client_class.return_value = mock_gcp_client
        mock_gcp_client.add_secret_version.return_value = MagicMock(name="projects/test/secrets/hf-token/versions/1")

        result = await save_hf_token("test-token")
        self.assertIn("Token saved", result)

    @patch("subprocess.run")
    def test_check_gpu_quotas(self, mock_run):
        """Test check_gpu_quotas tool formats Azure metrics correctly."""
        from server import check_gpu_quotas

        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = json.dumps(
            [{"localName": "Standard NVads A10 v5 Family Cores", "limit": 36, "currentValue": 0}]
        )
        mock_run.return_value = mock_res

        result = check_gpu_quotas(region="eastus")
        self.assertIn("Azure VM GPU/Core Quotas for region `eastus`", result)
        self.assertIn("Standard NVads A10 v5 Family Cores", result)
        self.assertIn("Limit: `36`", result)

    async def test_get_help(self):
        """Test get_help returns correct tool and configuration information."""
        from server import get_help

        result = await get_help()
        self.assertIn("Azure/GCP Gemma 4 SRE Agent Help", result)
        self.assertIn("deploy_vllm", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_verify_model_health(self, mock_model_name, mock_client_factory):
        """Test verify_model_health parses model response and calculates latency."""
        from server import verify_model_health

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Yes, the model is active and running."
        mock_choice.message = mock_message
        mock_choice.message.content = "Yes, the model is active and running."
        mock_completion.choices = [mock_choice]

        # Async mock for client.chat.completions.create
        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await verify_model_health()
        self.assertIn("Model health check PASSED", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Yes, the model is active and running.", result)

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4(self, mock_model_name, mock_client_factory):
        """Test query_gemma4 queries the model via chat completions."""
        from server import query_gemma4

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()
        mock_completion = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()

        mock_message.content = "Response from Gemma"
        mock_choice.message = mock_message
        mock_choice.message.content = "Response from Gemma"
        mock_completion.choices = [mock_choice]

        async def mock_create(*args, **kwargs):
            return mock_completion

        mock_chat.create = mock_create
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await query_gemma4("Hello")
        self.assertEqual(result, "Response from Gemma")

    @patch("server.get_vllm_client")
    @patch("server.get_active_model_name")
    async def test_query_gemma4_with_stats(self, mock_model_name, mock_client_factory):
        """Test query_gemma4_with_stats collects performance metrics."""
        from server import query_gemma4_with_stats

        mock_model_name.return_value = "test-model-name"
        mock_client = MagicMock()
        mock_chat = MagicMock()

        # We need mock chunks to simulate streaming
        class MockChunk:
            def __init__(self, content):
                mock_delta = MagicMock()
                mock_delta.content = content
                mock_choice = MagicMock()
                mock_choice.delta = mock_delta
                self.choices = [mock_choice]

        chunks = [MockChunk("Hello"), MockChunk(" world!")]

        # Async generator mock
        async def mock_create_stream(*args, **kwargs):
            async def async_gen():
                for chunk in chunks:
                    yield chunk

            return async_gen()

        mock_chat.create = mock_create_stream
        mock_client.chat = MagicMock()
        mock_client.chat.completions = mock_chat
        mock_client_factory.return_value = mock_client

        result = await query_gemma4_with_stats("Hello")
        self.assertIn("Performance Stats", result)
        self.assertIn("test-model-name", result)
        self.assertIn("Hello world!", result)

    @patch("server.get_vllm_client")
    @patch("server.get_vllm_url")
    @patch("server.get_auth_token")
    @patch("server.httpx.AsyncClient")
    async def test_get_model_details(
        self, mock_httpx_client_class, mock_auth_token, mock_vllm_url, mock_client_factory
    ):
        """Test get_model_details formats models list and health status."""
        from server import get_model_details

        mock_vllm_url.return_value = "http://test-url"
        mock_auth_token.return_value = "mock-token"

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_models_response = MagicMock()
        mock_model = MagicMock()
        mock_model.id = "test-model-id"
        mock_model.object = "model"
        mock_model.owned_by = "google"
        mock_models_response.data = [mock_model]

        async def mock_list():
            return mock_models_response

        mock_client.models.list = mock_list
        mock_client_factory.return_value = mock_client

        # Mock HTTPX response
        mock_httpx_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200

        async def mock_get(*args, **kwargs):
            return mock_response

        mock_httpx_client.get = mock_get
        mock_httpx_client.__aenter__.return_value = mock_httpx_client
        mock_httpx_client_class.return_value = mock_httpx_client

        result = await get_model_details()
        self.assertIn("Model Details (http://test-url)", result)
        self.assertIn("test-model-id", result)
        self.assertIn("Healthy", result)

    def test_get_vllm_deployment_config_spot(self):
        """Test get_vllm_deployment_config outputs Azure Linux 4.0 configuration."""
        from server import get_vllm_deployment_config

        result = get_vllm_deployment_config(
            service_name="test-service", model_path="google/gemma-4-12B-it-qat-w4a16-ct"
        )
        self.assertIn("Azure Linux 4.0 Deployment Config", result)
        self.assertIn("microsoftazurelinux:azurelinux-4:4:latest", result)
        self.assertIn("Standard_NV36ads_A10_v5", result)


if __name__ == "__main__":
    unittest.main()
