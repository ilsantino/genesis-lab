"""
Unit tests for AWS Bedrock client.
Uses mocks - no real AWS calls.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import json

from botocore.exceptions import ClientError

from src.utils.aws_client import BedrockClient, RateLimiter, RETRYABLE_ERRORS
from src.utils.config import get_config


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = get_config()
    return config


@pytest.fixture
def mock_boto3_client():
    """Create mock boto3 client."""
    with patch("src.utils.aws_client.boto3.client") as mock:
        yield mock


@pytest.fixture
def mock_bedrock_response():
    """Standard successful Bedrock response."""
    response_body = {
        "content": [{"text": "Hello from GENESIS-LAB"}]
    }
    
    mock_response = {
        "body": MagicMock()
    }
    mock_response["body"].read.return_value = json.dumps(response_body)
    
    return mock_response


# ============================================================================
# RATE LIMITER TESTS
# ============================================================================

class TestRateLimiter:
    """Tests for RateLimiter class."""
    
    def test_allows_requests_under_limit(self):
        """Should allow requests when under limit."""
        limiter = RateLimiter(max_requests_per_minute=10)
        
        # Should not block for first 10 requests
        for _ in range(10):
            limiter.acquire()  # Should not raise or block significantly
        
        assert len(limiter.requests) == 10
    
    def test_clears_old_requests(self):
        """Should clear requests older than 1 minute."""
        limiter = RateLimiter(max_requests_per_minute=5)
        
        # Add old requests manually
        old_time = datetime.now() - timedelta(minutes=2)
        limiter.requests.extend([old_time] * 5)
        
        # This should clear old requests and allow new one
        limiter.acquire()
        
        # Should only have the new request
        assert len(limiter.requests) == 1


# ============================================================================
# BEDROCK CLIENT TESTS
# ============================================================================

class TestBedrockClientInit:
    """Tests for BedrockClient initialization."""
    
    def test_initialization(self, mock_config, mock_boto3_client):
        """Should initialize with config."""
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        assert client is not None
        assert client.total_requests == 0
        assert client.total_retries == 0
        assert client.total_errors == 0
    
    def test_from_config_factory(self, mock_boto3_client):
        """Should create client using factory method."""
        client = BedrockClient.from_config()
        
        assert client is not None
        assert isinstance(client, BedrockClient)


class TestBedrockClientInvoke:
    """Tests for model invocation."""
    
    def test_successful_invoke(self, mock_config, mock_boto3_client, mock_bedrock_response):
        """Should return generated text on success."""
        # Setup mock
        mock_boto3_client.return_value.invoke_model.return_value = mock_bedrock_response
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        result = client.invoke_model(prompt="Test prompt")
        
        assert result == "Hello from GENESIS-LAB"
        assert client.total_requests == 1
        assert client.total_errors == 0
    
    def test_with_system_prompt(self, mock_config, mock_boto3_client, mock_bedrock_response):
        """Should include system prompt in request."""
        mock_boto3_client.return_value.invoke_model.return_value = mock_bedrock_response
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        result = client.invoke_model(
            prompt="Test prompt",
            system_prompt="You are a helpful assistant."
        )
        
        assert result == "Hello from GENESIS-LAB"
        
        # Verify system prompt was included in call
        call_args = mock_boto3_client.return_value.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["system"] == "You are a helpful assistant."


class TestBedrockClientRetry:
    """Tests for retry logic."""
    
    def test_retries_on_throttling(self, mock_config, mock_boto3_client, mock_bedrock_response):
        """Should retry on ThrottlingException."""
        # First call raises throttling, second succeeds
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel"
        )
        
        mock_boto3_client.return_value.invoke_model.side_effect = [
            throttle_error,
            mock_bedrock_response
        ]
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        # Patch sleep to avoid waiting
        with patch("src.utils.aws_client.time.sleep"):
            result = client.invoke_model(prompt="Test")
        
        assert result == "Hello from GENESIS-LAB"
        assert client.total_retries == 1
        assert client.total_requests == 2
    
    def test_raises_after_max_retries(self, mock_config, mock_boto3_client):
        """Should raise after exhausting retries."""
        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel"
        )
        
        # Always fail
        mock_boto3_client.return_value.invoke_model.side_effect = throttle_error
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        with patch("src.utils.aws_client.time.sleep"):
            with pytest.raises(RuntimeError, match="ThrottlingException"):
                client.invoke_model(prompt="Test")
        
        assert client.total_errors == mock_config.generation.retry_attempts


class TestBedrockClientBatch:
    """Tests for batch invocation."""
    
    def test_batch_invoke(self, mock_config, mock_boto3_client, mock_bedrock_response):
        """Should process multiple prompts."""
        mock_boto3_client.return_value.invoke_model.return_value = mock_bedrock_response
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = client.invoke_batch(prompts=prompts)
        
        assert len(results) == 3
        assert all(r == "Hello from GENESIS-LAB" for r in results)
        assert client.total_requests == 3


class TestBedrockClientMetrics:
    """Tests for metrics tracking."""
    
    def test_get_metrics(self, mock_config, mock_boto3_client, mock_bedrock_response):
        """Should return accurate metrics."""
        mock_boto3_client.return_value.invoke_model.return_value = mock_bedrock_response
        
        client = BedrockClient(
            aws_config=mock_config.aws,
            generation_config=mock_config.generation
        )
        
        client.invoke_model(prompt="Test 1")
        client.invoke_model(prompt="Test 2")
        
        metrics = client.get_metrics()
        
        assert metrics["total_requests"] == 2
        assert metrics["total_retries"] == 0
        assert metrics["total_errors"] == 0