"""
Integration tests for AWS Bedrock client.
These tests make REAL API calls - run manually or with: pytest -m integration

Usage:
    pytest tests/test_bedrock_integration.py -v
    pytest -m integration  # Run all integration tests
"""

import pytest
from src.utils.aws_client import BedrockClient
from src.utils.config import get_config


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def real_client():
    """Create real BedrockClient (requires AWS credentials)."""
    config = get_config()
    
    if not config.aws.validate_credentials_present():
        pytest.skip("AWS credentials not configured")
    
    return BedrockClient.from_config()


class TestBedrockIntegration:
    """Integration tests with real AWS calls."""
    
    def test_simple_generation(self, real_client):
        """Test real text generation."""
        result = real_client.invoke_model(
            prompt="Say exactly: 'GENESIS-LAB OK'",
            temperature=0.0,
            max_tokens=20
        )
        
        assert result is not None
        assert len(result) > 0
        print(f"✅ Generated: {result}")
    
    def test_generation_with_system_prompt(self, real_client):
        """Test generation with system prompt."""
        result = real_client.invoke_model(
            prompt="What is 2+2?",
            system_prompt="You are a math tutor. Answer briefly.",
            temperature=0.0,
            max_tokens=50
        )
        
        assert "4" in result
        print(f"✅ Math result: {result}")


if __name__ == "__main__":
    print("Running Bedrock Integration Tests...\n")
    pytest.main([__file__, "-v"])