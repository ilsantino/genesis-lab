"""
Unit tests for synthetic data generators.

Uses mocked BedrockClient to test generator logic without AWS calls.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from src.generation import CustomerServiceGenerator, TimeSeriesGenerator
from src.generation.schemas import CustomerServiceConversation, TimeSeries
from src.generation.templates.customer_service_prompts import ALL_INTENTS
from src.generation.templates.timeseries_prompts import ALL_SERIES_TYPES


# =============================================================================
# MOCK RESPONSES
# =============================================================================

MOCK_CONVERSATION_RESPONSE = json.dumps({
    "conversation_id": "conv_test123456",
    "intent": "card_arrival",
    "category": "cards",
    "sentiment": "neutral",
    "complexity": "simple",
    "language": "en",
    "turn_count": 4,
    "customer_emotion_arc": "stable_neutral",
    "resolution_time_category": "quick",
    "resolution_status": "resolved",
    "turns": [
        {"speaker": "customer", "text": "Hello, I ordered a new card last week. When will it arrive?"},
        {"speaker": "agent", "text": "Hi there! I'd be happy to help you track your card. Let me check the status for you."},
        {"speaker": "customer", "text": "Thank you, I appreciate that."},
        {"speaker": "agent", "text": "Your card was shipped yesterday and should arrive within 3-5 business days. Is there anything else I can help with?"}
    ]
})

MOCK_CONVERSATION_SPECIFIC_INTENT = json.dumps({
    "conversation_id": "conv_refund12345",
    "intent": "refund_not_showing_up",
    "category": "refunds",
    "sentiment": "negative",
    "complexity": "medium",
    "language": "en",
    "turn_count": 6,
    "customer_emotion_arc": "frustrated_to_satisfied",
    "resolution_time_category": "normal",
    "resolution_status": "resolved",
    "turns": [
        {"speaker": "customer", "text": "I was promised a refund 10 days ago and it still hasn't shown up in my account. This is unacceptable!"},
        {"speaker": "agent", "text": "I sincerely apologize for the delay with your refund. Let me look into this immediately."},
        {"speaker": "customer", "text": "Please do. I've been waiting way too long for this."},
        {"speaker": "agent", "text": "I found your refund request. It was processed but there was a system delay. I'm expediting it now."},
        {"speaker": "customer", "text": "When will I actually see the money?"},
        {"speaker": "agent", "text": "The refund will appear in your account within 24-48 hours. I've also added a $10 credit for the inconvenience."}
    ]
})

MOCK_TIMESERIES_RESPONSE = json.dumps({
    "series_id": "ts_test123456",
    "domain": "electricity",
    "series_type": "residential_consumption",
    "frequency": "1H",
    "complexity": "medium",
    "data_quality": "clean",
    "language": "en",
    "length": 24,
    "seasonality_types": ["daily"],
    "trend_type": "none",
    "anomaly_types": [],
    "anomaly_indices": [],
    "domain_context": "residential",
    "start": "2024-01-01T00:00:00Z",
    "target": [
        0.2, 0.1, -0.1, -0.2, -0.15, 0.3,
        0.8, 1.2, 1.0, 0.7, 0.5, 0.6,
        0.8, 0.9, 0.7, 0.5, 0.9, 1.5,
        1.8, 1.6, 1.2, 0.8, 0.5, 0.3
    ],
    "metadata": {
        "unit": "kWh",
        "description": "Residential electricity consumption"
    }
})

MOCK_TIMESERIES_SPECIFIC_TYPE = json.dumps({
    "series_id": "ts_sensor12345",
    "domain": "sensors",
    "series_type": "temperature",
    "frequency": "1H",
    "complexity": "simple",
    "data_quality": "clean",
    "language": "en",
    "length": 24,
    "seasonality_types": ["daily"],
    "trend_type": "none",
    "anomaly_types": [],
    "anomaly_indices": [],
    "domain_context": "industrial",
    "start": "2024-01-01T00:00:00Z",
    "target": [
        20.5, 20.3, 20.1, 19.8, 19.5, 19.8,
        20.2, 21.0, 22.0, 23.0, 23.5, 24.0,
        24.2, 24.5, 24.3, 24.0, 23.5, 23.0,
        22.5, 22.0, 21.5, 21.0, 20.8, 20.5
    ],
    "metadata": {
        "unit": "Celsius",
        "description": "Temperature sensor reading"
    }
})


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def mock_bedrock_client():
    """Create a mocked BedrockClient."""
    mock_client = MagicMock()
    mock_client.invoke_model = MagicMock(return_value=MOCK_CONVERSATION_RESPONSE)
    return mock_client


@pytest.fixture
def mock_config():
    """Mock the configuration."""
    mock_cfg = MagicMock()
    mock_cfg.aws.default_model = "claude_35_sonnet"
    
    # Mock generation params
    mock_params = MagicMock()
    mock_params.temperature = 0.7
    mock_params.max_tokens = 2000
    mock_params.top_p = 0.9
    
    # Mock domain config
    mock_domain_cfg = MagicMock()
    mock_domain_cfg.generation_params = mock_params
    
    mock_cfg.get_domain_config = MagicMock(return_value=mock_domain_cfg)
    
    return mock_cfg


# =============================================================================
# CUSTOMER SERVICE GENERATOR TESTS
# =============================================================================

class TestCustomerServiceGenerator:
    """Tests for CustomerServiceGenerator."""
    
    @patch('src.generation.generator.get_config')
    def test_generate_single_returns_valid_structure(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generate_single returns a valid conversation structure."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_RESPONSE
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        result = generator.generate_single()
        
        # Verify structure
        assert "conversation_id" in result
        assert "intent" in result
        assert "sentiment" in result
        assert "turns" in result
        assert "resolution_status" in result
        
        # Verify turns structure
        assert isinstance(result["turns"], list)
        assert len(result["turns"]) >= 2
        
        for turn in result["turns"]:
            assert "speaker" in turn
            assert "text" in turn
            assert turn["speaker"] in ["customer", "agent"]
        
        # First turn should be from customer
        assert result["turns"][0]["speaker"] == "customer"
        
        # Verify invoke_model was called
        mock_bedrock_client.invoke_model.assert_called_once()
    
    @patch('src.generation.generator.get_config')
    def test_generate_single_with_specific_intent(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test generation with a specific intent."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_SPECIFIC_INTENT
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        result = generator.generate_single(intent="refund_not_showing_up")
        
        # Verify intent is set correctly
        assert result["intent"] == "refund_not_showing_up"
        assert result["category"] == "refunds"
        
        # Verify conversation is about the refund issue
        assert "refund" in result["turns"][0]["text"].lower()
    
    @patch('src.generation.generator.get_config')
    def test_generate_batch_returns_list(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generate_batch returns a list of conversations."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_RESPONSE
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        results = generator.generate_batch(count=3, continue_on_error=True)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for conv in results:
            assert "conversation_id" in conv
            assert "turns" in conv
        
        # Verify invoke_model was called 3 times
        assert mock_bedrock_client.invoke_model.call_count == 3
    
    @patch('src.generation.generator.get_config')
    def test_invalid_intent_handled_gracefully(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that invalid intent is handled gracefully."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_RESPONSE
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        # Generator should still work with unknown intent (it passes to LLM)
        # but the response may be fixed by _fix_conversation_schema
        result = generator.generate_single(intent="unknown_intent_xyz")
        
        # Should still return a valid structure
        assert "conversation_id" in result
        assert "turns" in result
    
    @patch('src.generation.generator.get_config')
    def test_generator_metrics(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generator tracks metrics correctly."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_RESPONSE
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        # Generate a few items
        generator.generate_single()
        generator.generate_single()
        
        metrics = generator.get_metrics()
        
        assert metrics["total_generated"] == 2
        assert metrics["total_failed"] == 0
        assert metrics["success_rate"] == 1.0
    
    def test_all_intents_are_valid(self):
        """Test that ALL_INTENTS contains expected Banking77 intents."""
        # Should have 77 intents
        assert len(ALL_INTENTS) == 77
        
        # Check a few known intents
        assert "card_arrival" in ALL_INTENTS
        assert "activate_my_card" in ALL_INTENTS
        assert "getting_spare_card" in ALL_INTENTS


# =============================================================================
# TIME SERIES GENERATOR TESTS
# =============================================================================

class TestTimeSeriesGenerator:
    """Tests for TimeSeriesGenerator."""
    
    @patch('src.generation.timeseries_generator.get_config')
    def test_generate_single_returns_valid_structure(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generate_single returns a valid time series structure."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_TIMESERIES_RESPONSE
        
        generator = TimeSeriesGenerator(
            client=mock_bedrock_client,
            domain="time_series"
        )
        
        result = generator.generate_single()
        
        # Verify structure
        assert "series_id" in result
        assert "domain" in result
        assert "series_type" in result
        assert "frequency" in result
        assert "target" in result
        
        # Verify target is a list of numbers
        assert isinstance(result["target"], list)
        assert len(result["target"]) > 0
        assert all(isinstance(v, (int, float)) for v in result["target"])
        
        # Verify invoke_model was called
        mock_bedrock_client.invoke_model.assert_called_once()
    
    @patch('src.generation.timeseries_generator.get_config')
    def test_generate_single_with_specific_type(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test generation with a specific series type."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_TIMESERIES_SPECIFIC_TYPE
        
        generator = TimeSeriesGenerator(
            client=mock_bedrock_client,
            domain="time_series"
        )
        
        result = generator.generate_single(series_type="temperature")
        
        # Verify series type is set correctly
        assert result["series_type"] == "temperature"
        assert result["domain"] == "sensors"
        
        # Verify data points
        assert len(result["target"]) == 24
    
    @patch('src.generation.timeseries_generator.get_config')
    def test_generate_batch_returns_list(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generate_batch returns a list of time series."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_TIMESERIES_RESPONSE
        
        generator = TimeSeriesGenerator(
            client=mock_bedrock_client,
            domain="time_series"
        )
        
        results = generator.generate_batch(count=3, continue_on_error=True)
        
        assert isinstance(results, list)
        assert len(results) == 3
        
        for ts in results:
            assert "series_id" in ts
            assert "target" in ts
        
        # Verify invoke_model was called 3 times
        assert mock_bedrock_client.invoke_model.call_count == 3
    
    @patch('src.generation.timeseries_generator.get_config')
    def test_generator_properties(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test generator properties."""
        mock_get_config.return_value = mock_config
        
        generator = TimeSeriesGenerator(
            client=mock_bedrock_client,
            domain="time_series"
        )
        
        # Check available series types
        assert generator.series_type_count == 16
        assert len(generator.available_series_types) == 16
        
        # Check available domains
        domains = generator.available_domains
        assert "electricity" in domains
        assert "sensors" in domains
    
    @patch('src.generation.timeseries_generator.get_config')
    def test_get_series_types_for_domain(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test getting series types for a specific domain."""
        mock_get_config.return_value = mock_config
        
        generator = TimeSeriesGenerator(
            client=mock_bedrock_client,
            domain="time_series"
        )
        
        electricity_types = generator.get_series_types_for_domain("electricity")
        assert len(electricity_types) > 0
        assert "residential_consumption" in electricity_types
        
        sensor_types = generator.get_series_types_for_domain("sensors")
        assert len(sensor_types) > 0
    
    def test_all_series_types_defined(self):
        """Test that ALL_SERIES_TYPES contains expected types."""
        # Should have 16 series types
        assert len(ALL_SERIES_TYPES) == 16
        
        # Check a few known types from different domains
        assert "residential_consumption" in ALL_SERIES_TYPES  # electricity
        assert "temperature" in ALL_SERIES_TYPES  # sensors
        assert "solar_generation" in ALL_SERIES_TYPES  # energy
        assert "stock_price" in ALL_SERIES_TYPES  # financial


# =============================================================================
# JSON PARSING TESTS
# =============================================================================

class TestJSONParsing:
    """Tests for JSON response parsing."""
    
    @patch('src.generation.generator.get_config')
    def test_parse_json_in_markdown_block(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test parsing JSON wrapped in markdown code block."""
        mock_get_config.return_value = mock_config
        
        # Response with markdown wrapper
        markdown_response = f"""Here's the conversation:

```json
{MOCK_CONVERSATION_RESPONSE}
```

Hope this helps!"""
        
        mock_bedrock_client.invoke_model.return_value = markdown_response
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        result = generator.generate_single()
        
        # Should still parse correctly
        assert "conversation_id" in result
        assert "turns" in result
    
    @patch('src.generation.generator.get_config')
    def test_parse_raw_json(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test parsing raw JSON without markdown."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.return_value = MOCK_CONVERSATION_RESPONSE
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        result = generator.generate_single()
        
        assert "conversation_id" in result


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in generators."""
    
    @patch('src.generation.generator.get_config')
    def test_generation_failure_raises_runtime_error(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that generation failure raises RuntimeError."""
        mock_get_config.return_value = mock_config
        mock_bedrock_client.invoke_model.side_effect = Exception("AWS error")
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        with pytest.raises(RuntimeError, match="Failed to generate conversation"):
            generator.generate_single()
    
    @patch('src.generation.generator.get_config')
    def test_batch_continues_on_error(self, mock_get_config, mock_bedrock_client, mock_config):
        """Test that batch generation continues on error when flag is set."""
        mock_get_config.return_value = mock_config
        
        # First call succeeds, second fails, third succeeds
        mock_bedrock_client.invoke_model.side_effect = [
            MOCK_CONVERSATION_RESPONSE,
            Exception("Temporary error"),
            MOCK_CONVERSATION_RESPONSE
        ]
        
        generator = CustomerServiceGenerator(
            client=mock_bedrock_client,
            domain="customer_service"
        )
        
        results = generator.generate_batch(count=3, continue_on_error=True)
        
        # Should have 2 successful results
        assert len(results) == 2
        
        # Metrics should reflect failure
        metrics = generator.get_metrics()
        assert metrics["total_generated"] == 2
        assert metrics["total_failed"] == 1

