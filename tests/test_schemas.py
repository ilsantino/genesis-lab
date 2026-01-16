"""
Tests for schemas.py - validates Pydantic v1/v2 compatibility.
"""

import pytest
from typing import Dict, Any
from datetime import datetime, timezone

from src.generation.schemas import (
    ConversationTurn,
    CustomerServiceConversation,
    QualityMetrics,
    BiasMetrics,
    DatasetMetadata,
    SchemaConstants,
    PYDANTIC_V2,
    get_schema_info,
)


class TestConversationTurn:
    def test_valid_turn(self):
        turn = ConversationTurn(
            speaker="customer",
            text="Hello, I need help with my account please."
        )
        assert turn.speaker == "customer"
        assert "help" in turn.text

    def test_text_stripped(self):
        turn = ConversationTurn(
            speaker="agent",
            text="   Hello, how can I help?   "
        )
        assert turn.text == "Hello, how can I help?"

    def test_text_too_short(self):
        with pytest.raises(ValueError):
            ConversationTurn(speaker="customer", text="Hi")

    def test_empty_text_rejected(self):
        with pytest.raises(ValueError):
            ConversationTurn(speaker="customer", text="          ")

    def test_timestamp_becomes_tz_aware(self):
        naive_dt = datetime(2024, 1, 1, 12, 0, 0)
        turn = ConversationTurn(
            speaker="customer",
            text="This is a test message for timezone.",
            timestamp=naive_dt
        )
        assert turn.timestamp.tzinfo is not None


class TestCustomerServiceConversation:
    @pytest.fixture
    def valid_turns(self):
        return [
            ConversationTurn(speaker="customer", text="I have a billing issue with my account."),
            ConversationTurn(speaker="agent", text="I'd be happy to help you with that billing issue."),
        ]

    def test_valid_conversation(self, valid_turns):
        conv = CustomerServiceConversation(
            conversation_id="conv_001",
            intent="billing_issue",
            sentiment="negative",
            turns=valid_turns,
            resolution_status="resolved",
            language="en"
        )
        assert conv.conversation_id == "conv_001"
        assert len(conv.turns) == 2

    def test_first_turn_must_be_customer(self):
        agent_first = [
            ConversationTurn(speaker="agent", text="Hello, how can I help you today?"),
            ConversationTurn(speaker="customer", text="I need help with my account please."),
        ]
        with pytest.raises(ValueError, match="First turn must be from customer"):
            CustomerServiceConversation(
                conversation_id="conv_002",
                intent="general_inquiry",
                sentiment="neutral",
                turns=agent_first,
                resolution_status="resolved"
            )

    def test_invalid_language_rejected(self, valid_turns):
        with pytest.raises(ValueError, match="not supported"):
            CustomerServiceConversation(
                conversation_id="conv_003",
                intent="general_inquiry",
                sentiment="neutral",
                turns=valid_turns,
                resolution_status="resolved",
                language="invalid_lang"
            )

    def test_supported_languages(self, valid_turns):
        for lang in ["en", "es", "fr", "de"]:
            conv = CustomerServiceConversation(
                conversation_id=f"conv_{lang}",
                intent="general_inquiry",
                sentiment="neutral",
                turns=valid_turns,
                resolution_status="resolved",
                language=lang
            )
            assert conv.language == lang


class TestQualityMetrics:
    def test_valid_metrics(self):
        metrics = QualityMetrics(
            completeness_score=0.95,
            consistency_score=0.88,
            realism_score=0.92,
            diversity_score=0.75,
            overall_quality_score=87.5
        )
        assert metrics.overall_quality_score == 87.5

    def test_score_bounds(self):
        with pytest.raises(ValueError):
            QualityMetrics(
                completeness_score=1.5,
                consistency_score=0.88,
                realism_score=0.92,
                diversity_score=0.75,
                overall_quality_score=87.5
            )


class TestSchemaInfo:
    def test_schema_info_structure(self):
        info = get_schema_info()
        assert "version" in info
        assert "pydantic_v2" in info
        assert "domains" in info
        assert isinstance(info["domains"], list)

    def test_pydantic_version_detected(self):
        info = get_schema_info()
        assert isinstance(info["pydantic_v2"], bool)
    
    def test_customer_service_is_active_domain(self):
        info = get_schema_info()
        assert "customer_service" in info["domains"]
    
    def test_time_series_is_archived(self):
        info = get_schema_info()
        assert "time_series" in info.get("domains_archived", [])


class TestConstants:
    def test_constants_exist(self):
        assert SchemaConstants.MIN_TURNS == 2
        assert SchemaConstants.MAX_TURNS == 20

    def test_supported_languages_is_frozenset(self):
        assert isinstance(SchemaConstants.SUPPORTED_LANGUAGES, frozenset)
        assert "en" in SchemaConstants.SUPPORTED_LANGUAGES
