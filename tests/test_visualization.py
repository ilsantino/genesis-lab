"""
Unit tests for visualization utilities.

Tests all functions in src/utils/visualization.py including
data loading, analysis, percentage calculations, and summaries.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from src.utils.visualization import (
    load_conversations,
    analyze_conversations,
    get_sentiment_percentages,
    get_language_percentages,
    get_top_intents,
    conversations_to_dataframe,
    get_quality_summary,
    get_bias_summary,
    format_number,
    get_sample_conversations,
    compare_datasets,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_conversations() -> List[Dict[str, Any]]:
    """Create sample conversations with varied attributes."""
    return [
        {
            "conversation_id": "conv_1",
            "intent": "billing_issue",
            "category": "billing",
            "sentiment": "negative",
            "complexity": "medium",
            "language": "en",
            "turn_count": 4,
            "resolution_status": "resolved",
            "customer_emotion_arc": "frustrated_to_satisfied",
        },
        {
            "conversation_id": "conv_2",
            "intent": "account_inquiry",
            "category": "account",
            "sentiment": "neutral",
            "complexity": "simple",
            "language": "en",
            "turn_count": 2,
            "resolution_status": "resolved",
            "customer_emotion_arc": "neutral",
        },
        {
            "conversation_id": "conv_3",
            "intent": "technical_support",
            "category": "support",
            "sentiment": "negative",
            "complexity": "complex",
            "language": "es",
            "turn_count": 6,
            "resolution_status": "escalated",
            "customer_emotion_arc": "frustrated",
        },
        {
            "conversation_id": "conv_4",
            "intent": "billing_issue",
            "category": "billing",
            "sentiment": "positive",
            "complexity": "simple",
            "language": "en",
            "turn_count": 3,
            "resolution_status": "resolved",
            "customer_emotion_arc": "satisfied",
        },
        {
            "conversation_id": "conv_5",
            "intent": "account_inquiry",
            "category": "account",
            "sentiment": "neutral",
            "complexity": "medium",
            "language": "es",
            "turn_count": 4,
            "resolution_status": "resolved",
            "customer_emotion_arc": "neutral",
        },
    ]


@pytest.fixture
def sample_quality_metrics() -> Dict[str, float]:
    """Sample quality scores."""
    return {
        "completeness": 0.95,
        "consistency": 0.90,
        "realism": 0.85,
        "diversity": 0.80,
    }


@pytest.fixture
def sample_bias_metrics() -> Dict[str, Any]:
    """Sample bias detection results."""
    return {
        "severity": "low",
        "sentiment_distribution": {"positive": 0.2, "neutral": 0.5, "negative": 0.3},
        "intent_coverage": 0.75,
        "language_distribution": {"en": 0.6, "es": 0.4},
    }


# ============================================================================
# LOAD CONVERSATIONS TESTS
# ============================================================================

class TestLoadConversations:
    """Tests for load_conversations function."""
    
    def test_load_valid_file(self, tmp_path, sample_conversations):
        """Should load conversations from valid JSON file."""
        file_path = tmp_path / "conversations.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(sample_conversations, f)
        
        result = load_conversations(str(file_path))
        
        assert len(result) == 5
        assert result[0]["conversation_id"] == "conv_1"
    
    def test_load_missing_file(self):
        """Should return empty list for missing file."""
        result = load_conversations("nonexistent_file.json")
        
        assert result == []
    
    def test_load_empty_file(self, tmp_path):
        """Should return empty list for empty JSON array."""
        file_path = tmp_path / "empty.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump([], f)
        
        result = load_conversations(str(file_path))
        
        assert result == []
    
    def test_load_non_list_json(self, tmp_path):
        """Should return empty list if JSON is not a list."""
        file_path = tmp_path / "object.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"data": "not a list"}, f)
        
        result = load_conversations(str(file_path))
        
        assert result == []


# ============================================================================
# ANALYZE CONVERSATIONS TESTS
# ============================================================================

class TestAnalyzeConversations:
    """Tests for analyze_conversations function."""
    
    def test_analyze_normal_data(self, sample_conversations):
        """Should return correct analysis for normal data."""
        result = analyze_conversations(sample_conversations)
        
        assert result["count"] == 5
        assert result["intents"]["billing_issue"] == 2
        assert result["intents"]["account_inquiry"] == 2
        assert result["intents"]["technical_support"] == 1
        assert result["sentiments"]["negative"] == 2
        assert result["sentiments"]["neutral"] == 2
        assert result["sentiments"]["positive"] == 1
        assert result["languages"]["en"] == 3
        assert result["languages"]["es"] == 2
    
    def test_analyze_empty_list(self):
        """Should return zeroed stats for empty list."""
        result = analyze_conversations([])
        
        assert result["count"] == 0
        assert result["intents"] == {}
        assert result["sentiments"] == {}
        assert result["avg_turns"] == 0
    
    def test_analyze_missing_fields(self):
        """Should handle conversations with missing fields."""
        convs = [
            {"conversation_id": "1"},  # Missing most fields
            {"conversation_id": "2", "intent": "billing"},
        ]
        
        result = analyze_conversations(convs)
        
        assert result["count"] == 2
        assert "unknown" in result["intents"]
        assert result["intents"].get("billing") == 1


# ============================================================================
# PERCENTAGE CALCULATIONS TESTS
# ============================================================================

class TestPercentageCalculations:
    """Tests for percentage calculation functions."""
    
    def test_sentiment_percentages_normal(self):
        """Should calculate correct percentages."""
        sentiments = {"positive": 20, "neutral": 50, "negative": 30}
        
        result = get_sentiment_percentages(sentiments)
        
        assert result["positive"] == 20.0
        assert result["neutral"] == 50.0
        assert result["negative"] == 30.0
    
    def test_sentiment_percentages_empty(self):
        """Should return empty dict for empty input."""
        result = get_sentiment_percentages({})
        
        assert result == {}
    
    def test_sentiment_percentages_single(self):
        """Should handle single sentiment."""
        sentiments = {"positive": 10}
        
        result = get_sentiment_percentages(sentiments)
        
        assert result["positive"] == 100.0
    
    def test_language_percentages_normal(self):
        """Should calculate correct language percentages."""
        languages = {"en": 60, "es": 30, "fr": 10}
        
        result = get_language_percentages(languages)
        
        assert result["en"] == 60.0
        assert result["es"] == 30.0
        assert result["fr"] == 10.0
    
    def test_language_percentages_empty(self):
        """Should return empty dict for empty input."""
        result = get_language_percentages({})
        
        assert result == {}


# ============================================================================
# TOP INTENTS TESTS
# ============================================================================

class TestTopIntents:
    """Tests for get_top_intents function."""
    
    def test_top_intents_normal(self):
        """Should return top N intents."""
        intents = {
            "billing": 50,
            "support": 30,
            "inquiry": 20,
            "other": 10,
            "misc": 5,
        }
        
        result = get_top_intents(intents, top_n=3)
        
        assert len(result) == 3
        assert list(result.keys())[0] == "billing"
        assert result["billing"] == 50
    
    def test_top_intents_top_n_greater_than_len(self):
        """Should return all when top_n > number of intents."""
        intents = {"billing": 50, "support": 30}
        
        result = get_top_intents(intents, top_n=10)
        
        assert len(result) == 2
    
    def test_top_intents_empty(self):
        """Should return empty dict for empty input."""
        result = get_top_intents({}, top_n=5)
        
        assert result == {}


# ============================================================================
# DATAFRAME CONVERSION TESTS
# ============================================================================

class TestDataFrameConversion:
    """Tests for conversations_to_dataframe function."""
    
    def test_to_dataframe_normal(self, sample_conversations):
        """Should convert to DataFrame correctly."""
        result = conversations_to_dataframe(sample_conversations)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 5
        assert "conversation_id" in result.columns
        assert "intent" in result.columns
        assert "sentiment" in result.columns
        assert "language" in result.columns
    
    def test_to_dataframe_empty(self):
        """Should return empty DataFrame for empty input."""
        result = conversations_to_dataframe([])
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
    
    def test_to_dataframe_missing_fields(self):
        """Should handle missing fields with empty strings."""
        convs = [{"conversation_id": "1"}]  # Missing most fields
        
        result = conversations_to_dataframe(convs)
        
        assert len(result) == 1
        assert result.iloc[0]["intent"] == ""


# ============================================================================
# QUALITY SUMMARY TESTS
# ============================================================================

class TestQualitySummary:
    """Tests for get_quality_summary function."""
    
    def test_quality_summary_good_scores(self, sample_quality_metrics):
        """Should mark good scores correctly."""
        result = get_quality_summary(sample_quality_metrics)
        
        assert result["completeness"]["status"] == "good"
        assert result["consistency"]["status"] == "good"
        assert result["realism"]["status"] == "good"
        assert result["diversity"]["status"] == "good"
        assert "overall" in result
    
    def test_quality_summary_warning_scores(self):
        """Should mark warning scores correctly."""
        metrics = {
            "completeness": 0.80,  # Below 0.95 threshold but above 0.76
            "consistency": 0.75,   # Below 0.90 threshold but above 0.72
        }
        
        result = get_quality_summary(metrics)
        
        assert result["completeness"]["status"] == "warning"
        assert result["consistency"]["status"] == "warning"
    
    def test_quality_summary_error_scores(self):
        """Should mark error scores correctly."""
        metrics = {
            "completeness": 0.50,  # Well below threshold
            "diversity": 0.40,     # Well below threshold
        }
        
        result = get_quality_summary(metrics)
        
        assert result["completeness"]["status"] == "error"
        assert result["diversity"]["status"] == "error"
    
    def test_quality_summary_empty(self):
        """Should handle empty metrics."""
        result = get_quality_summary({})
        
        assert "overall" not in result


# ============================================================================
# BIAS SUMMARY TESTS
# ============================================================================

class TestBiasSummary:
    """Tests for get_bias_summary function."""
    
    def test_bias_summary_no_bias(self):
        """Should return no findings for balanced data."""
        metrics = {
            "severity": "none",
            "sentiment_distribution": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            "intent_coverage": 0.85,
            "language_distribution": {"en": 0.5, "es": 0.5},
        }
        
        result = get_bias_summary(metrics)
        
        assert result["severity"] == "none"
        assert len(result["findings"]) == 0
    
    def test_bias_summary_imbalanced_sentiment(self):
        """Should detect imbalanced sentiment."""
        metrics = {
            "severity": "medium",
            "sentiment_distribution": {"positive": 0.1, "neutral": 0.8, "negative": 0.1},
            "intent_coverage": 0.85,
            "language_distribution": {"en": 0.5, "es": 0.5},
        }
        
        result = get_bias_summary(metrics)
        
        assert "Sentiment distribution is imbalanced" in result["findings"]
        assert len(result["recommendations"]) > 0
    
    def test_bias_summary_low_intent_coverage(self):
        """Should detect low intent coverage."""
        metrics = {
            "severity": "high",
            "sentiment_distribution": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            "intent_coverage": 0.30,
            "language_distribution": {"en": 0.5, "es": 0.5},
        }
        
        result = get_bias_summary(metrics)
        
        findings_text = " ".join(result["findings"])
        assert "Low intent coverage" in findings_text
    
    def test_bias_summary_imbalanced_language(self):
        """Should detect imbalanced language distribution."""
        metrics = {
            "severity": "low",
            "sentiment_distribution": {"positive": 0.33, "neutral": 0.34, "negative": 0.33},
            "intent_coverage": 0.85,
            "language_distribution": {"en": 0.95, "es": 0.05},
        }
        
        result = get_bias_summary(metrics)
        
        assert "Language distribution is imbalanced" in result["findings"]


# ============================================================================
# FORMAT NUMBER TESTS
# ============================================================================

class TestFormatNumber:
    """Tests for format_number function."""
    
    def test_format_small_number(self):
        """Should format small numbers normally."""
        assert format_number(42) == "42.0"
        assert format_number(123.456) == "123.5"
    
    def test_format_thousands(self):
        """Should format thousands with K suffix."""
        assert format_number(1000) == "1.0K"
        assert format_number(5500) == "5.5K"
        assert format_number(999999) == "1000.0K"
    
    def test_format_millions(self):
        """Should format millions with M suffix."""
        assert format_number(1000000) == "1.0M"
        assert format_number(2500000) == "2.5M"
    
    def test_format_decimals(self):
        """Should respect decimals parameter."""
        assert format_number(1234, decimals=2) == "1.23K"
        assert format_number(1234567, decimals=2) == "1.23M"


# ============================================================================
# SAMPLE CONVERSATIONS TESTS
# ============================================================================

class TestSampleConversations:
    """Tests for get_sample_conversations function."""
    
    def test_get_sample_no_filter(self, sample_conversations):
        """Should return n samples without filter."""
        result = get_sample_conversations(sample_conversations, n=3)
        
        assert len(result) == 3
    
    def test_get_sample_intent_filter(self, sample_conversations):
        """Should filter by intent."""
        result = get_sample_conversations(
            sample_conversations,
            n=10,
            intent="billing_issue"
        )
        
        assert len(result) == 2
        assert all(c["intent"] == "billing_issue" for c in result)
    
    def test_get_sample_sentiment_filter(self, sample_conversations):
        """Should filter by sentiment."""
        result = get_sample_conversations(
            sample_conversations,
            n=10,
            sentiment="negative"
        )
        
        assert len(result) == 2
        assert all(c["sentiment"] == "negative" for c in result)
    
    def test_get_sample_combined_filter(self, sample_conversations):
        """Should apply both filters."""
        result = get_sample_conversations(
            sample_conversations,
            n=10,
            intent="billing_issue",
            sentiment="negative"
        )
        
        assert len(result) == 1
        assert result[0]["intent"] == "billing_issue"
        assert result[0]["sentiment"] == "negative"
    
    def test_get_sample_empty_result(self, sample_conversations):
        """Should return empty list when no matches."""
        result = get_sample_conversations(
            sample_conversations,
            n=10,
            intent="nonexistent_intent"
        )
        
        assert result == []


# ============================================================================
# COMPARE DATASETS TESTS
# ============================================================================

class TestCompareDatasets:
    """Tests for compare_datasets function."""
    
    def test_compare_same_size(self, sample_conversations):
        """Should compare two datasets of same size."""
        result = compare_datasets(sample_conversations, sample_conversations)
        
        assert result["dataset1_count"] == 5
        assert result["dataset2_count"] == 5
        assert result["intent_overlap"] == 1.0  # Same datasets
        assert result["avg_turns_diff"] == 0
    
    def test_compare_different_sizes(self, sample_conversations):
        """Should compare datasets of different sizes."""
        dataset1 = sample_conversations[:3]
        dataset2 = sample_conversations[2:]
        
        result = compare_datasets(dataset1, dataset2)
        
        assert result["dataset1_count"] == 3
        assert result["dataset2_count"] == 3
        assert 0 < result["intent_overlap"] <= 1.0
    
    def test_compare_with_empty(self, sample_conversations):
        """Should handle comparison with empty dataset."""
        result = compare_datasets(sample_conversations, [])
        
        assert result["dataset1_count"] == 5
        assert result["dataset2_count"] == 0
        assert result["intent_overlap"] == 0
    
    def test_compare_both_empty(self):
        """Should handle two empty datasets."""
        result = compare_datasets([], [])
        
        assert result["dataset1_count"] == 0
        assert result["dataset2_count"] == 0
    
    def test_compare_sentiment_diff(self, sample_conversations):
        """Should calculate sentiment differences."""
        # Create second dataset with different sentiments
        dataset2 = [
            {**c, "sentiment": "positive"} 
            for c in sample_conversations
        ]
        
        result = compare_datasets(sample_conversations, dataset2)
        
        assert "sentiment_comparison" in result
        assert "positive" in result["sentiment_comparison"]
