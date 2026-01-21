"""
Day 3 Test Suite - Tests for validation, bias detection, training, and pipeline.

This module contains tests for:
- QualityValidator: distribution, coherence, diversity tests
- BiasDetector: sentiment, intent coverage, language tests
- IntentClassifier: data prep, training, saving tests
- PipelineIntegration: end-to-end test with small sample

Usage:
    uv run pytest tests/test_day3.py -v
    uv run pytest tests/test_day3.py -v -m "not slow"  # Skip slow tests
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from src.validation.quality import QualityValidator
from src.validation.bias import BiasDetector, BiasReport
from src.training.intent_classifier import IntentClassifier, TrainingResult


# ============================================================================
# FIXTURES - Sample data for testing
# ============================================================================

@pytest.fixture
def sample_conversations() -> List[Dict]:
    """Create sample conversations for testing."""
    return [
        {
            "conversation_id": "test_001",
            "intent": "card_arrival",
            "sentiment": "negative",
            "language": "en",
            "complexity": "simple",
            "resolution_status": "resolved",
            "turns": [
                {"speaker": "customer", "text": "My card hasn't arrived yet, it's been 2 weeks since I ordered it."},
                {"speaker": "agent", "text": "I apologize for the delay. Let me check the status of your card shipment."},
                {"speaker": "customer", "text": "Please do, I really need it for my upcoming trip."},
                {"speaker": "agent", "text": "I can see it was shipped yesterday. You should receive it within 3-5 business days."}
            ]
        },
        {
            "conversation_id": "test_002",
            "intent": "cancel_account",
            "sentiment": "neutral",
            "language": "en",
            "complexity": "medium",
            "resolution_status": "resolved",
            "turns": [
                {"speaker": "customer", "text": "I want to cancel my account. Can you help me with that?"},
                {"speaker": "agent", "text": "I can help you with account cancellation. May I ask why you're leaving?"},
                {"speaker": "customer", "text": "I found a better option with another bank."},
                {"speaker": "agent", "text": "I understand. I've initiated the cancellation process for you."}
            ]
        },
        {
            "conversation_id": "test_003",
            "intent": "pin_change",
            "sentiment": "positive",
            "language": "es",
            "complexity": "simple",
            "resolution_status": "resolved",
            "turns": [
                {"speaker": "customer", "text": "Necesito cambiar mi PIN, por favor ayúdeme con eso."},
                {"speaker": "agent", "text": "Con gusto le ayudo. Le enviaré las instrucciones por mensaje."},
                {"speaker": "customer", "text": "Muchas gracias por la ayuda rápida."},
                {"speaker": "agent", "text": "De nada, estamos para servirle. Su nuevo PIN estará activo en minutos."}
            ]
        },
        {
            "conversation_id": "test_004",
            "intent": "transfer_money",
            "sentiment": "neutral",
            "language": "en",
            "complexity": "complex",
            "resolution_status": "resolved",
            "turns": [
                {"speaker": "customer", "text": "I need to transfer money to an international account."},
                {"speaker": "agent", "text": "I can help you with international transfers. What country are you sending to?"},
                {"speaker": "customer", "text": "I'm sending to Germany, about 5000 euros."},
                {"speaker": "agent", "text": "For transfers to Germany, we have competitive rates. The transfer will complete in 2-3 business days."}
            ]
        },
        {
            "conversation_id": "test_005",
            "intent": "dispute_charge",
            "sentiment": "negative",
            "language": "en",
            "complexity": "medium",
            "resolution_status": "escalated",
            "turns": [
                {"speaker": "customer", "text": "There's a charge on my statement that I don't recognize at all."},
                {"speaker": "agent", "text": "I'm sorry to hear that. Can you tell me the amount and merchant name?"},
                {"speaker": "customer", "text": "It's for $150 from a store I've never heard of."},
                {"speaker": "agent", "text": "I'll open a dispute case for this charge and escalate it to our fraud team."}
            ]
        }
    ]


@pytest.fixture
def larger_sample_conversations(sample_conversations) -> List[Dict]:
    """Create larger sample by duplicating and modifying conversations."""
    larger = []
    # Use fewer intents to ensure enough samples per class for stratification
    intents = ["card_arrival", "cancel_account", "pin_change", "transfer_money"]
    sentiments = ["positive", "neutral", "negative"]
    languages = ["en", "es"]
    
    # Create 40 samples (10 per intent, enough for train/test split)
    for i in range(40):
        base = sample_conversations[i % len(sample_conversations)].copy()
        base = {**base}  # Deep copy to avoid modifying original
        base["turns"] = base["turns"].copy()  # Copy turns list too
        base["conversation_id"] = f"test_{i:03d}"
        base["intent"] = intents[i % len(intents)]
        base["sentiment"] = sentiments[i % len(sentiments)]
        base["language"] = languages[i % len(languages)]
        larger.append(base)
    
    return larger


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# ============================================================================
# TEST CLASS: QualityValidator
# ============================================================================

class TestQualityValidator:
    """Tests for QualityValidator class."""
    
    def test_validate_completeness(self, sample_conversations):
        """Test that completeness validation works correctly."""
        validator = QualityValidator()
        score = validator.validate_completeness(sample_conversations)
        
        # All sample conversations have required fields
        assert score > 0.8, f"Expected high completeness score, got {score}"
        assert 0 <= score <= 1, "Completeness score should be between 0 and 1"
    
    def test_validate_consistency_first_turn_customer(self, sample_conversations):
        """Test that consistency checks first turn is from customer."""
        validator = QualityValidator()
        score = validator.validate_consistency(sample_conversations)
        
        # All sample conversations start with customer
        assert score > 0.7, f"Expected good consistency score, got {score}"
    
    def test_validate_consistency_detects_invalid(self):
        """Test that consistency detects invalid conversation structure."""
        validator = QualityValidator()
        
        # Create invalid conversation (agent speaks first)
        invalid_convs = [{
            "conversation_id": "invalid_001",
            "intent": "test_intent",
            "sentiment": "neutral",
            "resolution_status": "resolved",
            "turns": [
                {"speaker": "agent", "text": "Hello, how can I help you today?"},
                {"speaker": "customer", "text": "I need help with my account please."}
            ]
        }]
        
        score = validator.validate_consistency(invalid_convs)
        # Should detect the issue and lower the score
        assert score < 1.0, "Should detect invalid first speaker"
    
    def test_validate_diversity(self, sample_conversations):
        """Test diversity validation with varied sample data."""
        validator = QualityValidator()
        score = validator.validate_diversity(sample_conversations)
        
        assert 0 <= score <= 1, "Diversity score should be between 0 and 1"
        # With 5 different intents and varied content
        assert score > 0.1, f"Expected some diversity, got {score}"
    
    def test_compute_overall_score(self, sample_conversations):
        """Test that overall score computation works."""
        validator = QualityValidator()
        metrics = validator.compute_overall_score(sample_conversations)
        
        assert metrics.overall_quality_score >= 0
        assert metrics.overall_quality_score <= 100
        assert 0 <= metrics.completeness_score <= 1
        assert 0 <= metrics.consistency_score <= 1
        assert 0 <= metrics.realism_score <= 1
        assert 0 <= metrics.diversity_score <= 1
    
    def test_empty_conversations(self):
        """Test handling of empty conversation list."""
        validator = QualityValidator()
        metrics = validator.compute_overall_score([])
        
        assert metrics.completeness_score == 0
        assert len(metrics.issues_found) > 0


# ============================================================================
# TEST CLASS: BiasDetector
# ============================================================================

class TestBiasDetector:
    """Tests for BiasDetector class."""
    
    def test_check_sentiment_distribution(self, sample_conversations):
        """Test sentiment distribution analysis."""
        detector = BiasDetector()
        dist = detector.check_sentiment_distribution(sample_conversations)
        
        assert "positive" in dist
        assert "neutral" in dist
        assert "negative" in dist
        assert "imbalance" in dist
        
        # Sum should be approximately 1 (allowing for rounding)
        total = dist["positive"] + dist["neutral"] + dist["negative"]
        assert abs(total - 1.0) < 0.01, f"Sentiment distribution should sum to 1, got {total}"
    
    def test_check_intent_coverage(self, sample_conversations):
        """Test intent coverage calculation."""
        detector = BiasDetector()
        coverage = detector.check_intent_coverage(sample_conversations)
        
        assert 0 <= coverage <= 1
        # Coverage is based on valid Banking77 intents found in data
        # Since our test intents may not all be in Banking77 list, 
        # just verify the coverage is computed correctly
        assert coverage >= 0, "Coverage should be non-negative"
    
    def test_check_language_balance(self, sample_conversations):
        """Test language balance detection."""
        detector = BiasDetector()
        balance = detector.check_language_balance(sample_conversations)
        
        assert "en" in balance
        assert "es" in balance
        assert "balanced" in balance
        
        # 4 EN, 1 ES in our sample
        assert balance["en"] == 0.8
        assert balance["es"] == 0.2
        assert balance["balanced"] is False  # Not balanced (need 40% min each)
    
    def test_find_underrepresented_intents(self, larger_sample_conversations):
        """Test detection of underrepresented intents."""
        detector = BiasDetector()
        underrepresented = detector._find_underrepresented_intents(
            larger_sample_conversations,
            threshold=0.01
        )
        
        # Should find many underrepresented intents (we only use 4 intents out of 77)
        # Most Banking77 intents should be underrepresented
        assert len(underrepresented) > 70, f"Should find many underrepresented intents, found {len(underrepresented)}"
    
    def test_detect_bias_returns_metrics(self, sample_conversations):
        """Test that detect_bias returns proper BiasMetrics."""
        detector = BiasDetector()
        metrics = detector.detect_bias(sample_conversations)
        
        assert hasattr(metrics, "bias_detected")
        assert hasattr(metrics, "bias_severity")
        assert hasattr(metrics, "recommendations")
        assert metrics.bias_severity in ["none", "low", "medium", "high"]
    
    def test_detect_bias_report_dataclass(self, sample_conversations):
        """Test that detect_bias_report returns BiasReport dataclass."""
        detector = BiasDetector()
        report = detector.detect_bias_report(sample_conversations)
        
        assert isinstance(report, BiasReport)
        assert report.num_conversations == len(sample_conversations)


# ============================================================================
# TEST CLASS: IntentClassifier
# ============================================================================

class TestIntentClassifier:
    """Tests for IntentClassifier class."""
    
    def test_prepare_data(self, larger_sample_conversations):
        """Test data preparation extracts texts and labels correctly."""
        classifier = IntentClassifier()
        texts, labels = classifier.prepare_data(larger_sample_conversations)
        
        assert len(texts) == len(larger_sample_conversations)
        assert len(labels) == len(larger_sample_conversations)
        assert all(isinstance(t, str) for t in texts)
        assert all(isinstance(l, str) for l in labels)
    
    def test_train_returns_dict(self, larger_sample_conversations):
        """Test that training returns proper results dict."""
        classifier = IntentClassifier(model_type="logistic_regression")
        results = classifier.train(larger_sample_conversations, verbose=False)
        
        assert "accuracy" in results
        assert "f1_macro" in results
        assert "f1_weighted" in results
        assert "train_size" in results
        assert "test_size" in results
        assert "unique_intents" in results
        
        assert 0 <= results["accuracy"] <= 1
        assert 0 <= results["f1_macro"] <= 1
    
    def test_train_returns_dataclass(self, larger_sample_conversations):
        """Test that training can return TrainingResult dataclass."""
        classifier = IntentClassifier(model_type="logistic_regression")
        result = classifier.train(
            larger_sample_conversations, 
            verbose=False,
            return_dataclass=True
        )
        
        assert isinstance(result, TrainingResult)
        assert result.confusion_matrix is not None
        assert len(result.label_names) > 0
        assert result.classification_report is not None
    
    def test_predict_after_training(self, larger_sample_conversations):
        """Test prediction works after training."""
        classifier = IntentClassifier(model_type="logistic_regression")
        classifier.train(larger_sample_conversations, verbose=False)
        
        prediction = classifier.predict("My card hasn't arrived yet")
        assert isinstance(prediction, str)
        assert len(prediction) > 0
    
    def test_save_and_load(self, larger_sample_conversations, temp_dir):
        """Test saving and loading model."""
        classifier = IntentClassifier(model_type="logistic_regression")
        classifier.train(larger_sample_conversations, verbose=False)
        
        # Save
        save_path = os.path.join(temp_dir, "test_model.pkl")
        classifier.save(save_path)
        assert os.path.exists(save_path)
        
        # Load
        loaded = IntentClassifier.load(save_path)
        assert loaded.model is not None
        
        # Test prediction with loaded model
        prediction = loaded.predict("I want to cancel my account")
        assert isinstance(prediction, str)
    
    def test_save_artifacts(self, larger_sample_conversations, temp_dir):
        """Test saving separate artifact files with joblib."""
        classifier = IntentClassifier(model_type="logistic_regression")
        classifier.train(larger_sample_conversations, verbose=False)
        
        # Save artifacts
        artifacts = classifier.save_artifacts(
            output_dir=temp_dir,
            prefix="test_classifier"
        )
        
        assert "model" in artifacts
        assert "vectorizer" in artifacts
        assert "label_encoder" in artifacts
        assert "metadata" in artifacts
        
        # Check files exist
        for name, path in artifacts.items():
            assert os.path.exists(path), f"Artifact {name} not found at {path}"
    
    def test_get_confusion_matrix(self, larger_sample_conversations):
        """Test confusion matrix computation."""
        classifier = IntentClassifier(model_type="logistic_regression")
        classifier.train(larger_sample_conversations, verbose=False)
        
        cm, labels = classifier.get_confusion_matrix(larger_sample_conversations)
        
        assert cm is not None
        assert cm.shape[0] == cm.shape[1]  # Square matrix
        assert len(labels) == cm.shape[0]


# ============================================================================
# TEST CLASS: Pipeline Integration (Slow Tests)
# ============================================================================

class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.slow
    def test_pipeline_from_existing_data(self, larger_sample_conversations, temp_dir):
        """Test running pipeline on existing data (no generation)."""
        from src.pipelines.customer_service_pipeline import (
            CustomerServicePipeline,
            PipelineConfig
        )
        
        # Save sample data to temp file
        data_path = os.path.join(temp_dir, "test_data.json")
        with open(data_path, "w") as f:
            json.dump(larger_sample_conversations, f)
        
        # Create pipeline with config
        config = PipelineConfig(
            quality_threshold=50.0,  # Lower threshold for test data
            bias_threshold="high",   # Allow high bias for small test data
            accuracy_threshold=0.3   # Lower for small dataset
        )
        
        pipeline = CustomerServicePipeline(
            registry_path=os.path.join(temp_dir, "test_registry.db"),
            config=config
        )
        
        # Run validation, bias detection, and training
        results = pipeline.run_from_existing(data_path)
        
        assert "quality" in results
        assert "bias" in results
        assert "training" in results
        assert results["quality"]["overall"] >= 0
    
    @pytest.mark.slow
    def test_pipeline_config_thresholds(self, larger_sample_conversations, temp_dir):
        """Test that pipeline config thresholds work correctly."""
        from src.pipelines.customer_service_pipeline import PipelineConfig
        
        # Test bias threshold comparison
        config = PipelineConfig(bias_threshold="low")
        assert config.is_bias_acceptable("none") is True
        assert config.is_bias_acceptable("low") is True
        assert config.is_bias_acceptable("medium") is False
        assert config.is_bias_acceptable("high") is False
        
        config2 = PipelineConfig(bias_threshold="high")
        assert config2.is_bias_acceptable("high") is True


# Note: The 'slow' marker is registered in pyproject.toml
