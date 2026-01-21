"""
Unit tests for DatasetRegistry.

Tests CRUD operations, metrics updates, training runs, and statistics.
Uses isolated SQLite databases via tmp_path fixture.
"""

import pytest
from pathlib import Path

from src.registry.database import DatasetRegistry
from src.generation.schemas import QualityMetrics, BiasMetrics


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def registry(tmp_path):
    """Create isolated registry with temp SQLite DB."""
    db_path = tmp_path / "test_registry.db"
    return DatasetRegistry(db_path=str(db_path))


@pytest.fixture
def sample_quality_metrics():
    """Sample QualityMetrics for testing."""
    return QualityMetrics(
        completeness_score=0.95,
        consistency_score=0.90,
        realism_score=0.85,
        diversity_score=0.80,
        overall_quality_score=87.5,
        issues_found=[],
        warnings=["Low coverage"]
    )


@pytest.fixture
def sample_bias_metrics():
    """Sample BiasMetrics for testing."""
    return BiasMetrics(
        demographic_balance={"language": {"en": 0.5, "es": 0.5}},
        sentiment_distribution={"positive": 0.3, "neutral": 0.5, "negative": 0.2},
        topic_coverage={},
        bias_detected=False,
        bias_severity="none",
        recommendations=[]
    )


@pytest.fixture
def registered_dataset(registry):
    """Create and return a registered dataset ID."""
    dataset_id = registry.register_dataset(
        domain="customer_service",
        size=100,
        file_path="data/synthetic/test.json",
        file_format="json",
        model_used="claude_35_sonnet",
        notes="Test dataset"
    )
    return dataset_id


# ============================================================================
# CRUD TESTS
# ============================================================================

class TestDatasetCRUD:
    """Tests for CRUD operations on datasets."""
    
    def test_register_dataset(self, registry):
        """Should register dataset and return ID with correct format."""
        dataset_id = registry.register_dataset(
            domain="customer_service",
            size=50,
            file_path="data/synthetic/cs_test.json",
            file_format="json",
            model_used="claude_35_sonnet",
            notes="Test registration"
        )
        
        assert dataset_id is not None
        assert dataset_id.startswith("ds_")
        assert len(dataset_id) == 15  # ds_ + 12 hex chars
    
    def test_get_dataset_exists(self, registry, registered_dataset):
        """Should retrieve existing dataset with all fields."""
        dataset = registry.get_dataset(registered_dataset)
        
        assert dataset is not None
        assert dataset["id"] == registered_dataset
        assert dataset["domain"] == "customer_service"
        assert dataset["size"] == 100
        assert dataset["file_path"] == "data/synthetic/test.json"
        assert dataset["file_format"] == "json"
        assert dataset["model_used"] == "claude_35_sonnet"
        assert dataset["notes"] == "Test dataset"
    
    def test_get_dataset_not_found(self, registry):
        """Should return None for non-existent dataset."""
        result = registry.get_dataset("ds_nonexistent1")
        
        assert result is None
    
    def test_list_datasets_all(self, registry):
        """Should list all registered datasets."""
        # Register multiple datasets
        registry.register_dataset(
            domain="customer_service",
            size=10,
            file_path="data/test1.json",
            file_format="json"
        )
        registry.register_dataset(
            domain="customer_service",
            size=20,
            file_path="data/test2.json",
            file_format="json"
        )
        
        datasets = registry.list_datasets()
        
        assert len(datasets) == 2
    
    def test_list_datasets_filter_by_domain(self, registry):
        """Should filter datasets by domain."""
        # Register datasets in different domains
        registry.register_dataset(
            domain="customer_service",
            size=10,
            file_path="data/cs.json",
            file_format="json"
        )
        # Note: Currently only customer_service domain exists,
        # but the filter logic should still work
        
        cs_datasets = registry.list_datasets(domain="customer_service")
        other_datasets = registry.list_datasets(domain="time_series")
        
        assert len(cs_datasets) == 1
        assert len(other_datasets) == 0
    
    def test_delete_dataset_success(self, registry, registered_dataset):
        """Should delete existing dataset and return True."""
        result = registry.delete_dataset(registered_dataset)
        
        assert result is True
        
        # Verify deleted
        dataset = registry.get_dataset(registered_dataset)
        assert dataset is None
    
    def test_delete_dataset_not_found(self, registry):
        """Should return False when deleting non-existent dataset."""
        result = registry.delete_dataset("ds_nonexistent1")
        
        assert result is False
    
    def test_delete_cascades(self, registry, registered_dataset, sample_quality_metrics):
        """Should delete associated quality_metrics and training_runs."""
        # Add quality metrics
        registry.update_quality_metrics(registered_dataset, sample_quality_metrics)
        
        # Add training run
        registry.register_training_run(
            dataset_id=registered_dataset,
            model_type="logistic_regression",
            results={"accuracy": 0.85, "f1_score": 0.82}
        )
        
        # Delete dataset
        result = registry.delete_dataset(registered_dataset)
        assert result is True
        
        # Verify cascade - training history should be empty
        history = registry.get_training_history(registered_dataset)
        assert len(history) == 0


# ============================================================================
# METRICS TESTS
# ============================================================================

class TestMetricsUpdate:
    """Tests for quality and bias metrics updates."""
    
    def test_update_quality_metrics_insert(self, registry, registered_dataset, sample_quality_metrics):
        """Should insert new quality metrics."""
        registry.update_quality_metrics(registered_dataset, sample_quality_metrics)
        
        dataset = registry.get_dataset(registered_dataset)
        
        assert dataset["completeness"] == 0.95
        assert dataset["consistency"] == 0.90
        assert dataset["realism"] == 0.85
        assert dataset["diversity"] == 0.80
        assert dataset["quality_overall"] == 87.5
        assert dataset["quality_score"] == 87.5
    
    def test_update_quality_metrics_update(self, registry, registered_dataset, sample_quality_metrics):
        """Should update existing quality metrics."""
        # Insert first
        registry.update_quality_metrics(registered_dataset, sample_quality_metrics)
        
        # Update with new values
        updated_metrics = QualityMetrics(
            completeness_score=1.0,
            consistency_score=1.0,
            realism_score=0.90,
            diversity_score=0.85,
            overall_quality_score=93.5,
            issues_found=[],
            warnings=[]
        )
        registry.update_quality_metrics(registered_dataset, updated_metrics)
        
        dataset = registry.get_dataset(registered_dataset)
        
        assert dataset["completeness"] == 1.0
        assert dataset["consistency"] == 1.0
        assert dataset["quality_overall"] == 93.5
    
    def test_update_bias_metrics(self, registry, registered_dataset, sample_bias_metrics):
        """Should update bias_detected and bias_severity."""
        registry.update_bias_metrics(registered_dataset, sample_bias_metrics)
        
        dataset = registry.get_dataset(registered_dataset)
        
        assert dataset["bias_detected"] == 0  # False stored as 0
        assert dataset["bias_severity"] == "none"
    
    def test_update_bias_metrics_detected(self, registry, registered_dataset):
        """Should correctly store bias when detected."""
        bias_metrics = BiasMetrics(
            demographic_balance={"language": {"en": 0.9, "es": 0.1}},
            sentiment_distribution={"positive": 0.1, "neutral": 0.2, "negative": 0.7},
            topic_coverage={},
            bias_detected=True,
            bias_severity="high",
            recommendations=["Balance language distribution"]
        )
        registry.update_bias_metrics(registered_dataset, bias_metrics)
        
        dataset = registry.get_dataset(registered_dataset)
        
        assert dataset["bias_detected"] == 1  # True stored as 1
        assert dataset["bias_severity"] == "high"
    
    def test_quality_metrics_linked_to_dataset(self, registry, registered_dataset, sample_quality_metrics):
        """Should return metrics via join in get_dataset."""
        registry.update_quality_metrics(registered_dataset, sample_quality_metrics)
        
        # get_dataset should include quality metrics via LEFT JOIN
        dataset = registry.get_dataset(registered_dataset)
        
        assert "completeness" in dataset
        assert "consistency" in dataset
        assert "realism" in dataset
        assert "diversity" in dataset
        assert "quality_overall" in dataset


# ============================================================================
# TRAINING RUNS TESTS
# ============================================================================

class TestTrainingRuns:
    """Tests for training run registration and history."""
    
    def test_register_training_run(self, registry, registered_dataset):
        """Should register training run with metrics."""
        run_id = registry.register_training_run(
            dataset_id=registered_dataset,
            model_type="logistic_regression",
            results={
                "accuracy": 0.85,
                "f1_score": 0.82,
                "precision": 0.84,
                "recall": 0.80,
                "training_time": 2.5,
                "notes": "Baseline model"
            }
        )
        
        assert run_id is not None
        assert run_id.startswith("run_")
    
    def test_get_training_history(self, registry, registered_dataset):
        """Should return training history with all runs."""
        # Register multiple runs
        registry.register_training_run(
            dataset_id=registered_dataset,
            model_type="logistic_regression",
            results={"accuracy": 0.85}
        )
        registry.register_training_run(
            dataset_id=registered_dataset,
            model_type="xgboost",
            results={"accuracy": 0.90}
        )
        
        history = registry.get_training_history(registered_dataset)
        
        assert len(history) == 2
        # Verify both model types are present (order may vary with same timestamp)
        model_types = {run["model_type"] for run in history}
        assert model_types == {"logistic_regression", "xgboost"}
    
    def test_training_history_empty(self, registry, registered_dataset):
        """Should return empty list for dataset without training runs."""
        history = registry.get_training_history(registered_dataset)
        
        assert history == []


# ============================================================================
# STATS AND EDGE CASES
# ============================================================================

class TestStatsAndEdgeCases:
    """Tests for statistics and edge cases."""
    
    def test_get_stats_empty_db(self, registry):
        """Should return stats for empty database."""
        stats = registry.get_stats()
        
        assert stats["total_datasets"] == 0
        assert stats["total_training_runs"] == 0
        assert stats["datasets_by_domain"] == {}
        assert stats["average_quality_score"] is None
    
    def test_get_stats_populated(self, registry, sample_quality_metrics):
        """Should return accurate stats for populated database."""
        # Register multiple datasets
        ds1 = registry.register_dataset(
            domain="customer_service",
            size=100,
            file_path="data/cs1.json",
            file_format="json"
        )
        ds2 = registry.register_dataset(
            domain="customer_service",
            size=200,
            file_path="data/cs2.json",
            file_format="json"
        )
        
        # Add quality metrics to one
        registry.update_quality_metrics(ds1, sample_quality_metrics)
        
        # Add training run
        registry.register_training_run(
            dataset_id=ds1,
            model_type="logistic_regression",
            results={"accuracy": 0.85}
        )
        
        stats = registry.get_stats()
        
        assert stats["total_datasets"] == 2
        assert stats["total_training_runs"] == 1
        assert stats["datasets_by_domain"]["customer_service"] == 2
        assert stats["average_quality_score"] == 87.5  # Only ds1 has quality score
    
    def test_register_with_optional_notes_none(self, registry):
        """Should register dataset with notes=None."""
        dataset_id = registry.register_dataset(
            domain="customer_service",
            size=50,
            file_path="data/test.json",
            file_format="json",
            notes=None
        )
        
        dataset = registry.get_dataset(dataset_id)
        assert dataset["notes"] is None
    
    def test_register_with_optional_notes_string(self, registry):
        """Should register dataset with notes as string."""
        dataset_id = registry.register_dataset(
            domain="customer_service",
            size=50,
            file_path="data/test.json",
            file_format="json",
            notes="This is a test note"
        )
        
        dataset = registry.get_dataset(dataset_id)
        assert dataset["notes"] == "This is a test note"
    
    def test_list_datasets_ordered_by_date(self, registry):
        """Should list datasets ordered by generation_date DESC."""
        # Register in order
        ds1 = registry.register_dataset(
            domain="customer_service",
            size=10,
            file_path="data/first.json",
            file_format="json"
        )
        ds2 = registry.register_dataset(
            domain="customer_service",
            size=20,
            file_path="data/second.json",
            file_format="json"
        )
        
        datasets = registry.list_datasets()
        
        # Most recent first
        assert datasets[0]["id"] == ds2
        assert datasets[1]["id"] == ds1
