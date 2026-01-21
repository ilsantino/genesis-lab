"""
Training module for GENESIS-LAB.

Provides classifiers, training orchestration, and experiment tracking
for synthetic data.

Quick start:
    from src.training import Trainer, get_preset
    
    # Train with preset configuration
    trainer = Trainer(config=get_preset("best"))
    result = trainer.train(conversations)
    
    # Or use IntentClassifier directly
    from src.training import IntentClassifier
    classifier = IntentClassifier(model_type="xgboost")
    classifier.train(conversations)
"""

from .intent_classifier import IntentClassifier, TrainingResult
from .models import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    ExperimentConfig,
    get_preset,
    create_experiment_config,
    list_presets,
    PRESETS,
)
from .trainer import (
    Trainer,
    ExperimentTracker,
    HyperparameterSearch,
    CVResult,
    SearchResult,
    ExperimentResult,
)

__all__ = [
    # Core classifier
    "IntentClassifier",
    "TrainingResult",
    # Configurations
    "ModelConfig",
    "DataConfig",
    "TrainingConfig",
    "ExperimentConfig",
    "PRESETS",
    "get_preset",
    "create_experiment_config",
    "list_presets",
    # Trainer and tracking
    "Trainer",
    "ExperimentTracker",
    "HyperparameterSearch",
    # Results
    "CVResult",
    "SearchResult",
    "ExperimentResult",
]


