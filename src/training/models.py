"""
Model configurations and abstractions for GENESIS LAB training module.

Provides dataclasses for configuring models, data preprocessing, and training,
along with preset configurations for quick experimentation.

Usage:
    from src.training.models import get_preset, ModelConfig, ExperimentConfig
    
    # Use a preset
    config = get_preset("best")
    
    # Or create custom config
    config = ExperimentConfig(
        model_config=ModelConfig(model_type="xgboost", ...),
        data_config=DataConfig(...),
        training_config=TrainingConfig(...)
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

__all__ = [
    "ModelConfig",
    "DataConfig", 
    "TrainingConfig",
    "ExperimentConfig",
    "PRESETS",
    "get_preset",
    "create_experiment_config",
    "ModelType",
]

# Type aliases
ModelType = Literal["logistic_regression", "random_forest", "xgboost"]


@dataclass
class ModelConfig:
    """
    Configuration for a classifier model.
    
    Defines the model type and its hyperparameters.
    
    Attributes:
        model_type: Algorithm to use (logistic_regression, random_forest, xgboost)
        name: Human-readable name for the configuration
        description: Description of the configuration's purpose
        max_features: Maximum TF-IDF features
        ngram_range: Range of n-grams (min, max)
        min_df: Minimum document frequency for TF-IDF
        model_params: Additional model-specific parameters
    """
    model_type: ModelType = "logistic_regression"
    name: str = "default"
    description: str = ""
    
    # TF-IDF parameters
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 2
    
    # Model-specific parameters (passed to sklearn/xgboost)
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type,
            "name": self.name,
            "description": self.description,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "min_df": self.min_df,
            "model_params": self.model_params,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create from dictionary."""
        return cls(
            model_type=data.get("model_type", "logistic_regression"),
            name=data.get("name", "default"),
            description=data.get("description", ""),
            max_features=data.get("max_features", 5000),
            ngram_range=tuple(data.get("ngram_range", (1, 2))),
            min_df=data.get("min_df", 2),
            model_params=data.get("model_params", {}),
        )


@dataclass
class DataConfig:
    """
    Configuration for data preprocessing.
    
    Attributes:
        language_filter: Optional language to filter by (e.g., "en", "es")
        min_turns: Minimum number of turns per conversation
        max_turns: Maximum number of turns per conversation
        intent_filter: Optional list of intents to include
        sentiment_filter: Optional list of sentiments to include
        shuffle: Whether to shuffle data before splitting
        stratify: Whether to use stratified splitting
    """
    language_filter: Optional[str] = None
    min_turns: int = 2
    max_turns: int = 20
    intent_filter: Optional[List[str]] = None
    sentiment_filter: Optional[List[str]] = None
    shuffle: bool = True
    stratify: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "language_filter": self.language_filter,
            "min_turns": self.min_turns,
            "max_turns": self.max_turns,
            "intent_filter": self.intent_filter,
            "sentiment_filter": self.sentiment_filter,
            "shuffle": self.shuffle,
            "stratify": self.stratify,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DataConfig":
        """Create from dictionary."""
        return cls(
            language_filter=data.get("language_filter"),
            min_turns=data.get("min_turns", 2),
            max_turns=data.get("max_turns", 20),
            intent_filter=data.get("intent_filter"),
            sentiment_filter=data.get("sentiment_filter"),
            shuffle=data.get("shuffle", True),
            stratify=data.get("stratify", True),
        )


@dataclass
class TrainingConfig:
    """
    Configuration for training process.
    
    Attributes:
        test_size: Fraction of data for testing (0.0 to 1.0)
        random_state: Random seed for reproducibility
        verbose: Whether to print progress
        save_model: Whether to save the trained model
        model_output_path: Path to save the model
        cross_validation_folds: Number of CV folds (0 to disable)
    """
    test_size: float = 0.2
    random_state: int = 42
    verbose: bool = True
    save_model: bool = True
    model_output_path: str = "models/trained/intent_classifier.pkl"
    cross_validation_folds: int = 0  # 0 = disabled, >0 = number of folds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "test_size": self.test_size,
            "random_state": self.random_state,
            "verbose": self.verbose,
            "save_model": self.save_model,
            "model_output_path": self.model_output_path,
            "cross_validation_folds": self.cross_validation_folds,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create from dictionary."""
        return cls(
            test_size=data.get("test_size", 0.2),
            random_state=data.get("random_state", 42),
            verbose=data.get("verbose", True),
            save_model=data.get("save_model", True),
            model_output_path=data.get("model_output_path", "models/trained/intent_classifier.pkl"),
            cross_validation_folds=data.get("cross_validation_folds", 0),
        )


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a training experiment.
    
    Combines model, data, and training configurations into a single
    experiment specification that can be saved and reproduced.
    
    Attributes:
        model_config: Model hyperparameters
        data_config: Data preprocessing settings
        training_config: Training process settings
        experiment_name: Name for the experiment
        tags: Optional tags for organization
        notes: Optional notes about the experiment
    """
    model_config: ModelConfig = field(default_factory=ModelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "experiment"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (serializable)."""
        return {
            "model_config": self.model_config.to_dict(),
            "data_config": self.data_config.to_dict(),
            "training_config": self.training_config.to_dict(),
            "experiment_name": self.experiment_name,
            "tags": self.tags,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Create from dictionary."""
        return cls(
            model_config=ModelConfig.from_dict(data.get("model_config", {})),
            data_config=DataConfig.from_dict(data.get("data_config", {})),
            training_config=TrainingConfig.from_dict(data.get("training_config", {})),
            experiment_name=data.get("experiment_name", "experiment"),
            tags=data.get("tags", []),
            notes=data.get("notes", ""),
        )


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

PRESETS: Dict[str, ExperimentConfig] = {
    "fast": ExperimentConfig(
        model_config=ModelConfig(
            model_type="logistic_regression",
            name="fast",
            description="Quick training with logistic regression - good for iteration",
            max_features=3000,
            ngram_range=(1, 2),
            min_df=2,
            model_params={"max_iter": 500, "solver": "lbfgs"},
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(verbose=True),
        experiment_name="fast_experiment",
        tags=["baseline", "fast"],
    ),
    
    "balanced": ExperimentConfig(
        model_config=ModelConfig(
            model_type="random_forest",
            name="balanced",
            description="Balanced speed/accuracy with random forest",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            model_params={"n_estimators": 100, "max_depth": None, "n_jobs": -1},
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(verbose=True),
        experiment_name="balanced_experiment",
        tags=["balanced", "random_forest"],
    ),
    
    "best": ExperimentConfig(
        model_config=ModelConfig(
            model_type="xgboost",
            name="best",
            description="Best accuracy with XGBoost - slower but most accurate",
            max_features=5000,
            ngram_range=(1, 3),
            min_df=1,
            model_params={
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "n_jobs": -1,
            },
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(verbose=True),
        experiment_name="best_experiment",
        tags=["best", "xgboost", "production"],
    ),
    
    "quick_test": ExperimentConfig(
        model_config=ModelConfig(
            model_type="logistic_regression",
            name="quick_test",
            description="Minimal config for testing pipeline",
            max_features=1000,
            ngram_range=(1, 1),
            min_df=1,
            model_params={"max_iter": 100},
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(verbose=False, save_model=False),
        experiment_name="quick_test",
        tags=["test", "debug"],
    ),
    
    "cross_validation": ExperimentConfig(
        model_config=ModelConfig(
            model_type="logistic_regression",
            name="cross_validation",
            description="5-fold cross-validation for robust evaluation",
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(
            verbose=True,
            cross_validation_folds=5,
            save_model=False,
        ),
        experiment_name="cv_experiment",
        tags=["evaluation", "cross_validation"],
    ),
}


def get_preset(name: str) -> ExperimentConfig:
    """
    Get a preset experiment configuration by name.
    
    Available presets:
    - "fast": Quick logistic regression for iteration
    - "balanced": Random forest with good speed/accuracy
    - "best": XGBoost for best accuracy
    - "quick_test": Minimal config for testing
    - "cross_validation": 5-fold CV for evaluation
    
    Args:
        name: Preset name
    
    Returns:
        ExperimentConfig for the preset
    
    Raises:
        ValueError: If preset name is not found
    
    Example:
        >>> config = get_preset("best")
        >>> print(config.model_config.model_type)
        'xgboost'
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    
    # Return a copy to prevent modifications to the original
    preset = PRESETS[name]
    return ExperimentConfig.from_dict(preset.to_dict())


def create_experiment_config(
    model_type: ModelType = "logistic_regression",
    experiment_name: str = "custom_experiment",
    test_size: float = 0.2,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    cross_validation_folds: int = 0,
    **kwargs
) -> ExperimentConfig:
    """
    Create a custom experiment configuration with common defaults.
    
    This is a convenience function for creating configurations without
    manually constructing all the dataclasses.
    
    Args:
        model_type: Algorithm to use
        experiment_name: Name for the experiment
        test_size: Fraction for testing
        max_features: TF-IDF features
        ngram_range: N-gram range
        cross_validation_folds: CV folds (0 to disable)
        **kwargs: Additional model parameters
    
    Returns:
        Configured ExperimentConfig
    
    Example:
        >>> config = create_experiment_config(
        ...     model_type="xgboost",
        ...     experiment_name="my_experiment",
        ...     n_estimators=150,
        ...     max_depth=8
        ... )
    """
    return ExperimentConfig(
        model_config=ModelConfig(
            model_type=model_type,
            name=experiment_name,
            max_features=max_features,
            ngram_range=ngram_range,
            model_params=kwargs,
        ),
        data_config=DataConfig(),
        training_config=TrainingConfig(
            test_size=test_size,
            cross_validation_folds=cross_validation_folds,
        ),
        experiment_name=experiment_name,
    )


def list_presets() -> List[Dict[str, str]]:
    """
    List all available presets with descriptions.
    
    Returns:
        List of dicts with preset info
    """
    return [
        {
            "name": name,
            "model_type": config.model_config.model_type,
            "description": config.model_config.description,
        }
        for name, config in PRESETS.items()
    ]
