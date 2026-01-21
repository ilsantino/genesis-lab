"""
Training orchestration for GENESIS LAB.

Provides high-level training API with experiment tracking, hyperparameter search,
cross-validation, and registry integration.

Usage:
    from src.training import Trainer, get_preset
    
    # Quick training
    trainer = Trainer(config=get_preset("best"))
    result = trainer.train(conversations)
    
    # With hyperparameter search
    search_result = trainer.grid_search(
        param_grid={"max_features": [1000, 5000]},
        conversations=conversations
    )
"""

import itertools
import json
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .intent_classifier import IntentClassifier, TrainingResult
from .models import (
    ExperimentConfig,
    ModelConfig,
    DataConfig,
    TrainingConfig,
    get_preset,
)

__all__ = [
    "Trainer",
    "ExperimentTracker",
    "HyperparameterSearch",
    "CVResult",
    "SearchResult",
    "ExperimentResult",
]


# =============================================================================
# RESULT DATACLASSES
# =============================================================================

@dataclass
class CVResult:
    """
    Cross-validation results.
    
    Attributes:
        mean_accuracy: Mean accuracy across folds
        std_accuracy: Standard deviation of accuracy
        mean_f1: Mean F1 score across folds
        std_f1: Standard deviation of F1
        fold_results: List of results per fold
        num_folds: Number of CV folds used
    """
    mean_accuracy: float
    std_accuracy: float
    mean_f1: float
    std_f1: float
    fold_results: List[Dict[str, Any]] = field(default_factory=list)
    num_folds: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean_accuracy": self.mean_accuracy,
            "std_accuracy": self.std_accuracy,
            "mean_f1": self.mean_f1,
            "std_f1": self.std_f1,
            "fold_results": self.fold_results,
            "num_folds": self.num_folds,
        }
    
    def __str__(self) -> str:
        return (
            f"CVResult(accuracy={self.mean_accuracy:.2%} ± {self.std_accuracy:.2%}, "
            f"f1={self.mean_f1:.2%} ± {self.std_f1:.2%}, folds={self.num_folds})"
        )


@dataclass
class SearchResult:
    """
    Hyperparameter search results.
    
    Attributes:
        best_params: Best hyperparameters found
        best_score: Best score achieved
        all_results: All parameter combinations and scores
        search_time_seconds: Total search time
        num_combinations: Number of combinations tested
    """
    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Dict[str, Any]] = field(default_factory=list)
    search_time_seconds: float = 0.0
    num_combinations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.all_results,
            "search_time_seconds": self.search_time_seconds,
            "num_combinations": self.num_combinations,
        }
    
    def __str__(self) -> str:
        return (
            f"SearchResult(best_score={self.best_score:.2%}, "
            f"combinations={self.num_combinations}, "
            f"time={self.search_time_seconds:.1f}s)"
        )


@dataclass
class ExperimentResult:
    """
    Complete experiment result with all metrics and metadata.
    
    Attributes:
        experiment_id: Unique identifier
        config: Experiment configuration used
        metrics: Training metrics (accuracy, f1, etc.)
        artifacts: Paths to saved artifacts
        started_at: When experiment started
        completed_at: When experiment completed
        status: Experiment status (completed, failed, etc.)
        cv_result: Cross-validation results if performed
        search_result: Search results if performed
    """
    experiment_id: str
    config: ExperimentConfig
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    started_at: str = ""
    completed_at: str = ""
    status: str = "pending"
    cv_result: Optional[CVResult] = None
    search_result: Optional[SearchResult] = None
    training_result: Optional[TrainingResult] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "status": self.status,
            "cv_result": self.cv_result.to_dict() if self.cv_result else None,
            "search_result": self.search_result.to_dict() if self.search_result else None,
            "training_result": self.training_result.to_dict() if self.training_result else None,
        }


# =============================================================================
# EXPERIMENT TRACKER
# =============================================================================

class ExperimentTracker:
    """
    Track and manage training experiments.
    
    Provides methods to log metrics, parameters, and artifacts during training,
    and to retrieve and compare past experiments.
    
    Example:
        >>> tracker = ExperimentTracker(output_dir="experiments")
        >>> exp_id = tracker.start_experiment("my_exp", config)
        >>> tracker.log_metric("accuracy", 0.85)
        >>> tracker.end_experiment("completed")
    """
    
    def __init__(self, output_dir: str = "experiments"):
        """
        Initialize the experiment tracker.
        
        Args:
            output_dir: Directory to store experiment data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._current_experiment: Optional[ExperimentResult] = None
        self._experiments: Dict[str, ExperimentResult] = {}
        
        # Load existing experiments
        self._load_experiments()
    
    def _load_experiments(self) -> None:
        """Load existing experiments from disk."""
        experiments_file = self.output_dir / "experiments.json"
        if experiments_file.exists():
            with open(experiments_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp_id = exp_data["experiment_id"]
                    self._experiments[exp_id] = ExperimentResult(
                        experiment_id=exp_id,
                        config=ExperimentConfig.from_dict(exp_data.get("config", {})),
                        metrics=exp_data.get("metrics", {}),
                        artifacts=exp_data.get("artifacts", []),
                        started_at=exp_data.get("started_at", ""),
                        completed_at=exp_data.get("completed_at", ""),
                        status=exp_data.get("status", "unknown"),
                    )
    
    def _save_experiments(self) -> None:
        """Save experiments to disk."""
        experiments_file = self.output_dir / "experiments.json"
        data = {
            "experiments": [exp.to_dict() for exp in self._experiments.values()]
        }
        with open(experiments_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    
    def start_experiment(
        self,
        name: str,
        config: ExperimentConfig
    ) -> str:
        """
        Start a new experiment.
        
        Args:
            name: Experiment name
            config: Experiment configuration
        
        Returns:
            Unique experiment ID
        """
        experiment_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        self._current_experiment = ExperimentResult(
            experiment_id=experiment_id,
            config=config,
            started_at=datetime.now(timezone.utc).isoformat(),
            status="running",
        )
        
        self._experiments[experiment_id] = self._current_experiment
        self._save_experiments()
        
        return experiment_id
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None
    ) -> None:
        """
        Log a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number (for iterative metrics)
        """
        if self._current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        if step is not None:
            name = f"{name}_step_{step}"
        
        self._current_experiment.metrics[name] = value
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters.
        
        Args:
            params: Dictionary of parameters
        """
        if self._current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        # Store in metrics with param_ prefix
        for key, value in params.items():
            self._current_experiment.metrics[f"param_{key}"] = value
    
    def log_artifact(self, path: str, artifact_type: str = "model") -> None:
        """
        Log an artifact path.
        
        Args:
            path: Path to the artifact
            artifact_type: Type of artifact (model, data, etc.)
        """
        if self._current_experiment is None:
            raise RuntimeError("No active experiment. Call start_experiment() first.")
        
        self._current_experiment.artifacts.append(f"{artifact_type}:{path}")
    
    def end_experiment(self, status: str = "completed") -> ExperimentResult:
        """
        End the current experiment.
        
        Args:
            status: Final status (completed, failed, etc.)
        
        Returns:
            The completed ExperimentResult
        """
        if self._current_experiment is None:
            raise RuntimeError("No active experiment.")
        
        self._current_experiment.status = status
        self._current_experiment.completed_at = datetime.now(timezone.utc).isoformat()
        
        self._save_experiments()
        
        result = self._current_experiment
        self._current_experiment = None
        
        return result
    
    def get_experiment(self, experiment_id: str) -> Optional[ExperimentResult]:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)
    
    def list_experiments(
        self,
        status: Optional[str] = None,
        limit: int = 50
    ) -> List[ExperimentResult]:
        """
        List experiments, optionally filtered by status.
        
        Args:
            status: Filter by status
            limit: Maximum number to return
        
        Returns:
            List of experiments
        """
        experiments = list(self._experiments.values())
        
        if status:
            experiments = [e for e in experiments if e.status == status]
        
        # Sort by started_at descending
        experiments.sort(key=lambda e: e.started_at, reverse=True)
        
        return experiments[:limit]
    
    def compare_experiments(
        self,
        experiment_ids: List[str]
    ) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
        
        Returns:
            Comparison results
        """
        experiments = [self._experiments[eid] for eid in experiment_ids if eid in self._experiments]
        
        if not experiments:
            return {"error": "No experiments found"}
        
        # Extract common metrics
        all_metrics = set()
        for exp in experiments:
            all_metrics.update(exp.metrics.keys())
        
        comparison = {
            "experiments": [],
            "metrics_compared": list(all_metrics),
        }
        
        for exp in experiments:
            comparison["experiments"].append({
                "id": exp.experiment_id,
                "config_name": exp.config.model_config.name,
                "model_type": exp.config.model_config.model_type,
                "metrics": exp.metrics,
                "status": exp.status,
            })
        
        # Find best for each metric
        best = {}
        for metric in all_metrics:
            if metric.startswith("param_"):
                continue
            values = [(exp.experiment_id, exp.metrics.get(metric, 0)) for exp in experiments]
            if values:
                best[metric] = max(values, key=lambda x: x[1])
        
        comparison["best"] = best
        
        return comparison


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================

class HyperparameterSearch:
    """
    Grid and random search for optimal hyperparameters.
    
    Example:
        >>> search = HyperparameterSearch()
        >>> result = search.grid_search(
        ...     conversations,
        ...     model_type="xgboost",
        ...     param_grid={"max_features": [1000, 5000]}
        ... )
    """
    
    # Default parameter grids for each model type
    DEFAULT_PARAM_GRIDS = {
        "logistic_regression": {
            "max_features": [1000, 3000, 5000],
            "ngram_range": [(1, 1), (1, 2)],
            "min_df": [1, 2],
        },
        "random_forest": {
            "max_features": [3000, 5000],
            "ngram_range": [(1, 1), (1, 2)],
            "min_df": [1, 2],
        },
        "xgboost": {
            "max_features": [3000, 5000],
            "ngram_range": [(1, 1), (1, 2), (1, 3)],
            "min_df": [1, 2],
        },
    }
    
    def __init__(self, random_state: int = 42):
        """
        Initialize hyperparameter search.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
    
    def grid_search(
        self,
        conversations: List[Dict],
        model_type: str = "logistic_regression",
        param_grid: Optional[Dict[str, List]] = None,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> SearchResult:
        """
        Perform grid search over all parameter combinations.
        
        Args:
            conversations: Training data
            model_type: Model type to search
            param_grid: Parameters to search (uses defaults if None)
            test_size: Test set fraction
            verbose: Print progress
        
        Returns:
            SearchResult with best parameters
        """
        start_time = time.time()
        
        # Use default grid if none provided
        if param_grid is None:
            param_grid = self.DEFAULT_PARAM_GRIDS.get(model_type, {})
        
        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))
        
        if verbose:
            print(f"\n[Grid Search] {len(combinations)} combinations to test")
            print(f"  Model: {model_type}")
            print(f"  Parameters: {keys}")
        
        all_results = []
        best_score = 0
        best_params = {}
        
        for i, combo in enumerate(combinations):
            params = dict(zip(keys, combo))
            
            if verbose:
                print(f"\n  [{i+1}/{len(combinations)}] Testing: {params}")
            
            try:
                # Create classifier with these params
                classifier = IntentClassifier(
                    model_type=model_type,
                    max_features=params.get("max_features", 5000),
                    ngram_range=params.get("ngram_range", (1, 2)),
                    min_df=params.get("min_df", 2),
                    random_state=self.random_state,
                )
                
                # Train and evaluate
                result = classifier.train(
                    conversations,
                    test_size=test_size,
                    verbose=False
                )
                
                score = result["accuracy"]
                
                all_results.append({
                    "params": params,
                    "accuracy": score,
                    "f1_macro": result.get("f1_macro", 0),
                })
                
                if verbose:
                    print(f"    Accuracy: {score:.2%}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                if verbose:
                    print(f"    FAILED: {e}")
                all_results.append({
                    "params": params,
                    "error": str(e),
                })
        
        search_time = time.time() - start_time
        
        if verbose:
            print(f"\n[Grid Search Complete]")
            print(f"  Best accuracy: {best_score:.2%}")
            print(f"  Best params: {best_params}")
            print(f"  Time: {search_time:.1f}s")
        
        return SearchResult(
            best_params=best_params,
            best_score=best_score,
            all_results=all_results,
            search_time_seconds=search_time,
            num_combinations=len(combinations),
        )
    
    def random_search(
        self,
        conversations: List[Dict],
        model_type: str = "logistic_regression",
        param_distributions: Optional[Dict[str, List]] = None,
        n_iter: int = 10,
        test_size: float = 0.2,
        verbose: bool = True
    ) -> SearchResult:
        """
        Perform random search over parameter distributions.
        
        Args:
            conversations: Training data
            model_type: Model type
            param_distributions: Parameters to sample from
            n_iter: Number of iterations
            test_size: Test set fraction
            verbose: Print progress
        
        Returns:
            SearchResult with best parameters
        """
        start_time = time.time()
        
        # Use default grid if none provided
        if param_distributions is None:
            param_distributions = self.DEFAULT_PARAM_GRIDS.get(model_type, {})
        
        if verbose:
            print(f"\n[Random Search] {n_iter} iterations")
            print(f"  Model: {model_type}")
        
        rng = np.random.RandomState(self.random_state)
        all_results = []
        best_score = 0
        best_params = {}
        
        for i in range(n_iter):
            # Sample random parameters
            params = {
                key: rng.choice(values)
                for key, values in param_distributions.items()
            }
            
            if verbose:
                print(f"\n  [{i+1}/{n_iter}] Testing: {params}")
            
            try:
                classifier = IntentClassifier(
                    model_type=model_type,
                    max_features=params.get("max_features", 5000),
                    ngram_range=tuple(params.get("ngram_range", (1, 2))),
                    min_df=params.get("min_df", 2),
                    random_state=self.random_state,
                )
                
                result = classifier.train(
                    conversations,
                    test_size=test_size,
                    verbose=False
                )
                
                score = result["accuracy"]
                
                all_results.append({
                    "params": {k: (tuple(v) if isinstance(v, np.ndarray) else v) for k, v in params.items()},
                    "accuracy": score,
                    "f1_macro": result.get("f1_macro", 0),
                })
                
                if verbose:
                    print(f"    Accuracy: {score:.2%}")
                
                if score > best_score:
                    best_score = score
                    best_params = params
                    
            except Exception as e:
                if verbose:
                    print(f"    FAILED: {e}")
        
        search_time = time.time() - start_time
        
        return SearchResult(
            best_params={k: (tuple(v) if isinstance(v, np.ndarray) else v) for k, v in best_params.items()},
            best_score=best_score,
            all_results=all_results,
            search_time_seconds=search_time,
            num_combinations=n_iter,
        )


# =============================================================================
# TRAINER
# =============================================================================

class Trainer:
    """
    High-level training orchestration.
    
    Provides a unified interface for training intent classifiers with
    experiment tracking, cross-validation, and hyperparameter search.
    
    Example:
        >>> trainer = Trainer(config=get_preset("best"))
        >>> result = trainer.train(conversations)
        >>> print(f"Accuracy: {result.accuracy:.2%}")
        
        >>> # Cross-validation
        >>> cv_result = trainer.cross_validate(conversations, k=5)
        >>> print(f"Mean accuracy: {cv_result.mean_accuracy:.2%}")
    """
    
    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        tracker: Optional[ExperimentTracker] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            config: Experiment configuration (uses "balanced" preset if None)
            tracker: Optional experiment tracker (creates new if None)
        """
        self.config = config or get_preset("balanced")
        self.tracker = tracker or ExperimentTracker()
        self.search = HyperparameterSearch(
            random_state=self.config.training_config.random_state
        )
        
        self._classifier: Optional[IntentClassifier] = None
        self._last_result: Optional[ExperimentResult] = None
    
    def train(
        self,
        conversations: List[Dict],
        track_experiment: bool = True
    ) -> TrainingResult:
        """
        Train an intent classifier.
        
        Args:
            conversations: List of conversation dictionaries
            track_experiment: Whether to track in experiment tracker
        
        Returns:
            TrainingResult with metrics
        """
        model_config = self.config.model_config
        training_config = self.config.training_config
        
        # Start experiment tracking
        if track_experiment:
            exp_id = self.tracker.start_experiment(
                self.config.experiment_name,
                self.config
            )
        
        try:
            # Create classifier
            self._classifier = IntentClassifier(
                model_type=model_config.model_type,
                max_features=model_config.max_features,
                ngram_range=model_config.ngram_range,
                min_df=model_config.min_df,
                random_state=training_config.random_state,
            )
            
            # Train
            result = self._classifier.train(
                conversations,
                test_size=training_config.test_size,
                verbose=training_config.verbose,
                return_dataclass=True,
            )
            
            # Log metrics
            if track_experiment:
                self.tracker.log_metric("accuracy", result.accuracy)
                self.tracker.log_metric("f1_macro", result.f1_macro)
                self.tracker.log_metric("f1_weighted", result.f1_weighted)
                self.tracker.log_metric("train_size", result.train_size)
                self.tracker.log_metric("test_size", result.test_size)
            
            # Save model if configured
            if training_config.save_model:
                self._classifier.save(training_config.model_output_path)
                if track_experiment:
                    self.tracker.log_artifact(
                        training_config.model_output_path,
                        "model"
                    )
            
            # End experiment
            if track_experiment:
                exp_result = self.tracker.end_experiment("completed")
                exp_result.training_result = result
                self._last_result = exp_result
            
            return result
            
        except Exception as e:
            if track_experiment:
                self.tracker.end_experiment("failed")
            raise
    
    def evaluate(
        self,
        conversations: List[Dict]
    ) -> Dict[str, Any]:
        """
        Evaluate the trained classifier on new data.
        
        Args:
            conversations: Test conversations
        
        Returns:
            Evaluation metrics
        """
        if self._classifier is None:
            raise ValueError("No trained classifier. Call train() first.")
        
        texts, labels = self._classifier.prepare_data(conversations)
        predictions = self._classifier.predict_batch(texts)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score
        
        accuracy = accuracy_score(labels, predictions)
        f1_macro = f1_score(labels, predictions, average="macro", zero_division=0)
        f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "num_samples": len(labels),
            "unique_intents": len(set(labels)),
        }
    
    def cross_validate(
        self,
        conversations: List[Dict],
        k: int = 5
    ) -> CVResult:
        """
        Perform k-fold cross-validation.
        
        Args:
            conversations: Training data
            k: Number of folds
        
        Returns:
            CVResult with mean and std metrics
        """
        model_config = self.config.model_config
        training_config = self.config.training_config
        
        if training_config.verbose:
            print(f"\n[Cross-Validation] {k}-fold")
            print(f"  Model: {model_config.model_type}")
        
        # Prepare data
        temp_classifier = IntentClassifier(model_type=model_config.model_type)
        texts, labels = temp_classifier.prepare_data(conversations)
        
        # Encode labels for stratification
        unique_labels = sorted(set(labels))
        label_to_idx = {l: i for i, l in enumerate(unique_labels)}
        y = np.array([label_to_idx[l] for l in labels])
        
        # K-fold
        kfold = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=training_config.random_state
        )
        
        fold_results = []
        accuracies = []
        f1_scores = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(texts, y)):
            if training_config.verbose:
                print(f"\n  Fold {fold_idx + 1}/{k}...", end=" ")
            
            # Create subset conversations
            train_convs = [conversations[i] for i in train_idx]
            test_convs = [conversations[i] for i in test_idx]
            
            # Train
            classifier = IntentClassifier(
                model_type=model_config.model_type,
                max_features=model_config.max_features,
                ngram_range=model_config.ngram_range,
                min_df=model_config.min_df,
                random_state=training_config.random_state + fold_idx,
            )
            
            # Train on fold (use all data, no split)
            result = classifier.train(
                train_convs,
                test_size=0.0001,  # Minimal split
                verbose=False
            )
            
            # Evaluate on held-out fold
            eval_result = self._evaluate_classifier(classifier, test_convs)
            
            fold_results.append({
                "fold": fold_idx + 1,
                "accuracy": eval_result["accuracy"],
                "f1_macro": eval_result["f1_macro"],
                "train_size": len(train_idx),
                "test_size": len(test_idx),
            })
            
            accuracies.append(eval_result["accuracy"])
            f1_scores.append(eval_result["f1_macro"])
            
            if training_config.verbose:
                print(f"accuracy={eval_result['accuracy']:.2%}")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_f1 = np.mean(f1_scores)
        std_f1 = np.std(f1_scores)
        
        if training_config.verbose:
            print(f"\n[CV Results]")
            print(f"  Accuracy: {mean_acc:.2%} ± {std_acc:.2%}")
            print(f"  F1 Score: {mean_f1:.2%} ± {std_f1:.2%}")
        
        return CVResult(
            mean_accuracy=mean_acc,
            std_accuracy=std_acc,
            mean_f1=mean_f1,
            std_f1=std_f1,
            fold_results=fold_results,
            num_folds=k,
        )
    
    def _evaluate_classifier(
        self,
        classifier: IntentClassifier,
        conversations: List[Dict]
    ) -> Dict[str, float]:
        """Evaluate a classifier on conversations."""
        texts, labels = classifier.prepare_data(conversations)
        predictions = classifier.predict_batch(texts)
        
        from sklearn.metrics import accuracy_score, f1_score
        
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        }
    
    def grid_search(
        self,
        conversations: List[Dict],
        param_grid: Optional[Dict[str, List]] = None
    ) -> SearchResult:
        """
        Perform grid search for best hyperparameters.
        
        Args:
            conversations: Training data
            param_grid: Parameters to search
        
        Returns:
            SearchResult with best parameters
        """
        return self.search.grid_search(
            conversations,
            model_type=self.config.model_config.model_type,
            param_grid=param_grid,
            test_size=self.config.training_config.test_size,
            verbose=self.config.training_config.verbose,
        )
    
    def random_search(
        self,
        conversations: List[Dict],
        param_distributions: Optional[Dict[str, List]] = None,
        n_iter: int = 10
    ) -> SearchResult:
        """
        Perform random search for best hyperparameters.
        
        Args:
            conversations: Training data
            param_distributions: Parameters to sample from
            n_iter: Number of iterations
        
        Returns:
            SearchResult with best parameters
        """
        return self.search.random_search(
            conversations,
            model_type=self.config.model_config.model_type,
            param_distributions=param_distributions,
            n_iter=n_iter,
            test_size=self.config.training_config.test_size,
            verbose=self.config.training_config.verbose,
        )
    
    def save_experiment(self, path: str) -> None:
        """
        Save the last experiment result to a file.
        
        Args:
            path: Output file path
        """
        if self._last_result is None:
            raise ValueError("No experiment to save. Run train() first.")
        
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self._last_result.to_dict(), f, indent=2, default=str)
    
    @property
    def classifier(self) -> Optional[IntentClassifier]:
        """Get the trained classifier."""
        return self._classifier
    
    @property
    def last_result(self) -> Optional[ExperimentResult]:
        """Get the last experiment result."""
        return self._last_result
