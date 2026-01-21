"""
Intent classification with multiple model support.

This module provides intent classification for customer service conversations
with support for multiple algorithms: LogisticRegression, RandomForest, XGBoost.

Usage:
    uv run python -m src.training.intent_classifier
"""

import json
import pickle
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix,
    f1_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# XGBoost is optional
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

ModelType = Literal["logistic_regression", "random_forest", "xgboost"]

__all__ = ["IntentClassifier", "ModelType", "XGBOOST_AVAILABLE", "TrainingResult"]


@dataclass
class TrainingResult:
    """
    Structured result from intent classifier training.
    
    Contains all metrics and artifacts from a training run.
    """
    accuracy: float
    f1_macro: float
    f1_weighted: float
    train_size: int
    test_size: int
    unique_intents: int
    model_type: str
    confusion_matrix: Optional[np.ndarray] = None
    label_names: List[str] = field(default_factory=list)
    classification_report: Optional[Dict[str, Any]] = None
    trained_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (serializable)."""
        result = {
            "accuracy": self.accuracy,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "unique_intents": self.unique_intents,
            "model_type": self.model_type,
            "trained_at": self.trained_at,
            "label_names": self.label_names,
        }
        if self.confusion_matrix is not None:
            result["confusion_matrix"] = self.confusion_matrix.tolist()
        if self.classification_report is not None:
            result["classification_report"] = self.classification_report
        return result


class IntentClassifier:
    """
    Intent classification with multiple model support.
    
    This classifier extracts the first customer message from conversations
    and uses TF-IDF vectorization with configurable classifiers to predict intents.
    
    Supported models:
    - logistic_regression: Fast, interpretable baseline
    - random_forest: Better with limited data, feature importance
    - xgboost: Often best accuracy, slower training
    
    Attributes:
        vectorizer: TF-IDF vectorizer for text feature extraction
        model: Trained classifier (LogisticRegression, RandomForest, or XGBoost)
        label_encoder: Mapping from intent strings to integer labels
        label_decoder: Mapping from integer labels to intent strings
    
    Example:
        >>> classifier = IntentClassifier(model_type="xgboost")
        >>> conversations = json.load(open("data/synthetic/customer_service_100.json"))
        >>> results = classifier.train(conversations)
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        >>> 
        >>> # Predict on new text
        >>> intent = classifier.predict("My card hasn't arrived yet")
        >>> print(f"Predicted intent: {intent}")
    """
    
    def __init__(
        self,
        model_type: ModelType = "logistic_regression",
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        random_state: int = 42
    ):
        """
        Initialize the intent classifier.
        
        Args:
            model_type: Algorithm to use ("logistic_regression", "random_forest", "xgboost")
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract (default: unigrams and bigrams)
            min_df: Minimum document frequency for TF-IDF (default: 2)
            random_state: Random seed for reproducibility
        """
        self._model_type = model_type
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode"
        )
        self.model = None
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self._sklearn_label_encoder: Optional[LabelEncoder] = None
        self._random_state = random_state
        self._training_metadata: Dict[str, Any] = {}
        self._last_training_result: Optional[TrainingResult] = None
        
        # Validate model type
        if model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not installed. Run: uv add xgboost")
    
    def _create_model(self):
        """Create the appropriate model based on model_type."""
        if self._model_type == "logistic_regression":
            return LogisticRegression(
                max_iter=1000,
                random_state=self._random_state,
                class_weight="balanced",
                solver="lbfgs"
            )
        elif self._model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                random_state=self._random_state,
                class_weight="balanced",
                n_jobs=-1
            )
        elif self._model_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self._random_state,
                use_label_encoder=False,
                eval_metric="mlogloss",
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")
    
    def prepare_data(self, conversations: List[Dict]) -> Tuple[List[str], List[str]]:
        """
        Extract features and labels from conversations.
        
        Uses the first customer message as input text and the intent as label.
        
        Args:
            conversations: List of conversation dictionaries with 'turns' and 'intent'
        
        Returns:
            Tuple of (texts, labels) where:
            - texts: List of first customer messages
            - labels: List of intent strings
        
        Raises:
            ValueError: If conversations are empty or missing required fields
        """
        if not conversations:
            raise ValueError("No conversations provided")
        
        texts: List[str] = []
        labels: List[str] = []
        skipped = 0
        
        for conv in conversations:
            intent = conv.get("intent")
            turns = conv.get("turns", [])
            
            # Find first customer message
            customer_text = None
            for turn in turns:
                if turn.get("speaker") == "customer":
                    customer_text = turn.get("text", "").strip()
                    break
            
            if not customer_text or not intent:
                skipped += 1
                continue
            
            texts.append(customer_text)
            labels.append(intent)
        
        if skipped > 0:
            print(f"  [WARN] Skipped {skipped} conversations with missing data")
        
        return texts, labels
    
    def train(
        self,
        conversations: List[Dict],
        test_size: float = 0.2,
        verbose: bool = True,
        return_dataclass: bool = False
    ) -> Dict[str, Any] | TrainingResult:
        """
        Train classifier on conversation data.
        
        Pipeline:
        1. Extract first customer message and intent from each conversation
        2. Split into train/test sets
        3. Fit TF-IDF vectorizer on training data
        4. Train classifier
        5. Evaluate on test set
        
        Args:
            conversations: List of conversation dictionaries
            test_size: Fraction of data to use for testing (default: 0.2)
            verbose: Whether to print progress (default: True)
            return_dataclass: If True, return TrainingResult dataclass
        
        Returns:
            Dict with training results (or TrainingResult if return_dataclass=True):
            - accuracy: Test set accuracy
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
            - train_size: Number of training samples
            - test_size: Number of test samples
            - unique_intents: Number of unique intent classes
            - confusion_matrix: Confusion matrix (if return_dataclass=True)
        """
        if verbose:
            print("\n[1/5] Preparing data...")
        
        texts, labels = self.prepare_data(conversations)
        
        if verbose:
            print(f"       Total samples: {len(texts)}")
            print(f"       Unique intents: {len(set(labels))}")
        
        # Build label encoder (both custom dict and sklearn)
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        
        # Also use sklearn LabelEncoder for compatibility
        self._sklearn_label_encoder = LabelEncoder()
        self._sklearn_label_encoder.fit(unique_labels)
        
        # Encode labels
        y = [self.label_encoder[label] for label in labels]
        
        if verbose:
            print("\n[2/5] Splitting train/test...")
        
        # Split data
        # Use stratified split only if we have enough samples per class
        from collections import Counter
        label_counts = Counter(y)
        min_count = min(label_counts.values())
        
        # Need at least 2 samples per class for stratification
        use_stratify = min_count >= 2
        
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            texts, y,
            test_size=test_size,
            random_state=self._random_state,
            stratify=y if use_stratify else None
        )
        
        if not use_stratify and verbose:
            print("       [Note] Stratification disabled (some classes have <2 samples)")
        
        if verbose:
            print(f"       Train size: {len(X_train_text)}")
            print(f"       Test size: {len(X_test_text)}")
        
        if verbose:
            print("\n[3/5] Fitting TF-IDF vectorizer...")
        
        # Fit TF-IDF on training data
        X_train = self.vectorizer.fit_transform(X_train_text)
        X_test = self.vectorizer.transform(X_test_text)
        
        if verbose:
            print(f"       Features: {X_train.shape[1]}")
        
        if verbose:
            model_name = self._model_type.replace("_", " ").title()
            print(f"\n[4/5] Training {model_name}...")
        
        # Create and train model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        
        if verbose:
            print("\n[5/5] Evaluating on test set...")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Compute confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Get classification report as dict
        class_report = classification_report(
            y_test, y_pred,
            target_names=[self.label_decoder[i] for i in range(len(unique_labels))],
            output_dict=True,
            zero_division=0
        )
        
        # Store metadata
        self._training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "model_type": self._model_type,
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
            "unique_intents": len(unique_labels),
            "vocab_size": len(self.vectorizer.vocabulary_),
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
        
        # Create TrainingResult
        training_result = TrainingResult(
            accuracy=accuracy,
            f1_macro=f1_macro,
            f1_weighted=f1_weighted,
            train_size=len(X_train_text),
            test_size=len(X_test_text),
            unique_intents=len(unique_labels),
            model_type=self._model_type,
            confusion_matrix=conf_matrix,
            label_names=unique_labels,
            classification_report=class_report,
            trained_at=datetime.now().isoformat()
        )
        self._last_training_result = training_result
        
        if return_dataclass:
            return training_result
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
            "unique_intents": len(unique_labels)
        }
    
    def predict(self, text: str) -> str:
        """
        Predict intent for a single text.
        
        Args:
            text: Input text (typically first customer message)
        
        Returns:
            Predicted intent string
        
        Raises:
            ValueError: If model is not trained
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        y_pred = self.model.predict(X)[0]
        
        return self.label_decoder[y_pred]
    
    def predict_proba(self, text: str) -> Dict[str, float]:
        """
        Get prediction probabilities for all intents.
        
        Args:
            text: Input text
        
        Returns:
            Dict mapping intent names to probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform([text])
        probs = self.model.predict_proba(X)[0]
        
        return {
            self.label_decoder[i]: prob 
            for i, prob in enumerate(probs)
        }
    
    def predict_batch(self, texts: List[str]) -> List[str]:
        """
        Predict intents for multiple texts.
        
        Args:
            texts: List of input texts
        
        Returns:
            List of predicted intent strings
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X = self.vectorizer.transform(texts)
        y_pred = self.model.predict(X)
        
        return [self.label_decoder[y] for y in y_pred]
    
    def save(self, path: str = "models/trained/intent_classifier.pkl") -> None:
        """
        Save model, vectorizer, and encoders to disk (single pickle file).
        
        Args:
            path: Output file path (pickle format)
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "model_type": self._model_type,
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "sklearn_label_encoder": self._sklearn_label_encoder,
            "metadata": self._training_metadata,
            "last_training_result": self._last_training_result.to_dict() if self._last_training_result else None
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"  Model saved to: {save_path}")
    
    def save_artifacts(
        self, 
        output_dir: str = "models/trained",
        prefix: str = "intent_classifier"
    ) -> Dict[str, str]:
        """
        Save model artifacts as separate joblib files.
        
        Creates separate files for:
        - {prefix}_model.joblib: The trained classifier
        - {prefix}_vectorizer.joblib: The TF-IDF vectorizer
        - {prefix}_label_encoder.joblib: The sklearn LabelEncoder
        - {prefix}_metadata.json: Training metadata and results
        
        Args:
            output_dir: Directory to save artifacts
            prefix: Prefix for file names
        
        Returns:
            Dict mapping artifact names to file paths
        """
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # Save model
        model_path = output_path / f"{prefix}_model.joblib"
        joblib.dump(self.model, model_path)
        artifacts["model"] = str(model_path)
        
        # Save vectorizer
        vectorizer_path = output_path / f"{prefix}_vectorizer.joblib"
        joblib.dump(self.vectorizer, vectorizer_path)
        artifacts["vectorizer"] = str(vectorizer_path)
        
        # Save label encoder
        encoder_path = output_path / f"{prefix}_label_encoder.joblib"
        encoder_data = {
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "sklearn_label_encoder": self._sklearn_label_encoder
        }
        joblib.dump(encoder_data, encoder_path)
        artifacts["label_encoder"] = str(encoder_path)
        
        # Save metadata as JSON
        metadata_path = output_path / f"{prefix}_metadata.json"
        metadata = {
            **self._training_metadata,
            "artifacts": artifacts
        }
        if self._last_training_result:
            metadata["training_result"] = self._last_training_result.to_dict()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        artifacts["metadata"] = str(metadata_path)
        
        print(f"  Artifacts saved to: {output_path}/")
        for name, path in artifacts.items():
            print(f"    - {name}: {Path(path).name}")
        
        return artifacts
    
    @classmethod
    def load_from_artifacts(
        cls, 
        input_dir: str = "models/trained",
        prefix: str = "intent_classifier"
    ) -> "IntentClassifier":
        """
        Load model from separate joblib artifact files.
        
        Args:
            input_dir: Directory containing artifacts
            prefix: Prefix used when saving
        
        Returns:
            Loaded IntentClassifier instance
        """
        input_path = Path(input_dir)
        
        # Load metadata to get model type
        metadata_path = input_path / f"{prefix}_metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        model_type = metadata.get("model_type", "logistic_regression")
        classifier = cls(model_type=model_type)
        
        # Load model
        model_path = input_path / f"{prefix}_model.joblib"
        classifier.model = joblib.load(model_path)
        
        # Load vectorizer
        vectorizer_path = input_path / f"{prefix}_vectorizer.joblib"
        classifier.vectorizer = joblib.load(vectorizer_path)
        
        # Load label encoder
        encoder_path = input_path / f"{prefix}_label_encoder.joblib"
        encoder_data = joblib.load(encoder_path)
        classifier.label_encoder = encoder_data["label_encoder"]
        classifier.label_decoder = encoder_data["label_decoder"]
        classifier._sklearn_label_encoder = encoder_data.get("sklearn_label_encoder")
        
        classifier._training_metadata = metadata
        
        return classifier
    
    @classmethod
    def load(cls, path: str = "models/trained/intent_classifier.pkl") -> "IntentClassifier":
        """
        Load model from disk.
        
        Args:
            path: Path to saved model file
        
        Returns:
            Loaded IntentClassifier instance
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        
        model_type = data.get("model_type", "logistic_regression")
        classifier = cls(model_type=model_type)
        classifier.vectorizer = data["vectorizer"]
        classifier.model = data["model"]
        classifier.label_encoder = data["label_encoder"]
        classifier.label_decoder = data["label_decoder"]
        classifier._sklearn_label_encoder = data.get("sklearn_label_encoder")
        classifier._training_metadata = data.get("metadata", {})
        
        # Reconstruct TrainingResult if available
        if data.get("last_training_result"):
            tr_data = data["last_training_result"]
            classifier._last_training_result = TrainingResult(
                accuracy=tr_data.get("accuracy", 0),
                f1_macro=tr_data.get("f1_macro", 0),
                f1_weighted=tr_data.get("f1_weighted", 0),
                train_size=tr_data.get("train_size", 0),
                test_size=tr_data.get("test_size", 0),
                unique_intents=tr_data.get("unique_intents", 0),
                model_type=tr_data.get("model_type", model_type),
                confusion_matrix=np.array(tr_data["confusion_matrix"]) if tr_data.get("confusion_matrix") else None,
                label_names=tr_data.get("label_names", []),
                classification_report=tr_data.get("classification_report"),
                trained_at=tr_data.get("trained_at", "")
            )
        
        return classifier
    
    def get_confusion_matrix(
        self, 
        conversations: List[Dict]
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Compute confusion matrix for given conversations.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Tuple of (confusion_matrix, label_names)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        texts, labels = self.prepare_data(conversations)
        predictions = self.predict_batch(texts)
        
        # Get unique labels in order
        unique_labels = sorted(set(labels) | set(predictions))
        label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        
        y_true = [label_to_idx[l] for l in labels]
        y_pred = [label_to_idx[p] for p in predictions]
        
        cm = confusion_matrix(y_true, y_pred)
        
        return cm, unique_labels
    
    @property
    def last_training_result(self) -> Optional[TrainingResult]:
        """Get the last training result."""
        return self._last_training_result
    
    def get_classification_report(
        self,
        conversations: List[Dict],
        return_dict: bool = False
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            conversations: List of conversation dictionaries
            return_dict: If True, return dict instead of string
        
        Returns:
            Classification report as string (or dict if return_dict=True)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        texts, labels = self.prepare_data(conversations)
        predictions = self.predict_batch(texts)
        
        return classification_report(
            labels, 
            predictions,
            output_dict=return_dict,
            zero_division=0
        )
    
    @property
    def training_metadata(self) -> Dict[str, Any]:
        """Get training metadata."""
        return self._training_metadata.copy()


def estimate_generation_cost(conversations: List[Dict]) -> Dict[str, Any]:
    """
    Estimate token usage and cost for generating conversations.
    
    Uses approximate token counts based on character length.
    Pricing: Claude 3.5 Sonnet on AWS Bedrock
    - Input: $0.003 per 1K tokens
    - Output: $0.015 per 1K tokens
    
    Args:
        conversations: List of generated conversations
    
    Returns:
        Dict with token estimates and costs
    """
    # Approximate tokens (roughly 4 chars per token for English)
    CHARS_PER_TOKEN = 4
    INPUT_COST_PER_1K = 0.003
    OUTPUT_COST_PER_1K = 0.015
    
    # Estimate input tokens (prompt template ~500 tokens per request)
    PROMPT_TOKENS_PER_REQUEST = 500
    
    total_output_chars = 0
    total_turns = 0
    
    for conv in conversations:
        # Count output characters (the generated conversation)
        for turn in conv.get("turns", []):
            total_output_chars += len(turn.get("text", ""))
            total_turns += 1
        
        # Add metadata fields
        total_output_chars += len(json.dumps(conv, default=str))
    
    # Calculate tokens
    total_input_tokens = PROMPT_TOKENS_PER_REQUEST * len(conversations)
    total_output_tokens = total_output_chars // CHARS_PER_TOKEN
    total_tokens = total_input_tokens + total_output_tokens
    
    # Calculate costs
    input_cost = (total_input_tokens / 1000) * INPUT_COST_PER_1K
    output_cost = (total_output_tokens / 1000) * OUTPUT_COST_PER_1K
    total_cost = input_cost + output_cost
    cost_per_item = total_cost / len(conversations) if conversations else 0
    
    return {
        "num_conversations": len(conversations),
        "total_turns": total_turns,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "total_tokens": total_tokens,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
        "cost_per_item_usd": cost_per_item
    }


def compare_models(conversations: List[Dict], test_size: float = 0.2) -> Dict[str, Dict]:
    """
    Compare all available models on the same data.
    
    Args:
        conversations: List of conversation dictionaries
        test_size: Fraction of data for testing
    
    Returns:
        Dict mapping model names to their results
    """
    models_to_test = ["logistic_regression", "random_forest"]
    if XGBOOST_AVAILABLE:
        models_to_test.append("xgboost")
    
    results = {}
    
    for model_type in models_to_test:
        print(f"\n  Training {model_type}...", end=" ", flush=True)
        
        try:
            classifier = IntentClassifier(model_type=model_type)
            result = classifier.train(conversations, test_size=test_size, verbose=False)
            results[model_type] = result
            print(f"Accuracy: {result['accuracy']:.1%}, F1: {result['f1_macro']:.1%}")
        except Exception as e:
            print(f"FAILED - {type(e).__name__}")
            # XGBoost can fail with sparse class labels - skip it
            results[model_type] = {
                "accuracy": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "error": str(e)
            }
    
    return results


def main():
    """Train intent classifier on synthetic data and evaluate."""
    print("=" * 60)
    print("INTENT CLASSIFIER TRAINING")
    print("=" * 60)
    
    # 1. Load data
    data_path = Path("data/synthetic/customer_service_100.json")
    
    if not data_path.exists():
        print(f"\n[ERROR] Data not found: {data_path}")
        print("  Run generate_100.py first to create the dataset.")
        return 1
    
    print(f"\n[LOAD] Loading data from: {data_path}")
    
    with open(data_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    print(f"       Loaded {len(conversations)} conversations")
    
    # 2. Compare all models
    print("\n" + "-" * 50)
    print("MODEL COMPARISON")
    print("-" * 50)
    
    all_results = compare_models(conversations, test_size=0.2)
    
    # Find best model (excluding failed ones)
    valid_results = {k: v for k, v in all_results.items() if "error" not in v}
    
    if not valid_results:
        print("\n  [ERROR] All models failed!")
        return 1
    
    best_model = max(valid_results.keys(), key=lambda k: valid_results[k]["accuracy"])
    best_accuracy = valid_results[best_model]["accuracy"]
    
    print(f"\n  Best model: {best_model} ({best_accuracy:.1%} accuracy)")
    
    # 3. Train best model with verbose output
    print("\n" + "-" * 50)
    print(f"DETAILED TRAINING: {best_model.upper()}")
    print("-" * 50)
    
    classifier = IntentClassifier(model_type=best_model)
    results = classifier.train(conversations, test_size=0.2, verbose=True)
    
    # 4. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n  Model:        {best_model}")
    print(f"  Accuracy:     {results['accuracy']:.2%}")
    print(f"  F1 (macro):   {results['f1_macro']:.2%}")
    print(f"  F1 (weighted):{results['f1_weighted']:.2%}")
    print(f"\n  Train samples: {results['train_size']}")
    print(f"  Test samples:  {results['test_size']}")
    print(f"  Unique intents: {results['unique_intents']}")
    
    # 5. Save best model
    print("\n" + "-" * 40)
    print("SAVING MODEL")
    print("-" * 40)
    
    model_path = "models/trained/intent_classifier.pkl"
    classifier.save(model_path)
    
    # 6. Test predictions
    print("\n" + "-" * 40)
    print("SAMPLE PREDICTIONS")
    print("-" * 40)
    
    test_samples = [
        "My card hasn't arrived yet, it's been 2 weeks",
        "I want to cancel my account",
        "How do I change my PIN?",
        "There's a charge I don't recognize on my statement",
        "Can I use Apple Pay with my card?",
        "Mi tarjeta no funciona en el cajero",  # Spanish
        "Quiero transferir dinero a otra cuenta",  # Spanish
    ]
    
    print()
    for text in test_samples:
        intent = classifier.predict(text)
        # Get top probability
        probs = classifier.predict_proba(text)
        top_prob = max(probs.values())
        print(f"  Input: \"{text[:50]}{'...' if len(text) > 50 else ''}\"")
        print(f"  -> {intent} ({top_prob:.1%} confidence)\n")
    
    # 7. Token usage and cost estimate
    print("\n" + "-" * 40)
    print("GENERATION COST ESTIMATE")
    print("-" * 40)
    
    cost_info = estimate_generation_cost(conversations)
    
    print(f"\n  Conversations: {cost_info['num_conversations']}")
    print(f"  Total turns:   {cost_info['total_turns']}")
    print(f"\n  Input tokens:  {cost_info['input_tokens']:,}")
    print(f"  Output tokens: {cost_info['output_tokens']:,}")
    print(f"  Total tokens:  {cost_info['total_tokens']:,}")
    print(f"\n  Input cost:    ${cost_info['input_cost_usd']:.4f}")
    print(f"  Output cost:   ${cost_info['output_cost_usd']:.4f}")
    print(f"  Total cost:    ${cost_info['total_cost_usd']:.4f}")
    print(f"  Cost per item: ${cost_info['cost_per_item_usd']:.4f}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print("\n  Model Comparison:")
    for model_name, res in sorted(all_results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = " <-- BEST" if model_name == best_model else ""
        print(f"    {model_name:20s}: {res['accuracy']:.1%} accuracy, {res['f1_macro']:.1%} F1{marker}")
    
    print(f"\n  Selected Model: {best_model}")
    print(f"  Data: {len(conversations)} synthetic conversations")
    print(f"  Model saved: {model_path}")
    print(f"  Est. generation cost: ${cost_info['total_cost_usd']:.2f}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

