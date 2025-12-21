"""
Intent classification baseline using TF-IDF + LogisticRegression.

This module provides a simple but effective baseline for intent classification
on customer service conversations. Uses scikit-learn for feature extraction
and classification.

Usage:
    uv run python -m src.training.intent_classifier
"""

import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split


__all__ = ["IntentClassifier"]


class IntentClassifier:
    """
    Intent classification baseline using TF-IDF + LogisticRegression.
    
    This classifier extracts the first customer message from conversations
    and uses TF-IDF vectorization with logistic regression to predict intents.
    
    Attributes:
        vectorizer: TF-IDF vectorizer for text feature extraction
        model: LogisticRegression classifier
        label_encoder: Mapping from intent strings to integer labels
        label_decoder: Mapping from integer labels to intent strings
    
    Example:
        >>> classifier = IntentClassifier()
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
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42
    ):
        """
        Initialize the intent classifier.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract (default: unigrams and bigrams)
            random_state: Random seed for reproducibility
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode"
        )
        self.model: Optional[LogisticRegression] = None
        self.label_encoder: Dict[str, int] = {}
        self.label_decoder: Dict[int, str] = {}
        self._random_state = random_state
        self._training_metadata: Dict[str, Any] = {}
    
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
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train LogisticRegression classifier on conversation data.
        
        Pipeline:
        1. Extract first customer message and intent from each conversation
        2. Split into train/test sets
        3. Fit TF-IDF vectorizer on training data
        4. Train logistic regression classifier
        5. Evaluate on test set
        
        Args:
            conversations: List of conversation dictionaries
            test_size: Fraction of data to use for testing (default: 0.2)
            verbose: Whether to print progress (default: True)
        
        Returns:
            Dict with training results:
            - accuracy: Test set accuracy
            - f1_macro: Macro-averaged F1 score
            - f1_weighted: Weighted F1 score
            - train_size: Number of training samples
            - test_size: Number of test samples
            - unique_intents: Number of unique intent classes
        """
        if verbose:
            print("\n[1/5] Preparing data...")
        
        texts, labels = self.prepare_data(conversations)
        
        if verbose:
            print(f"       Total samples: {len(texts)}")
            print(f"       Unique intents: {len(set(labels))}")
        
        # Build label encoder
        unique_labels = sorted(set(labels))
        self.label_encoder = {label: i for i, label in enumerate(unique_labels)}
        self.label_decoder = {i: label for label, i in self.label_encoder.items()}
        
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
        
        if not use_stratify:
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
            print("\n[4/5] Training LogisticRegression...")
        
        # Train model
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=self._random_state,
            class_weight="balanced",  # Handle class imbalance
            solver="lbfgs"
        )
        self.model.fit(X_train, y_train)
        
        if verbose:
            print("\n[5/5] Evaluating on test set...")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        
        # Store metadata
        self._training_metadata = {
            "trained_at": datetime.now().isoformat(),
            "train_size": len(X_train_text),
            "test_size": len(X_test_text),
            "unique_intents": len(unique_labels),
            "vocab_size": len(self.vectorizer.vocabulary_),
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted
        }
        
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
        Save model, vectorizer, and encoders to disk.
        
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
            "label_encoder": self.label_encoder,
            "label_decoder": self.label_decoder,
            "metadata": self._training_metadata
        }
        
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        
        print(f"  Model saved to: {save_path}")
    
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
        
        classifier = cls()
        classifier.vectorizer = data["vectorizer"]
        classifier.model = data["model"]
        classifier.label_encoder = data["label_encoder"]
        classifier.label_decoder = data["label_decoder"]
        classifier._training_metadata = data.get("metadata", {})
        
        return classifier
    
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
    
    # 2. Train classifier
    print("\n" + "-" * 40)
    print("TRAINING")
    print("-" * 40)
    
    classifier = IntentClassifier()
    results = classifier.train(conversations, test_size=0.2, verbose=True)
    
    # 3. Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print(f"\n  Accuracy:     {results['accuracy']:.2%}")
    print(f"  F1 (macro):   {results['f1_macro']:.2%}")
    print(f"  F1 (weighted):{results['f1_weighted']:.2%}")
    print(f"\n  Train samples: {results['train_size']}")
    print(f"  Test samples:  {results['test_size']}")
    print(f"  Unique intents: {results['unique_intents']}")
    
    # 4. Save model
    print("\n" + "-" * 40)
    print("SAVING MODEL")
    print("-" * 40)
    
    model_path = "models/trained/intent_classifier.pkl"
    classifier.save(model_path)
    
    # 5. Test predictions
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
    
    # 6. Token usage and cost estimate
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
    
    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Model: TF-IDF + LogisticRegression")
    print(f"  Data:  {len(conversations)} synthetic conversations")
    print(f"  Accuracy: {results['accuracy']:.1%}")
    print(f"  F1 Score: {results['f1_macro']:.1%}")
    print(f"  Model saved: {model_path}")
    print(f"  Est. generation cost: ${cost_info['total_cost_usd']:.2f}")
    print("\n" + "=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

