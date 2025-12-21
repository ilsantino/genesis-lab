"""
Quality validation module for synthetic data.

Validates generated data against reference datasets and quality metrics.
Supports customer service conversations and time series data.

Usage:
    uv run python -m src.validation.quality
"""

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation.schemas import QualityMetrics
from src.generation.templates.customer_service_prompts import ALL_INTENTS, INTENT_TO_CATEGORY


__all__ = ["QualityValidator"]


class QualityValidator:
    """
    Validates quality of generated synthetic customer service data.
    
    Compares synthetic data against banking77 reference dataset and
    checks for completeness, consistency, realism, and diversity.
    
    Example:
        >>> validator = QualityValidator()
        >>> conversations = json.load(open("data/synthetic/cs_smoke_v2.json"))
        >>> metrics = validator.compute_overall_score(conversations)
        >>> print(f"Quality: {metrics.overall_quality_score:.1f}/100")
    """
    
    # Required fields for validation
    REQUIRED_CONVERSATION_FIELDS = ["conversation_id", "intent", "sentiment", "turns", "resolution_status"]
    REQUIRED_TURN_FIELDS = ["speaker", "text"]
    VALID_SPEAKERS = {"customer", "agent"}
    VALID_SENTIMENTS = {"positive", "neutral", "negative"}
    
    def __init__(
        self, 
        reference_path: str = "data/reference/customer_service_reference.json"
    ):
        """
        Initialize validator with reference data.
        
        Args:
            reference_path: Path to banking77 reference dataset
        """
        self._reference_path = Path(reference_path)
        self._reference_data: Optional[List[Dict]] = None
        self._reference_intent_dist: Optional[Dict[str, float]] = None
        self._issues: List[str] = []
        self._warnings: List[str] = []
        
        # Load reference data
        self._load_reference_data()
    
    def _load_reference_data(self) -> None:
        """Load and process reference dataset."""
        if not self._reference_path.exists():
            self._warnings.append(f"Reference data not found: {self._reference_path}")
            return
        
        try:
            with open(self._reference_path, "r", encoding="utf-8") as f:
                self._reference_data = json.load(f)
            
            # Calculate intent distribution from reference
            if self._reference_data:
                intent_counts = Counter(
                    item.get("intent", item.get("label", "unknown")) 
                    for item in self._reference_data
                )
                total = sum(intent_counts.values())
                self._reference_intent_dist = {
                    intent: count / total 
                    for intent, count in intent_counts.items()
                }
        except Exception as e:
            self._warnings.append(f"Error loading reference data: {e}")
    
    def validate_completeness(self, conversations: List[Dict]) -> float:
        """
        Check all required fields are present in each conversation.
        
        Required fields:
        - conversation_id, intent, sentiment, turns, resolution_status
        - Each turn must have: speaker, text
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = all fields present in all conversations)
        """
        if not conversations:
            self._issues.append("No conversations to validate")
            return 0.0
        
        total_checks = 0
        passed_checks = 0
        
        for i, conv in enumerate(conversations):
            # Check top-level fields
            for field in self.REQUIRED_CONVERSATION_FIELDS:
                total_checks += 1
                if field in conv and conv[field] is not None:
                    passed_checks += 1
                else:
                    self._issues.append(f"Conv {i}: Missing field '{field}'")
            
            # Check turns
            turns = conv.get("turns", [])
            if isinstance(turns, list):
                for j, turn in enumerate(turns):
                    for field in self.REQUIRED_TURN_FIELDS:
                        total_checks += 1
                        if isinstance(turn, dict) and field in turn and turn[field]:
                            passed_checks += 1
                        else:
                            self._issues.append(f"Conv {i}, Turn {j}: Missing '{field}'")
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    def validate_consistency(self, conversations: List[Dict]) -> float:
        """
        Check conversational coherence.
        
        Checks:
        - First turn is always from customer
        - Speakers alternate (customer/agent pattern)
        - At least 2 turns per conversation
        - Sentiment is valid
        - Intent is from Banking77 list
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = all consistency checks pass)
        """
        if not conversations:
            return 0.0
        
        total_checks = 0
        passed_checks = 0
        
        for i, conv in enumerate(conversations):
            turns = conv.get("turns", [])
            
            # Check 1: At least 2 turns
            total_checks += 1
            if len(turns) >= 2:
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: Only {len(turns)} turns (need >=2)")
            
            # Check 2: First turn from customer
            total_checks += 1
            if turns and turns[0].get("speaker") == "customer":
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: First turn not from customer")
            
            # Check 3: Alternating speakers (soft check - allow some flexibility)
            total_checks += 1
            if self._check_alternating_speakers(turns):
                passed_checks += 1
            else:
                self._warnings.append(f"Conv {i}: Non-alternating speaker pattern")
            
            # Check 4: Valid sentiment
            total_checks += 1
            sentiment = conv.get("sentiment", "")
            if sentiment in self.VALID_SENTIMENTS:
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: Invalid sentiment '{sentiment}'")
            
            # Check 5: Valid intent (from Banking77)
            total_checks += 1
            intent = conv.get("intent", "")
            if intent in ALL_INTENTS:
                passed_checks += 1
            else:
                self._warnings.append(f"Conv {i}: Unknown intent '{intent}'")
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    def _check_alternating_speakers(self, turns: List[Dict]) -> bool:
        """Check if speakers generally alternate."""
        if len(turns) < 2:
            return True
        
        # Allow some flexibility - check that we don't have >2 consecutive same speakers
        consecutive = 1
        max_consecutive = 1
        
        for i in range(1, len(turns)):
            if turns[i].get("speaker") == turns[i-1].get("speaker"):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        
        return max_consecutive <= 2  # Allow up to 2 consecutive (some follow-ups are natural)
    
    def validate_realism(self, conversations: List[Dict]) -> float:
        """
        Compare intent distribution vs banking77 reference.
        
        Uses Jensen-Shannon divergence to measure distribution similarity.
        Score is converted to 0-1 range (1 = identical distributions).
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = perfect distribution match)
        """
        if not conversations:
            return 0.0
        
        if not self._reference_intent_dist:
            self._warnings.append("No reference data for realism check - using default score")
            return 0.5  # Neutral score when no reference
        
        # Calculate synthetic intent distribution
        synthetic_intents = Counter(conv.get("intent", "unknown") for conv in conversations)
        total = sum(synthetic_intents.values())
        synthetic_dist = {intent: count / total for intent, count in synthetic_intents.items()}
        
        # Get all unique intents from both distributions
        all_intents = set(self._reference_intent_dist.keys()) | set(synthetic_dist.keys())
        
        # Calculate Jensen-Shannon divergence
        js_div = self._jensen_shannon_divergence(
            self._reference_intent_dist,
            synthetic_dist,
            all_intents
        )
        
        # Convert to score (JS divergence is 0-1, where 0 is identical)
        # We want 1 to be identical, so invert
        realism_score = 1.0 - js_div
        
        # Also check coverage of reference intents
        reference_intents_used = len(set(synthetic_dist.keys()) & set(self._reference_intent_dist.keys()))
        coverage = reference_intents_used / len(self._reference_intent_dist) if self._reference_intent_dist else 0
        
        # Blend scores (70% distribution match, 30% coverage)
        final_score = (0.7 * realism_score) + (0.3 * coverage)
        
        return min(1.0, max(0.0, final_score))
    
    def _jensen_shannon_divergence(
        self, 
        p: Dict[str, float], 
        q: Dict[str, float],
        all_keys: Set[str]
    ) -> float:
        """
        Calculate Jensen-Shannon divergence between two distributions.
        
        Returns value in [0, 1] where 0 means identical distributions.
        """
        # Small epsilon to avoid log(0)
        eps = 1e-10
        
        # Create aligned distributions
        p_aligned = [p.get(k, eps) for k in all_keys]
        q_aligned = [q.get(k, eps) for k in all_keys]
        
        # Normalize
        p_sum = sum(p_aligned)
        q_sum = sum(q_aligned)
        p_aligned = [x / p_sum for x in p_aligned]
        q_aligned = [x / q_sum for x in q_aligned]
        
        # Calculate midpoint distribution
        m = [(p_aligned[i] + q_aligned[i]) / 2 for i in range(len(p_aligned))]
        
        # Calculate KL divergences
        kl_pm = sum(p_aligned[i] * math.log2(p_aligned[i] / m[i]) for i in range(len(p_aligned)) if p_aligned[i] > eps)
        kl_qm = sum(q_aligned[i] * math.log2(q_aligned[i] / m[i]) for i in range(len(q_aligned)) if q_aligned[i] > eps)
        
        # JS divergence is average of KL divergences
        js = (kl_pm + kl_qm) / 2
        
        # Normalize to [0, 1] (max JS divergence is 1 for log base 2)
        return min(1.0, js)
    
    def validate_diversity(self, conversations: List[Dict]) -> float:
        """
        Check variety in generated data.
        
        Metrics:
        - Intent variety: unique intents used / total available intents
        - Message variety: unique first customer messages / total conversations
        - Vocabulary size: unique words in customer messages
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = maximum diversity)
        """
        if not conversations:
            return 0.0
        
        # 1. Intent variety
        unique_intents = set(conv.get("intent", "") for conv in conversations)
        intent_variety = len(unique_intents) / len(ALL_INTENTS) if ALL_INTENTS else 0
        
        # 2. First message variety
        first_messages = []
        for conv in conversations:
            turns = conv.get("turns", [])
            if turns and turns[0].get("speaker") == "customer":
                first_messages.append(turns[0].get("text", "").lower().strip())
        
        unique_first_messages = len(set(first_messages))
        message_variety = unique_first_messages / len(conversations) if conversations else 0
        
        # 3. Vocabulary diversity (unique words / total words in customer messages)
        all_customer_words = []
        for conv in conversations:
            for turn in conv.get("turns", []):
                if turn.get("speaker") == "customer":
                    text = turn.get("text", "")
                    # Simple tokenization
                    words = text.lower().split()
                    all_customer_words.extend(words)
        
        unique_words = len(set(all_customer_words))
        total_words = len(all_customer_words)
        vocab_diversity = unique_words / total_words if total_words > 0 else 0
        
        # 4. Sentiment variety
        sentiments = set(conv.get("sentiment", "") for conv in conversations)
        sentiment_variety = len(sentiments) / len(self.VALID_SENTIMENTS)
        
        # Weighted average
        diversity_score = (
            0.30 * intent_variety +
            0.25 * message_variety +
            0.25 * vocab_diversity +
            0.20 * sentiment_variety
        )
        
        # Log diversity stats
        if len(unique_intents) < 5 and len(conversations) >= 10:
            self._warnings.append(f"Low intent variety: only {len(unique_intents)} unique intents")
        
        return min(1.0, max(0.0, diversity_score))
    
    def compute_overall_score(self, conversations: List[Dict]) -> QualityMetrics:
        """
        Run all validations and return QualityMetrics.
        
        Overall score = weighted average of all metrics, scaled to 0-100.
        Weights: completeness 25%, consistency 25%, realism 25%, diversity 25%
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            QualityMetrics with all scores and issues
        """
        # Reset issues/warnings for this run
        self._issues = []
        self._warnings = []
        
        # Run all validations
        completeness = self.validate_completeness(conversations)
        consistency = self.validate_consistency(conversations)
        realism = self.validate_realism(conversations)
        diversity = self.validate_diversity(conversations)
        
        # Calculate overall score (0-100)
        overall = (
            0.25 * completeness +
            0.25 * consistency +
            0.25 * realism +
            0.25 * diversity
        ) * 100
        
        return QualityMetrics(
            completeness_score=completeness,
            consistency_score=consistency,
            realism_score=realism,
            diversity_score=diversity,
            overall_quality_score=overall,
            issues_found=self._issues.copy(),
            warnings=self._warnings.copy(),
            metadata={
                "num_conversations": len(conversations),
                "reference_data": str(self._reference_path),
                "has_reference": self._reference_data is not None
            }
        )
    
    def print_report(self, metrics: QualityMetrics) -> None:
        """Print a formatted quality report."""
        print("\n" + "=" * 60)
        print("QUALITY VALIDATION REPORT")
        print("=" * 60)
        
        print(f"\nDataset: {metrics.metadata.get('num_conversations', 0)} conversations")
        print(f"Reference: {metrics.metadata.get('reference_data', 'N/A')}")
        
        print("\n" + "-" * 40)
        print("SCORES")
        print("-" * 40)
        
        def score_bar(score: float, width: int = 20) -> str:
            filled = int(score * width)
            return "[" + "#" * filled + "-" * (width - filled) + "]"
        
        print(f"  Completeness: {metrics.completeness_score:.2f} {score_bar(metrics.completeness_score)}")
        print(f"  Consistency:  {metrics.consistency_score:.2f} {score_bar(metrics.consistency_score)}")
        print(f"  Realism:      {metrics.realism_score:.2f} {score_bar(metrics.realism_score)}")
        print(f"  Diversity:    {metrics.diversity_score:.2f} {score_bar(metrics.diversity_score)}")
        
        print("\n" + "-" * 40)
        overall_bar = score_bar(metrics.overall_quality_score / 100, 30)
        print(f"  OVERALL: {metrics.overall_quality_score:.1f}/100 {overall_bar}")
        print("-" * 40)
        
        # Thresholds
        if metrics.overall_quality_score >= 85:
            print("\n  [PASS] Quality meets production threshold (>=85)")
        elif metrics.overall_quality_score >= 70:
            print("\n  [WARN] Quality is acceptable but could improve (70-85)")
        else:
            print("\n  [FAIL] Quality below acceptable threshold (<70)")
        
        # Issues
        if metrics.issues_found:
            print(f"\n  Issues ({len(metrics.issues_found)}):")
            for issue in metrics.issues_found[:5]:  # Show first 5
                print(f"    - {issue}")
            if len(metrics.issues_found) > 5:
                print(f"    ... and {len(metrics.issues_found) - 5} more")
        
        # Warnings
        if metrics.warnings:
            print(f"\n  Warnings ({len(metrics.warnings)}):")
            for warning in metrics.warnings[:5]:
                print(f"    - {warning}")
            if len(metrics.warnings) > 5:
                print(f"    ... and {len(metrics.warnings) - 5} more")
        
        print("\n" + "=" * 60)


def main():
    """Run quality validation on synthetic data."""
    # Try v2 first, fall back to original
    synthetic_paths = [
        Path("data/synthetic/cs_smoke_v2.json"),
        Path("data/synthetic/customer_service_smoke_test.json"),
    ]
    
    synthetic_path = None
    for path in synthetic_paths:
        if path.exists():
            synthetic_path = path
            break
    
    if not synthetic_path:
        print("[ERROR] No synthetic data found. Run smoke_test.py first.")
        print("  Tried:", [str(p) for p in synthetic_paths])
        return 1
    
    print(f"Loading synthetic data from: {synthetic_path}")
    
    with open(synthetic_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Create validator and run
    validator = QualityValidator()
    metrics = validator.compute_overall_score(conversations)
    
    # Print report
    validator.print_report(metrics)
    
    # Return exit code based on quality
    return 0 if metrics.overall_quality_score >= 70 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

