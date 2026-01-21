"""
Quality validation module for synthetic data.

Validates generated data against reference datasets and quality metrics.
Supports customer service conversations with comprehensive metrics.

Metrics:
- Completeness: Required fields present
- Consistency: Conversational coherence (structural + semantic)
- Realism: Distribution matching vs reference (JS divergence)
- Diversity: Intent, vocabulary, n-gram, sentiment variety

Usage:
    validator = QualityValidator()
    metrics = validator.compute_overall_score(conversations)
    validator.print_report(metrics)
"""

import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.generation.schemas import QualityMetrics
from src.generation.templates.customer_service_prompts import (
    ALL_INTENTS, 
    INTENT_TO_CATEGORY,
    BANKING77_INTENTS,
)


__all__ = ["QualityValidator"]

logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Validates quality of generated synthetic customer service data.
    
    Compares synthetic data against banking77 reference dataset and
    checks for completeness, consistency, realism, and diversity.
    
    Features:
    - Distribution matching (Jensen-Shannon divergence)
    - Conversational coherence (structural and semantic)
    - N-gram diversity analysis
    - Intent/category coverage
    - Optional embedding-based semantic similarity
    
    Example:
        >>> validator = QualityValidator()
        >>> conversations = json.load(open("data/synthetic/customer_service_100.json"))
        >>> metrics = validator.compute_overall_score(conversations)
        >>> print(f"Quality: {metrics.overall_quality_score:.1f}/100")
    """
    
    # Required fields for validation
    REQUIRED_CONVERSATION_FIELDS = [
        "conversation_id", "intent", "sentiment", "turns", "resolution_status"
    ]
    REQUIRED_TURN_FIELDS = ["speaker", "text"]
    VALID_SPEAKERS = {"customer", "agent"}
    VALID_SENTIMENTS = {"positive", "neutral", "negative"}
    VALID_RESOLUTIONS = {"resolved", "escalated", "unresolved"}
    
    def __init__(
        self, 
        reference_path: str = "data/reference/customer_service_reference.json",
        use_embeddings: bool = False
    ):
        """
        Initialize validator with reference data.
        
        Args:
            reference_path: Path to banking77 reference dataset
            use_embeddings: Whether to use sentence embeddings for coherence
        """
        self._reference_path = Path(reference_path)
        self._reference_data: Optional[List[Dict]] = None
        self._reference_intent_dist: Optional[Dict[str, float]] = None
        self._issues: List[str] = []
        self._warnings: List[str] = []
        self._use_embeddings = use_embeddings
        self._embedding_model = None
        
        # Detailed metrics storage
        self._detailed_metrics: Dict[str, Any] = {}
        
        # Load reference data
        self._load_reference_data()
        
        # Load embedding model if requested
        if use_embeddings:
            self._load_embedding_model()
    
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
                
                logger.info(
                    f"Loaded reference data: {len(self._reference_data)} items, "
                    f"{len(self._reference_intent_dist)} unique intents"
                )
        except Exception as e:
            self._warnings.append(f"Error loading reference data: {e}")
            logger.error(f"Failed to load reference data: {e}")
    
    def _load_embedding_model(self) -> None:
        """Load sentence transformer model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Loaded embedding model: all-MiniLM-L6-v2")
        except ImportError:
            self._warnings.append(
                "sentence-transformers not installed. "
                "Run: uv add sentence-transformers"
            )
            self._use_embeddings = False
        except Exception as e:
            self._warnings.append(f"Failed to load embedding model: {e}")
            self._use_embeddings = False
    
    # =========================================================================
    # COMPLETENESS VALIDATION
    # =========================================================================
    
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
        missing_fields: Dict[str, int] = Counter()
        
        for i, conv in enumerate(conversations):
            # Check top-level fields
            for field in self.REQUIRED_CONVERSATION_FIELDS:
                total_checks += 1
                if field in conv and conv[field] is not None:
                    passed_checks += 1
                else:
                    missing_fields[field] += 1
                    if missing_fields[field] <= 3:  # Only log first 3
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
                            missing_fields[f"turn.{field}"] += 1
        
        # Store detailed metrics
        self._detailed_metrics["completeness"] = {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "missing_fields": dict(missing_fields)
        }
        
        return passed_checks / total_checks if total_checks > 0 else 0.0
    
    # =========================================================================
    # CONSISTENCY VALIDATION (Structural + Semantic Coherence)
    # =========================================================================
    
    def validate_consistency(self, conversations: List[Dict]) -> float:
        """
        Check conversational coherence (structural and semantic).
        
        Structural checks:
        - First turn is always from customer
        - Speakers alternate (customer/agent pattern)
        - At least 2 turns per conversation
        - Valid sentiment and resolution values
        - Intent is from Banking77 list
        
        Semantic checks (if embeddings enabled):
        - Agent response relevance to customer query
        - Intent alignment with conversation content
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = all coherence checks pass)
        """
        if not conversations:
            return 0.0
        
        total_checks = 0
        passed_checks = 0
        
        semantic_scores = []
        
        for i, conv in enumerate(conversations):
            turns = conv.get("turns", [])
            
            # Structural Check 1: At least 2 turns
            total_checks += 1
            if len(turns) >= 2:
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: Only {len(turns)} turns (need >=2)")
            
            # Structural Check 2: First turn from customer
            total_checks += 1
            if turns and turns[0].get("speaker") == "customer":
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: First turn not from customer")
            
            # Structural Check 3: Alternating speakers (soft check)
            total_checks += 1
            if self._check_alternating_speakers(turns):
                passed_checks += 1
            else:
                self._warnings.append(f"Conv {i}: Non-alternating speaker pattern")
            
            # Structural Check 4: Valid sentiment
            total_checks += 1
            sentiment = conv.get("sentiment", "")
            if sentiment in self.VALID_SENTIMENTS:
                passed_checks += 1
            else:
                self._issues.append(f"Conv {i}: Invalid sentiment '{sentiment}'")
            
            # Structural Check 5: Valid resolution status
            total_checks += 1
            resolution = conv.get("resolution_status", "")
            if resolution in self.VALID_RESOLUTIONS:
                passed_checks += 1
            else:
                self._warnings.append(f"Conv {i}: Invalid resolution '{resolution}'")
            
            # Structural Check 6: Valid intent (from Banking77)
            total_checks += 1
            intent = conv.get("intent", "")
            if intent in ALL_INTENTS:
                passed_checks += 1
            else:
                self._warnings.append(f"Conv {i}: Unknown intent '{intent}'")
            
            # Semantic Check: Agent relevance (if embeddings enabled)
            if self._use_embeddings and self._embedding_model and len(turns) >= 2:
                semantic_score = self._check_semantic_coherence(turns, intent)
                semantic_scores.append(semantic_score)
        
        # Calculate base structural score
        structural_score = passed_checks / total_checks if total_checks > 0 else 0.0
        
        # Blend with semantic score if available
        if semantic_scores:
            avg_semantic = sum(semantic_scores) / len(semantic_scores)
            final_score = 0.7 * structural_score + 0.3 * avg_semantic
            self._detailed_metrics["semantic_coherence"] = {
                "avg_score": avg_semantic,
                "min_score": min(semantic_scores),
                "max_score": max(semantic_scores)
            }
        else:
            final_score = structural_score
        
        return final_score
    
    def _check_alternating_speakers(self, turns: List[Dict]) -> bool:
        """Check if speakers generally alternate."""
        if len(turns) < 2:
            return True
        
        consecutive = 1
        max_consecutive = 1
        
        for i in range(1, len(turns)):
            if turns[i].get("speaker") == turns[i-1].get("speaker"):
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 1
        
        return max_consecutive <= 2
    
    def _check_semantic_coherence(
        self, 
        turns: List[Dict], 
        declared_intent: str
    ) -> float:
        """
        Check semantic coherence using embeddings.
        
        Measures:
        1. Agent response relevance to customer query
        2. Intent alignment with content
        
        Returns:
            Score 0-1 (1 = highly coherent)
        """
        if not self._embedding_model:
            return 0.5
        
        try:
            # Extract first customer and agent messages
            customer_msgs = [t["text"] for t in turns if t.get("speaker") == "customer"]
            agent_msgs = [t["text"] for t in turns if t.get("speaker") == "agent"]
            
            if not customer_msgs or not agent_msgs:
                return 0.5
            
            # Encode messages
            customer_emb = self._embedding_model.encode(customer_msgs[0])
            agent_emb = self._embedding_model.encode(agent_msgs[0])
            
            # Cosine similarity between first customer query and agent response
            from numpy import dot
            from numpy.linalg import norm
            
            similarity = dot(customer_emb, agent_emb) / (norm(customer_emb) * norm(agent_emb))
            
            # Score: similarity should be moderate to high (0.3-0.8 is typical)
            # Too low = irrelevant, too high = might be copying
            if 0.2 <= similarity <= 0.9:
                return min(1.0, similarity + 0.3)
            else:
                return max(0.3, similarity)
                
        except Exception as e:
            logger.warning(f"Semantic coherence check failed: {e}")
            return 0.5
    
    # =========================================================================
    # REALISM VALIDATION (Distribution Matching)
    # =========================================================================
    
    def validate_realism(self, conversations: List[Dict]) -> float:
        """
        Compare intent distribution vs banking77 reference.
        
        Uses Jensen-Shannon divergence to measure distribution similarity.
        Also checks category coverage and turn length distribution.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Score 0-1 (1 = perfect distribution match)
        """
        if not conversations:
            return 0.0
        
        if not self._reference_intent_dist:
            self._warnings.append("No reference data for realism check")
            return 0.5
        
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
        distribution_score = 1.0 - js_div
        
        # Check category coverage (11 categories in Banking77)
        synthetic_categories = set(
            INTENT_TO_CATEGORY.get(conv.get("intent", ""), "unknown")
            for conv in conversations
        )
        total_categories = len(BANKING77_INTENTS)
        category_coverage = len(synthetic_categories) / total_categories
        
        # Check intent coverage
        reference_intents_used = len(set(synthetic_dist.keys()) & set(self._reference_intent_dist.keys()))
        intent_coverage = reference_intents_used / len(self._reference_intent_dist)
        
        # Check turn length realism (typically 4-8 turns)
        turn_lengths = [len(conv.get("turns", [])) for conv in conversations]
        avg_turns = sum(turn_lengths) / len(turn_lengths) if turn_lengths else 0
        # Realistic range is 4-8 turns on average
        turn_realism = 1.0 if 3 <= avg_turns <= 10 else max(0.5, 1 - abs(avg_turns - 6) / 10)
        
        # Store detailed metrics
        self._detailed_metrics["realism"] = {
            "js_divergence": js_div,
            "distribution_score": distribution_score,
            "category_coverage": category_coverage,
            "intent_coverage": intent_coverage,
            "avg_turn_length": avg_turns,
            "categories_covered": list(synthetic_categories)
        }
        
        # Weighted final score
        final_score = (
            0.40 * distribution_score +
            0.25 * intent_coverage +
            0.20 * category_coverage +
            0.15 * turn_realism
        )
        
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
        eps = 1e-10
        
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
        kl_pm = sum(
            p_aligned[i] * math.log2(p_aligned[i] / m[i]) 
            for i in range(len(p_aligned)) if p_aligned[i] > eps
        )
        kl_qm = sum(
            q_aligned[i] * math.log2(q_aligned[i] / m[i]) 
            for i in range(len(q_aligned)) if q_aligned[i] > eps
        )
        
        js = (kl_pm + kl_qm) / 2
        return min(1.0, js)
    
    # =========================================================================
    # DIVERSITY VALIDATION
    # =========================================================================
    
    def validate_diversity(self, conversations: List[Dict]) -> float:
        """
        Check variety in generated data with comprehensive metrics.
        
        Metrics:
        - Intent variety: unique intents / available intents
        - Message variety: unique first messages / total
        - Vocabulary richness: unique words / total words
        - N-gram diversity: unique bigrams and trigrams
        - Sentiment variety: coverage of all sentiments
        - Resolution variety: coverage of resolution statuses
        
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
                msg = turns[0].get("text", "").lower().strip()
                # Normalize: remove punctuation for comparison
                msg_normalized = re.sub(r'[^\w\s]', '', msg)
                first_messages.append(msg_normalized)
        
        unique_first_messages = len(set(first_messages))
        message_variety = unique_first_messages / len(conversations) if conversations else 0
        
        # 3. Vocabulary diversity
        all_customer_words = []
        all_agent_words = []
        for conv in conversations:
            for turn in conv.get("turns", []):
                text = turn.get("text", "").lower()
                words = re.findall(r'\b\w+\b', text)
                if turn.get("speaker") == "customer":
                    all_customer_words.extend(words)
                else:
                    all_agent_words.extend(words)
        
        customer_vocab_diversity = (
            len(set(all_customer_words)) / len(all_customer_words) 
            if all_customer_words else 0
        )
        agent_vocab_diversity = (
            len(set(all_agent_words)) / len(all_agent_words)
            if all_agent_words else 0
        )
        vocab_diversity = (customer_vocab_diversity + agent_vocab_diversity) / 2
        
        # 4. N-gram diversity (bigrams and trigrams)
        bigram_diversity = self._calculate_ngram_diversity(all_customer_words, 2)
        trigram_diversity = self._calculate_ngram_diversity(all_customer_words, 3)
        ngram_score = (bigram_diversity + trigram_diversity) / 2
        
        # 5. Sentiment variety
        sentiments = set(conv.get("sentiment", "") for conv in conversations)
        sentiment_variety = len(sentiments & self.VALID_SENTIMENTS) / len(self.VALID_SENTIMENTS)
        
        # 6. Resolution variety
        resolutions = set(conv.get("resolution_status", "") for conv in conversations)
        resolution_variety = len(resolutions & self.VALID_RESOLUTIONS) / len(self.VALID_RESOLUTIONS)
        
        # 7. Category coverage
        categories = set(
            INTENT_TO_CATEGORY.get(conv.get("intent", ""), "unknown")
            for conv in conversations
        )
        category_variety = len(categories) / len(BANKING77_INTENTS)
        
        # Store detailed metrics
        self._detailed_metrics["diversity"] = {
            "unique_intents": len(unique_intents),
            "unique_first_messages": unique_first_messages,
            "customer_vocab_size": len(set(all_customer_words)),
            "agent_vocab_size": len(set(all_agent_words)),
            "bigram_diversity": bigram_diversity,
            "trigram_diversity": trigram_diversity,
            "sentiments_covered": list(sentiments),
            "resolutions_covered": list(resolutions),
            "categories_covered": list(categories)
        }
        
        # Weighted average
        diversity_score = (
            0.20 * intent_variety +
            0.15 * message_variety +
            0.15 * vocab_diversity +
            0.15 * ngram_score +
            0.15 * sentiment_variety +
            0.10 * resolution_variety +
            0.10 * category_variety
        )
        
        # Warnings for low diversity
        if len(unique_intents) < 10 and len(conversations) >= 50:
            self._warnings.append(f"Low intent variety: only {len(unique_intents)} unique intents")
        if message_variety < 0.5:
            self._warnings.append(f"Low message variety: {message_variety:.1%} unique messages")
        
        return min(1.0, max(0.0, diversity_score))
    
    def _calculate_ngram_diversity(self, words: List[str], n: int) -> float:
        """
        Calculate n-gram diversity score.
        
        Args:
            words: List of words
            n: N-gram size (2 for bigrams, 3 for trigrams)
        
        Returns:
            Score 0-1 (ratio of unique n-grams to total n-grams)
        """
        if len(words) < n:
            return 0.0
        
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        
        if not ngrams:
            return 0.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        # Normalize: for large datasets, even 10% unique is good
        # For small datasets, we expect higher uniqueness
        expected_unique = min(0.3, 1.0 / (len(words) ** 0.3))
        diversity = unique_ngrams / total_ngrams
        
        # Scale to 0-1 where expected_unique = 0.5
        return min(1.0, diversity / (expected_unique * 2))
    
    # =========================================================================
    # OVERALL SCORING
    # =========================================================================
    
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
        # Reset for this run
        self._issues = []
        self._warnings = []
        self._detailed_metrics = {}
        
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
                "has_reference": self._reference_data is not None,
                "embeddings_enabled": self._use_embeddings,
            }
        )
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed metrics from last validation run."""
        return self._detailed_metrics
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def print_report(self, metrics: QualityMetrics) -> None:
        """Print a formatted quality report."""
        print("\n" + "=" * 70)
        print("  QUALITY VALIDATION REPORT")
        print("=" * 70)
        
        num_convs = metrics.metadata.get("num_conversations", 0)
        print(f"\n  Dataset: {num_convs} conversations")
        print(f"  Reference: {metrics.metadata.get('reference_data', 'N/A')}")
        
        print("\n" + "-" * 50)
        print("  SCORES")
        print("-" * 50)
        
        def score_bar(score: float, width: int = 25) -> str:
            filled = int(score * width)
            bar = "#" * filled + "-" * (width - filled)
            return f"[{bar}]"
        
        print(f"  Completeness:  {metrics.completeness_score:.2f} {score_bar(metrics.completeness_score)}")
        print(f"  Consistency:   {metrics.consistency_score:.2f} {score_bar(metrics.consistency_score)}")
        print(f"  Realism:       {metrics.realism_score:.2f} {score_bar(metrics.realism_score)}")
        print(f"  Diversity:     {metrics.diversity_score:.2f} {score_bar(metrics.diversity_score)}")
        
        print("\n" + "-" * 50)
        overall_bar = score_bar(metrics.overall_quality_score / 100, 35)
        print(f"  OVERALL: {metrics.overall_quality_score:.1f}/100 {overall_bar}")
        print("-" * 50)
        
        # Quality threshold
        if metrics.overall_quality_score >= 85:
            status = "[PASS]"
            msg = "Quality meets production threshold (>=85)"
        elif metrics.overall_quality_score >= 70:
            status = "[WARN]"
            msg = "Quality is acceptable but could improve (70-85)"
        else:
            status = "[FAIL]"
            msg = "Quality below acceptable threshold (<70)"
        
        print(f"\n  {status} {msg}")
        
        # Detailed metrics (from validator internal state)
        detailed = self._detailed_metrics
        if detailed:
            print("\n" + "-" * 50)
            print("  DETAILED METRICS")
            print("-" * 50)
            
            if "realism" in detailed:
                r = detailed["realism"]
                print(f"  - JS Divergence: {r.get('js_divergence', 0):.3f}")
                print(f"  - Intent Coverage: {r.get('intent_coverage', 0):.1%}")
                print(f"  - Category Coverage: {r.get('category_coverage', 0):.1%}")
                print(f"  - Avg Turn Length: {r.get('avg_turn_length', 0):.1f}")
            
            if "diversity" in detailed:
                d = detailed["diversity"]
                print(f"  - Unique Intents: {d.get('unique_intents', 0)}")
                print(f"  - Unique First Messages: {d.get('unique_first_messages', 0)}")
                print(f"  - Customer Vocab Size: {d.get('customer_vocab_size', 0)}")
        
        # Issues
        if metrics.issues_found:
            print(f"\n  Issues ({len(metrics.issues_found)}):")
            for issue in metrics.issues_found[:5]:
                print(f"    - {issue}")
            if len(metrics.issues_found) > 5:
                print(f"    ... and {len(metrics.issues_found) - 5} more")
        
        # Warnings
        if metrics.warnings:
            print(f"\n  Warnings ({len(metrics.warnings)}):")
            for warning in metrics.warnings[:5]:
                print(f"    ! {warning}")
            if len(metrics.warnings) > 5:
                print(f"    ... and {len(metrics.warnings) - 5} more")
        
        print("\n" + "=" * 70)


def main():
    """Run quality validation on synthetic data."""
    import sys
    
    # Try different synthetic data paths
    synthetic_paths = [
        Path("data/synthetic/customer_service_100.json"),
        Path("data/synthetic/cs_smoke_v2.json"),
    ]
    
    synthetic_path = None
    for path in synthetic_paths:
        if path.exists():
            synthetic_path = path
            break
    
    if not synthetic_path:
        print("[ERROR] No synthetic data found. Run generation first.")
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
