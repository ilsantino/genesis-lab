"""
Bias detection module for synthetic data.

Detects various biases in generated datasets including:
- Sentiment distribution imbalance (with TextBlob NLP analysis)
- Intent coverage gaps
- Language balance (EN/ES)
- Complexity distribution
- Underrepresented intent detection

Usage:
    uv run python -m src.validation.bias
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

from src.generation.schemas import BiasMetrics
from src.generation.templates.customer_service_prompts import ALL_INTENTS

# TextBlob for real NLP sentiment analysis
try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False


__all__ = ["BiasDetector", "BiasReport"]


@dataclass
class BiasReport:
    """
    Dataclass for bias detection results.
    
    Alternative to Pydantic BiasMetrics for simpler use cases.
    """
    sentiment_distribution: Dict[str, float] = field(default_factory=dict)
    intent_coverage: float = 0.0
    language_balance: Dict[str, float] = field(default_factory=dict)
    complexity_distribution: Dict[str, float] = field(default_factory=dict)
    underrepresented_intents: List[str] = field(default_factory=list)
    bias_detected: bool = False
    bias_severity: str = "none"
    recommendations: List[str] = field(default_factory=list)
    num_conversations: int = 0


class BiasDetector:
    """
    Detects bias in synthetic customer service datasets.
    
    Analyzes distribution of sentiments, intents, languages, and complexity
    to identify potential biases that could affect model training.
    
    Example:
        >>> detector = BiasDetector()
        >>> conversations = json.load(open("data/synthetic/cs_smoke_v2.json"))
        >>> metrics = detector.detect_bias(conversations)
        >>> print(f"Bias detected: {metrics.bias_detected}")
    """
    
    # Expected distributions (based on realistic customer service data)
    EXPECTED_SENTIMENT = {"positive": 0.30, "neutral": 0.50, "negative": 0.20}
    EXPECTED_COMPLEXITY = {"simple": 0.30, "medium": 0.50, "complex": 0.20}
    EXPECTED_LANGUAGE = {"en": 0.50, "es": 0.50}  # Balanced bilingual
    
    # Thresholds for bias detection
    SENTIMENT_IMBALANCE_THRESHOLD = 0.30  # Max deviation from expected
    INTENT_COVERAGE_THRESHOLD = 0.10  # Minimum coverage of 77 intents
    LANGUAGE_BALANCE_THRESHOLD = 0.40  # Minimum for each language
    COMPLEXITY_IMBALANCE_THRESHOLD = 0.30
    
    def __init__(self, expected_intents: int = 77, use_textblob: bool = True):
        """
        Initialize bias detector.
        
        Args:
            expected_intents: Number of expected unique intents (Banking77 = 77)
            use_textblob: Whether to use TextBlob for real NLP sentiment analysis
        """
        self._expected_intents = expected_intents
        self._recommendations: List[str] = []
        self._use_textblob = use_textblob and TEXTBLOB_AVAILABLE
        
        if use_textblob and not TEXTBLOB_AVAILABLE:
            import warnings
            warnings.warn(
                "TextBlob not available. Install with: uv add textblob. "
                "Falling back to metadata-based sentiment analysis."
            )
    
    def _analyze_sentiment_from_text(self, text: str) -> str:
        """
        Analyze sentiment from actual text using TextBlob.
        
        Uses TextBlob's polarity score:
        - polarity > 0.1: positive
        - polarity < -0.1: negative
        - otherwise: neutral
        
        Args:
            text: The text to analyze
        
        Returns:
            Sentiment label: "positive", "neutral", or "negative"
        """
        if not self._use_textblob or not text:
            return "neutral"
        
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            if polarity > 0.1:
                return "positive"
            elif polarity < -0.1:
                return "negative"
            else:
                return "neutral"
        except Exception:
            return "neutral"
    
    def _find_underrepresented_intents(
        self, 
        conversations: List[Dict], 
        threshold: float = 0.01
    ) -> List[str]:
        """
        Find intents that are underrepresented in the dataset.
        
        An intent is underrepresented if it appears in less than threshold
        percent of the examples (default: 1%).
        
        Args:
            conversations: List of conversation dictionaries
            threshold: Minimum representation threshold (default: 0.01 = 1%)
        
        Returns:
            List of underrepresented intent names
        """
        if not conversations:
            return []
        
        # Count intent occurrences
        intent_counts = Counter(
            conv.get("intent", "unknown") 
            for conv in conversations
        )
        
        total = len(conversations)
        underrepresented = []
        
        # Check each Banking77 intent
        for intent in ALL_INTENTS:
            count = intent_counts.get(intent, 0)
            ratio = count / total
            
            if ratio < threshold:
                underrepresented.append(intent)
        
        return underrepresented
    
    def check_sentiment_distribution(
        self, 
        conversations: List[Dict],
        use_nlp: bool = False
    ) -> Dict[str, float]:
        """
        Analyze sentiment distribution.
        
        Expected distribution: ~30% positive, ~50% neutral, ~20% negative
        
        Args:
            conversations: List of conversation dictionaries
            use_nlp: If True, use TextBlob to analyze actual text instead of metadata
        
        Returns:
            Dict with sentiment percentages and imbalance score
        """
        if not conversations:
            return {"positive": 0, "neutral": 0, "negative": 0, "imbalance": 1.0}
        
        # Determine sentiment source
        if use_nlp and self._use_textblob:
            # Use TextBlob on actual conversation text
            sentiments = []
            for conv in conversations:
                # Extract all customer text for sentiment analysis
                customer_text = " ".join(
                    turn.get("text", "")
                    for turn in conv.get("turns", [])
                    if turn.get("speaker") == "customer"
                )
                sentiments.append(self._analyze_sentiment_from_text(customer_text))
            sentiment_counts = Counter(sentiments)
        else:
            # Use metadata field
            sentiment_counts = Counter(
                conv.get("sentiment", "unknown").lower() 
                for conv in conversations
            )
        
        total = len(conversations)
        
        # Calculate distribution
        distribution = {
            "positive": sentiment_counts.get("positive", 0) / total,
            "neutral": sentiment_counts.get("neutral", 0) / total,
            "negative": sentiment_counts.get("negative", 0) / total,
        }
        
        # Calculate imbalance (max deviation from expected)
        deviations = [
            abs(distribution["positive"] - self.EXPECTED_SENTIMENT["positive"]),
            abs(distribution["neutral"] - self.EXPECTED_SENTIMENT["neutral"]),
            abs(distribution["negative"] - self.EXPECTED_SENTIMENT["negative"]),
        ]
        
        distribution["imbalance"] = max(deviations)
        
        return distribution
    
    def check_sentiment_distribution_nlp(self, conversations: List[Dict]) -> Dict[str, float]:
        """
        Analyze sentiment using TextBlob NLP on actual conversation text.
        
        This method uses real NLP analysis instead of relying on metadata.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Dict with sentiment percentages and imbalance score
        """
        return self.check_sentiment_distribution(conversations, use_nlp=True)
    
    def check_intent_coverage(self, conversations: List[Dict]) -> float:
        """
        Check coverage of Banking77 intents.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Coverage score (unique intents used / 77)
        """
        if not conversations:
            return 0.0
        
        # Get unique intents
        unique_intents = set(
            conv.get("intent", "unknown") 
            for conv in conversations
        )
        
        # Filter to only valid Banking77 intents
        valid_intents = unique_intents & set(ALL_INTENTS)
        
        coverage = len(valid_intents) / self._expected_intents
        
        return min(1.0, coverage)
    
    def check_language_balance(self, conversations: List[Dict]) -> Dict[str, Any]:
        """
        Check EN/ES language balance.
        
        Expected: 40-60% each for balanced bilingual dataset.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Dict with language percentages and balance status
        """
        if not conversations:
            return {"en": 0, "es": 0, "other": 0, "balanced": False}
        
        # Count languages
        lang_counts = Counter(
            conv.get("language", "unknown").lower() 
            for conv in conversations
        )
        
        total = len(conversations)
        
        en_ratio = lang_counts.get("en", 0) / total
        es_ratio = lang_counts.get("es", 0) / total
        other_ratio = 1.0 - en_ratio - es_ratio
        
        # Check if balanced (each language >= 40%)
        is_balanced = (
            en_ratio >= self.LANGUAGE_BALANCE_THRESHOLD and
            es_ratio >= self.LANGUAGE_BALANCE_THRESHOLD
        )
        
        return {
            "en": en_ratio,
            "es": es_ratio,
            "other": other_ratio,
            "balanced": is_balanced
        }
    
    def check_complexity_distribution(self, conversations: List[Dict]) -> Dict[str, float]:
        """
        Analyze complexity distribution.
        
        Expected: ~30% simple, ~50% medium, ~20% complex
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Dict with complexity percentages and imbalance score
        """
        if not conversations:
            return {"simple": 0, "medium": 0, "complex": 0, "imbalance": 1.0}
        
        # Count complexities
        complexity_counts = Counter(
            conv.get("complexity", "unknown").lower() 
            for conv in conversations
        )
        
        total = len(conversations)
        
        distribution = {
            "simple": complexity_counts.get("simple", 0) / total,
            "medium": complexity_counts.get("medium", 0) / total,
            "complex": complexity_counts.get("complex", 0) / total,
        }
        
        # Calculate imbalance
        deviations = [
            abs(distribution["simple"] - self.EXPECTED_COMPLEXITY["simple"]),
            abs(distribution["medium"] - self.EXPECTED_COMPLEXITY["medium"]),
            abs(distribution["complex"] - self.EXPECTED_COMPLEXITY["complex"]),
        ]
        
        distribution["imbalance"] = max(deviations)
        
        return distribution
    
    def check_resolution_distribution(self, conversations: List[Dict]) -> Dict[str, float]:
        """
        Analyze resolution status distribution.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Dict with resolution status percentages
        """
        if not conversations:
            return {}
        
        resolution_counts = Counter(
            conv.get("resolution_status", "unknown").lower()
            for conv in conversations
        )
        
        total = len(conversations)
        
        return {
            status: count / total 
            for status, count in resolution_counts.items()
        }
    
    def _determine_severity(
        self,
        sentiment_imbalance: float,
        intent_coverage: float,
        language_balanced: bool,
        complexity_imbalance: float
    ) -> Literal["none", "low", "medium", "high"]:
        """Determine overall bias severity."""
        issues = 0
        
        if sentiment_imbalance > self.SENTIMENT_IMBALANCE_THRESHOLD:
            issues += 1
        if intent_coverage < self.INTENT_COVERAGE_THRESHOLD:
            issues += 1
        if not language_balanced:
            issues += 1
        if complexity_imbalance > self.COMPLEXITY_IMBALANCE_THRESHOLD:
            issues += 1
        
        if issues == 0:
            return "none"
        elif issues == 1:
            return "low"
        elif issues == 2:
            return "medium"
        else:
            return "high"
    
    def detect_bias(
        self, 
        conversations: List[Dict],
        use_nlp_sentiment: bool = False
    ) -> BiasMetrics:
        """
        Run all bias checks and return BiasMetrics.
        
        Checks sentiment, intent coverage, language balance, and complexity.
        Sets bias_detected=True if any check exceeds threshold.
        
        Args:
            conversations: List of conversation dictionaries
            use_nlp_sentiment: If True, use TextBlob for sentiment analysis
        
        Returns:
            BiasMetrics with all bias information and recommendations
        """
        self._recommendations = []
        
        # Run all checks
        sentiment_dist = self.check_sentiment_distribution(
            conversations, 
            use_nlp=use_nlp_sentiment
        )
        intent_coverage = self.check_intent_coverage(conversations)
        language_balance = self.check_language_balance(conversations)
        complexity_dist = self.check_complexity_distribution(conversations)
        resolution_dist = self.check_resolution_distribution(conversations)
        
        # Find underrepresented intents (< 1%)
        underrepresented = self._find_underrepresented_intents(conversations)
        
        # Analyze and generate recommendations
        bias_detected = False
        
        # Sentiment check
        if sentiment_dist["imbalance"] > self.SENTIMENT_IMBALANCE_THRESHOLD:
            bias_detected = True
            dominant = max(["positive", "neutral", "negative"], key=lambda s: sentiment_dist[s])
            self._recommendations.append(
                f"Sentiment imbalance detected: '{dominant}' is overrepresented "
                f"({sentiment_dist[dominant]*100:.0f}%). Generate more varied sentiments."
            )
        
        # Intent coverage check
        if intent_coverage < self.INTENT_COVERAGE_THRESHOLD:
            bias_detected = True
            self._recommendations.append(
                f"Low intent coverage: only {intent_coverage*100:.1f}% of 77 intents covered. "
                f"Generate more diverse intents to improve model generalization."
            )
        elif intent_coverage < 0.50:
            self._recommendations.append(
                f"Intent coverage is {intent_coverage*100:.1f}%. Consider generating more "
                f"samples to cover underrepresented intents."
            )
        
        # Underrepresented intents check
        if underrepresented and len(underrepresented) > 50:
            self._recommendations.append(
                f"Found {len(underrepresented)} underrepresented intents (< 1% each). "
                f"Consider targeted generation for: {', '.join(underrepresented[:5])}..."
            )
        
        # Language balance check
        if not language_balance["balanced"]:
            bias_detected = True
            if language_balance["en"] > language_balance["es"]:
                self._recommendations.append(
                    f"Language imbalance: EN={language_balance['en']*100:.0f}%, "
                    f"ES={language_balance['es']*100:.0f}%. Generate more Spanish conversations."
                )
            else:
                self._recommendations.append(
                    f"Language imbalance: EN={language_balance['en']*100:.0f}%, "
                    f"ES={language_balance['es']*100:.0f}%. Generate more English conversations."
                )
        
        # Complexity check
        if complexity_dist["imbalance"] > self.COMPLEXITY_IMBALANCE_THRESHOLD:
            bias_detected = True
            dominant = max(["simple", "medium", "complex"], key=lambda c: complexity_dist[c])
            self._recommendations.append(
                f"Complexity imbalance: '{dominant}' is overrepresented "
                f"({complexity_dist[dominant]*100:.0f}%). Balance complexity levels."
            )
        
        # Determine severity
        severity = self._determine_severity(
            sentiment_dist["imbalance"],
            intent_coverage,
            language_balance["balanced"],
            complexity_dist["imbalance"]
        )
        
        # Build topic coverage from intent distribution
        intent_counts = Counter(conv.get("intent", "unknown") for conv in conversations)
        total = len(conversations) if conversations else 1
        topic_coverage = {
            intent: count / total 
            for intent, count in intent_counts.most_common(10)  # Top 10
        }
        
        return BiasMetrics(
            demographic_balance={
                "language": language_balance,
                "complexity": complexity_dist,
                "resolution": resolution_dist,
                "underrepresented_intents": underrepresented[:10]  # Top 10 most underrepresented
            },
            sentiment_distribution={
                "positive": sentiment_dist["positive"],
                "neutral": sentiment_dist["neutral"],
                "negative": sentiment_dist["negative"]
            },
            topic_coverage=topic_coverage,
            bias_detected=bias_detected,
            bias_severity=severity,
            recommendations=self._recommendations.copy(),
            metadata={
                "num_conversations": len(conversations),
                "unique_intents": int(intent_coverage * self._expected_intents),
                "intent_coverage_pct": intent_coverage * 100,
                "sentiment_imbalance": sentiment_dist["imbalance"],
                "complexity_imbalance": complexity_dist["imbalance"],
                "underrepresented_count": len(underrepresented),
                "nlp_sentiment_used": use_nlp_sentiment and self._use_textblob
            }
        )
    
    def detect_bias_report(
        self, 
        conversations: List[Dict],
        use_nlp_sentiment: bool = False
    ) -> BiasReport:
        """
        Run all bias checks and return a BiasReport dataclass.
        
        Alternative to detect_bias() that returns a simpler dataclass.
        
        Args:
            conversations: List of conversation dictionaries
            use_nlp_sentiment: If True, use TextBlob for sentiment analysis
        
        Returns:
            BiasReport dataclass with all bias information
        """
        metrics = self.detect_bias(conversations, use_nlp_sentiment)
        
        return BiasReport(
            sentiment_distribution=dict(metrics.sentiment_distribution),
            intent_coverage=metrics.metadata.get("intent_coverage_pct", 0) / 100,
            language_balance={
                "en": metrics.demographic_balance.get("language", {}).get("en", 0),
                "es": metrics.demographic_balance.get("language", {}).get("es", 0)
            },
            complexity_distribution={
                "simple": metrics.demographic_balance.get("complexity", {}).get("simple", 0),
                "medium": metrics.demographic_balance.get("complexity", {}).get("medium", 0),
                "complex": metrics.demographic_balance.get("complexity", {}).get("complex", 0)
            },
            underrepresented_intents=metrics.demographic_balance.get("underrepresented_intents", []),
            bias_detected=metrics.bias_detected,
            bias_severity=metrics.bias_severity,
            recommendations=metrics.recommendations,
            num_conversations=metrics.metadata.get("num_conversations", 0)
        )
    
    def print_report(self, metrics: BiasMetrics) -> None:
        """Print a formatted bias detection report."""
        print("\n" + "=" * 60)
        print("BIAS DETECTION REPORT")
        print("=" * 60)
        
        num_conv = metrics.metadata.get("num_conversations", 0)
        print(f"\nDataset: {num_conv} conversations")
        
        # Overall status
        print("\n" + "-" * 40)
        if metrics.bias_detected:
            severity_icon = {"low": "[!]", "medium": "[!!]", "high": "[!!!]"}.get(metrics.bias_severity, "[?]")
            print(f"  BIAS DETECTED {severity_icon} (Severity: {metrics.bias_severity.upper()})")
        else:
            print("  [OK] No significant bias detected")
        print("-" * 40)
        
        # Sentiment Distribution
        print("\n  SENTIMENT DISTRIBUTION")
        print("  " + "-" * 35)
        sent = metrics.sentiment_distribution
        imbalance = metrics.metadata.get("sentiment_imbalance", 0)
        print(f"    Positive: {sent.get('positive', 0)*100:5.1f}% (expected: 30%)")
        print(f"    Neutral:  {sent.get('neutral', 0)*100:5.1f}% (expected: 50%)")
        print(f"    Negative: {sent.get('negative', 0)*100:5.1f}% (expected: 20%)")
        status = "[OK]" if imbalance <= self.SENTIMENT_IMBALANCE_THRESHOLD else "[BIAS]"
        print(f"    Imbalance: {imbalance*100:.1f}% {status}")
        
        # Intent Coverage
        print("\n  INTENT COVERAGE")
        print("  " + "-" * 35)
        coverage = metrics.metadata.get("intent_coverage_pct", 0)
        unique = metrics.metadata.get("unique_intents", 0)
        status = "[OK]" if coverage >= self.INTENT_COVERAGE_THRESHOLD * 100 else "[LOW]"
        print(f"    Unique intents: {unique}/77 ({coverage:.1f}%) {status}")
        
        # Top intents
        if metrics.topic_coverage:
            print("    Top intents:")
            for intent, pct in list(metrics.topic_coverage.items())[:5]:
                print(f"      - {intent}: {pct*100:.1f}%")
        
        # Language Balance
        print("\n  LANGUAGE BALANCE")
        print("  " + "-" * 35)
        lang = metrics.demographic_balance.get("language", {})
        en_pct = lang.get("en", 0) * 100
        es_pct = lang.get("es", 0) * 100
        balanced = lang.get("balanced", False)
        status = "[OK]" if balanced else "[IMBALANCED]"
        print(f"    English: {en_pct:5.1f}%")
        print(f"    Spanish: {es_pct:5.1f}%")
        print(f"    Status: {status}")
        
        # Complexity Distribution
        print("\n  COMPLEXITY DISTRIBUTION")
        print("  " + "-" * 35)
        comp = metrics.demographic_balance.get("complexity", {})
        comp_imbalance = metrics.metadata.get("complexity_imbalance", 0)
        print(f"    Simple:  {comp.get('simple', 0)*100:5.1f}% (expected: 30%)")
        print(f"    Medium:  {comp.get('medium', 0)*100:5.1f}% (expected: 50%)")
        print(f"    Complex: {comp.get('complex', 0)*100:5.1f}% (expected: 20%)")
        status = "[OK]" if comp_imbalance <= self.COMPLEXITY_IMBALANCE_THRESHOLD else "[BIAS]"
        print(f"    Imbalance: {comp_imbalance*100:.1f}% {status}")
        
        # Recommendations
        if metrics.recommendations:
            print("\n  RECOMMENDATIONS")
            print("  " + "-" * 35)
            for i, rec in enumerate(metrics.recommendations, 1):
                # Wrap long recommendations
                words = rec.split()
                lines = []
                current_line = []
                for word in words:
                    if len(" ".join(current_line + [word])) > 50:
                        lines.append(" ".join(current_line))
                        current_line = [word]
                    else:
                        current_line.append(word)
                if current_line:
                    lines.append(" ".join(current_line))
                
                print(f"    {i}. {lines[0]}")
                for line in lines[1:]:
                    print(f"       {line}")
        
        print("\n" + "=" * 60)


def main():
    """Run bias detection on synthetic data."""
    # Try v2 first, fall back to original
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
        print("[ERROR] No synthetic data found. Run smoke_test.py first.")
        print("  Tried:", [str(p) for p in synthetic_paths])
        return 1
    
    print(f"Loading synthetic data from: {synthetic_path}")
    
    with open(synthetic_path, "r", encoding="utf-8") as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Create detector and run
    detector = BiasDetector()
    metrics = detector.detect_bias(conversations)
    
    # Print report
    detector.print_report(metrics)
    
    # Return exit code based on bias severity
    if metrics.bias_severity in ["high", "medium"]:
        return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


