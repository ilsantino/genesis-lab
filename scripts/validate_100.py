"""Validate the 100-item customer service dataset."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.quality import QualityValidator
from src.validation.bias import BiasDetector


def main():
    data_path = Path("data/synthetic/customer_service_100.json")
    
    if not data_path.exists():
        print(f"Error: {data_path} not found")
        return 1
    
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    print(f"Dataset: {len(data)} conversations")
    print("=" * 60)
    
    # Quality validation
    print("\nQUALITY VALIDATION")
    print("-" * 40)
    
    validator = QualityValidator()
    quality = validator.compute_overall_score(data)
    
    def bar(score, width=20):
        filled = int(score * width)
        return "#" * filled + "-" * (width - filled)
    
    print(f"  Completeness: {quality.completeness_score:.2f} [{bar(quality.completeness_score)}]")
    print(f"  Consistency:  {quality.consistency_score:.2f} [{bar(quality.consistency_score)}]")
    print(f"  Realism:      {quality.realism_score:.2f} [{bar(quality.realism_score)}]")
    print(f"  Diversity:    {quality.diversity_score:.2f} [{bar(quality.diversity_score)}]")
    print("-" * 40)
    print(f"  OVERALL: {quality.overall_quality_score:.1f}/100")
    
    if quality.issues_found:
        print(f"\n  Issues: {quality.issues_found}")
    
    # Bias detection
    print("\n" + "=" * 60)
    print("BIAS DETECTION")
    print("-" * 40)
    
    detector = BiasDetector()
    bias = detector.detect_bias(data)
    
    status = "[OK]" if not bias.bias_detected else f"[ALERT] Severity: {bias.bias_severity}"
    print(f"  Status: {status}")
    
    print("\n  SENTIMENT DISTRIBUTION")
    sent = bias.sentiment_distribution
    print(f"    Positive: {sent.get('positive', 0)*100:5.1f}% (target: 30%)")
    print(f"    Neutral:  {sent.get('neutral', 0)*100:5.1f}% (target: 50%)")
    print(f"    Negative: {sent.get('negative', 0)*100:5.1f}% (target: 20%)")
    
    print("\n  INTENT COVERAGE")
    coverage = bias.metadata.get("intent_coverage_pct", 0) / 100
    unique_intents = bias.metadata.get("unique_intents", 0)
    print(f"    Unique intents: {unique_intents}/77 ({coverage*100:.0f}%)")
    print("    Top 5 intents:")
    for intent, pct in list(bias.topic_coverage.items())[:5]:
        print(f"      - {intent}: {pct*100:.1f}%")
    
    print("\n  LANGUAGE BALANCE")
    lang = bias.demographic_balance.get("language", {})
    print(f"    English: {lang.get('en', 0)*100:.1f}%")
    print(f"    Spanish: {lang.get('es', 0)*100:.1f}%")
    bal_status = "[OK]" if lang.get("balanced", False) else "[IMBALANCED]"
    print(f"    Status:  {bal_status}")
    
    print("\n  COMPLEXITY DISTRIBUTION")
    comp = bias.demographic_balance.get("complexity", {})
    print(f"    Simple:  {comp.get('simple', 0)*100:5.1f}% (target: 30%)")
    print(f"    Medium:  {comp.get('medium', 0)*100:5.1f}% (target: 50%)")
    print(f"    Complex: {comp.get('complex', 0)*100:5.1f}% (target: 20%)")
    
    if bias.recommendations:
        print("\n  RECOMMENDATIONS")
        for i, rec in enumerate(bias.recommendations, 1):
            print(f"    {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Quality Score: {quality.overall_quality_score:.1f}/100")
    print(f"  Bias Detected: {bias.bias_detected} (severity: {bias.bias_severity})")
    print(f"  Intent Coverage: {unique_intents}/77 ({coverage*100:.0f}%)")
    print(f"  Language: {lang.get('en', 0)*100:.0f}% EN / {lang.get('es', 0)*100:.0f}% ES")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

