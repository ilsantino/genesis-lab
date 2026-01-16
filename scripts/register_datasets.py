"""Register smoke test datasets in the registry."""

import json
from pathlib import Path

from src.registry import DatasetRegistry
from src.validation import QualityValidator, BiasDetector


def main():
    print("=" * 50)
    print("Registering Smoke Test Datasets")
    print("=" * 50)
    
    # Initialize registry
    registry = DatasetRegistry()
    
    # Load CS data
    cs_path = Path("data/synthetic/cs_smoke_v2.json")
    if cs_path.exists():
        with open(cs_path, "r", encoding="utf-8") as f:
            cs_data = json.load(f)
        
        # Register CS dataset
        cs_id = registry.register_dataset(
            domain="customer_service",
            size=len(cs_data),
            file_path=str(cs_path),
            file_format="json",
            notes="Bilingual smoke test v2 (5 EN + 5 ES)"
        )
        print(f"\n[OK] Registered CS dataset: {cs_id}")
        
        # Add quality metrics
        validator = QualityValidator()
        quality = validator.compute_overall_score(cs_data)
        registry.update_quality_metrics(cs_id, quality)
        print(f"     Quality score: {quality.overall_quality_score:.1f}/100")
        
        # Add bias metrics
        detector = BiasDetector()
        bias = detector.detect_bias(cs_data)
        registry.update_bias_metrics(cs_id, bias)
        print(f"     Bias detected: {bias.bias_detected}")
    else:
        print(f"[SKIP] CS data not found: {cs_path}")
    
    # Load TS data
    ts_path = Path("data/synthetic/ts_smoke_v2.json")
    if ts_path.exists():
        with open(ts_path, "r", encoding="utf-8") as f:
            ts_data = json.load(f)
        
        # Register TS dataset
        ts_id = registry.register_dataset(
            domain="time_series",
            size=len(ts_data),
            file_path=str(ts_path),
            file_format="json",
            notes="Time series smoke test v2"
        )
        print(f"\n[OK] Registered TS dataset: {ts_id}")
    else:
        print(f"[SKIP] TS data not found: {ts_path}")
    
    # Print summary
    registry.print_summary()
    
    print("\n[OK] Registration complete!")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


