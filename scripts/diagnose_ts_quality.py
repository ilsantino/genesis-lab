"""
Diagnostic script: Compare synthetic vs reference time series stationarity.

This script validates whether low stationarity scores in synthetic data
are expected (matching reference) or a problem to fix.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.validation.timeseries_quality import TimeSeriesValidator


def main():
    print("=" * 60)
    print("COMPARISON: SYNTHETIC VS REFERENCE TIME SERIES")
    print("=" * 60)
    
    # Load synthetic data
    synthetic_path = Path("data/synthetic/ts_smoke_test.json")
    if not synthetic_path.exists():
        print(f"[ERROR] Synthetic data not found: {synthetic_path}")
        return 1
    
    with open(synthetic_path, "r", encoding="utf-8") as f:
        synthetic = json.load(f)
    print(f"\nLoaded synthetic: {len(synthetic)} series")
    
    # Load reference data
    reference_path = Path("data/reference/timeseries_reference.json")
    if not reference_path.exists():
        print(f"[ERROR] Reference data not found: {reference_path}")
        return 1
    
    with open(reference_path, "r", encoding="utf-8") as f:
        reference = json.load(f)
    print(f"Loaded reference: {len(reference)} series")
    
    # Initialize validator (without reference to avoid circular comparison)
    validator = TimeSeriesValidator(expected_length=24)
    
    print("\n" + "=" * 60)
    print("STATIONARITY COMPARISON (ADF Test)")
    print("=" * 60)
    
    # Stationarity - Synthetic
    syn_stat = validator.validate_stationarity(synthetic)
    print(f"\nSynthetic ({len(synthetic)} series):")
    print(f"  Stationary: {syn_stat*100:.1f}%")
    
    # Stationarity - Reference (sample first 50 for efficiency)
    ref_sample = reference[:50]
    # Reference uses "values" key, need to adapt
    ref_adapted = [{"target": s.get("values", s.get("target", []))} for s in ref_sample]
    ref_stat = validator.validate_stationarity(ref_adapted)
    print(f"\nReference ({len(ref_sample)} series sampled):")
    print(f"  Stationary: {ref_stat*100:.1f}%")
    
    # Comparison
    diff = abs(syn_stat - ref_stat)
    print(f"\nDifference: {diff*100:.1f} percentage points")
    
    if diff < 0.20:
        status = "[OK] SIMILAR"
        conclusion = "Synthetic data matches reference stationarity pattern"
    elif diff < 0.35:
        status = "[WARN] MODERATE DIFFERENCE"
        conclusion = "Some difference, but within acceptable range"
    else:
        status = "[ALERT] SIGNIFICANT DIFFERENCE"
        conclusion = "Consider adjusting generation prompts"
    
    print(f"Status: {status}")
    print(f"Conclusion: {conclusion}")
    
    print("\n" + "=" * 60)
    print("BASIC STATS COMPARISON")
    print("=" * 60)
    
    syn_stats = validator.validate_basic_stats(synthetic)
    ref_stats = validator.validate_basic_stats(ref_adapted)
    
    print(f"\n{'Metric':<20} {'Synthetic':>12} {'Reference':>12}")
    print("-" * 46)
    print(f"{'Completeness':<20} {syn_stats['completeness']*100:>11.1f}% {ref_stats['completeness']*100:>11.1f}%")
    print(f"{'Length correct':<20} {syn_stats['length_correct']*100:>11.1f}% {ref_stats['length_correct']*100:>11.1f}%")
    print(f"{'Bounds OK':<20} {syn_stats['bounds_ok']*100:>11.1f}% {ref_stats['bounds_ok']*100:>11.1f}%")
    
    print("\n" + "=" * 60)
    print("AUTOCORRELATION COMPARISON")
    print("=" * 60)
    
    syn_acf = validator.validate_autocorrelation(synthetic)
    ref_acf = validator.validate_autocorrelation(ref_adapted)
    
    print(f"\nSynthetic ACF score: {syn_acf:.2f}")
    print(f"Reference ACF score: {ref_acf:.2f}")
    
    print("\n" + "=" * 60)
    print("FINAL VERDICT")
    print("=" * 60)
    
    # Decision matrix
    print(f"""
    Reference stationarity: {ref_stat*100:.1f}%
    Synthetic stationarity: {syn_stat*100:.1f}%
    Difference: {diff*100:.1f}%
    
    INTERPRETATION:
    """)
    
    if ref_stat < 0.60:
        print("    Reference data is mostly NON-STATIONARY (as expected for electricity)")
        if syn_stat < 0.60:
            print("    Synthetic data is also mostly NON-STATIONARY")
            print("    --> [OK] This is CORRECT behavior - no fix needed")
            return 0
        else:
            print("    Synthetic data is TOO STATIONARY")
            print("    --> [WARN] May need to add more trends/seasonality to prompts")
    else:
        print("    Reference data is mostly STATIONARY")
        if syn_stat < 0.50:
            print("    Synthetic data is NON-STATIONARY")
            print("    --> [WARN] May need to reduce trends in prompts")
        else:
            print("    Synthetic data matches reference pattern")
            print("    --> [OK] No fix needed")
            return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

