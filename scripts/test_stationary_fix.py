"""
Test script to validate the stationary parameter fix.

Generates 10 time series with stationary=True and runs validation.
Target: >70% stationarity (up from 44%)
"""

import json
import time
from pathlib import Path

from src.generation import TimeSeriesGenerator
from src.validation.timeseries_quality import TimeSeriesValidator


def main():
    print("=" * 60)
    print("STATIONARY FIX VALIDATION TEST")
    print("=" * 60)
    
    # Configuration
    total = 10
    delay = 10  # seconds between calls
    output_path = Path("data/synthetic/ts_stationary_test.json")
    
    print(f"\nConfiguration:")
    print(f"  Generating: {total} time series")
    print(f"  stationary: True")
    print(f"  Delay: {delay}s between calls")
    print(f"  Output: {output_path}")
    
    # Initialize generator
    generator = TimeSeriesGenerator.from_config()
    print(f"\n[OK] Generator ready ({generator.series_type_count} series types)")
    
    # Generate with stationary=True
    print(f"\n{'='*60}")
    print("GENERATING STATIONARY TIME SERIES")
    print("=" * 60)
    
    results = []
    failures = 0
    start_time = time.time()
    
    for i in range(total):
        print(f"  [{i+1}/{total}] Generating...", end=" ", flush=True)
        try:
            ts = generator.generate_single(
                length=168,  # 1 week hourly
                frequency="1H",
                stationary=True  # ← THE FIX
            )
            results.append(ts)
            print(f"[OK] {ts['series_type']}")
        except Exception as e:
            failures += 1
            print(f"[FAIL] {str(e)[:50]}")
        
        if i < total - 1:
            time.sleep(delay)
    
    elapsed = time.time() - start_time
    success_rate = len(results) / total
    
    print(f"\n{'='*60}")
    print("GENERATION RESULTS")
    print("=" * 60)
    print(f"  Success: {len(results)}/{total} ({success_rate*100:.0f}%)")
    print(f"  Failures: {failures}")
    print(f"  Time: {elapsed:.1f}s")
    
    if not results:
        print("\n[FAIL] No results to validate")
        return
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to: {output_path}")
    
    # Run validation
    print(f"\n{'='*60}")
    print("RUNNING TIME SERIES VALIDATION")
    print("=" * 60)
    
    validator = TimeSeriesValidator()
    metrics = validator.compute_overall_score(results)
    
    print(f"\n  Stationarity:    {metrics.stationarity_score*100:.0f}%")
    print(f"  Autocorrelation: {metrics.autocorrelation_score*100:.0f}%")
    print(f"  Completeness:    {metrics.completeness_score*100:.0f}%")
    print(f"  Overall Score:   {metrics.overall_quality_score:.1f}/100")
    
    # Compare with previous
    print(f"\n{'='*60}")
    print("COMPARISON: BEFORE vs AFTER FIX")
    print("=" * 60)
    print(f"                    Before    After     Target")
    print(f"  Stationarity:       44%    {metrics.stationarity_score*100:>5.0f}%      >70%")
    print(f"  Autocorrelation:    50%    {metrics.autocorrelation_score*100:>5.0f}%      >60%")
    
    # Verdict
    stationarity_pass = metrics.stationarity_score >= 0.70
    autocorr_pass = metrics.autocorrelation_score >= 0.60
    
    print(f"\n{'='*60}")
    if stationarity_pass and autocorr_pass:
        print("[OK] FIX SUCCESSFUL! Thresholds met.")
    elif stationarity_pass:
        print("[PARTIAL] Stationarity improved, autocorrelation needs work.")
    else:
        print("[FAIL] Fix did not meet stationarity threshold.")
    print("=" * 60)


if __name__ == "__main__":
    main()

