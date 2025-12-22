"""
Smoke test for Time Series generation only.

Generates 10 time series with various types and validates output.

Usage:
    uv run python scripts/smoke_test_ts.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import TimeSeriesGenerator
from src.generation.schemas import TimeSeries


TOTAL_SERIES = 10
DELAY_SECONDS = 10  # Increased from 5s to avoid throttling
OUTPUT_FILE = Path("data/synthetic/ts_smoke_test.json")


def validate_series(series: dict) -> tuple[bool, str]:
    """Validate a time series against schema."""
    try:
        # Check required fields
        required = ["series_id", "domain", "series_type", "target"]
        for field in required:
            if field not in series:
                return False, f"Missing field: {field}"
        
        # Check target is list of numbers
        target = series.get("target", [])
        if not isinstance(target, list):
            return False, "target is not a list"
        
        if len(target) < 10:
            return False, f"target too short: {len(target)}"
        
        for i, val in enumerate(target):
            if not isinstance(val, (int, float)):
                return False, f"target[{i}] is not numeric: {type(val)}"
        
        return True, ""
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 60)
    print("Time Series Smoke Test")
    print("=" * 60)
    print(f"Total: {TOTAL_SERIES} series")
    print(f"Delay: {DELAY_SECONDS}s between calls")
    print("=" * 60)
    
    # Initialize generator
    print("\nInitializing generator...")
    generator = TimeSeriesGenerator.from_config()
    print(f"Ready with {generator.series_type_count} series types")
    print(f"Domains: {generator.available_domains}\n")
    
    # Generate series
    results = []
    failed = []
    start_time = time.time()
    
    # Cycle through domains
    domains = ["electricity", "energy", "sensors", "financial"]
    
    print("-" * 60)
    print("GENERATING...")
    print("-" * 60)
    
    for i in range(TOTAL_SERIES):
        domain = domains[i % len(domains)]
        series_types = generator.get_series_types_for_domain(domain)
        series_type = series_types[i % len(series_types)] if series_types else None
        
        print(f"  [{i+1:2d}/{TOTAL_SERIES}] [{domain}] {series_type or 'random':25s} ", end="", flush=True)
        
        try:
            series = generator.generate_single(
                series_type=series_type,
                length=24,
                frequency="1H",
                complexity="medium"
            )
            
            # Validate
            is_valid, error = validate_series(series)
            
            if is_valid:
                results.append(series)
                print("[OK]")
            else:
                failed.append({"index": i, "error": error})
                print(f"[INVALID] {error}")
                
        except Exception as e:
            error_type = "Throttled" if "Throttling" in str(e) else "Error"
            failed.append({"index": i, "error": str(e)[:50]})
            print(f"[FAIL] {error_type}")
        
        # Delay before next
        if i < TOTAL_SERIES - 1:
            time.sleep(DELAY_SECONDS)
    
    # Calculate stats
    elapsed = time.time() - start_time
    success_rate = len(results) / TOTAL_SERIES * 100
    
    # Save results
    if results:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Success: {len(results)}/{TOTAL_SERIES} ({success_rate:.0f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/TOTAL_SERIES:.1f}s avg)")
    
    if results:
        # Analyze domains
        domain_counts = {}
        for s in results:
            d = s.get("domain", "unknown")
            domain_counts[d] = domain_counts.get(d, 0) + 1
        print(f"\n  Domains: {domain_counts}")
        
        # Show sample
        print(f"\n  Sample series:")
        for s in results[:3]:
            target = s.get("target", [])
            print(f"    - {s.get('series_type', 'unknown')}: {len(target)} points")
            print(f"      First 5: {target[:5]}")
        
        print(f"\n  Saved to: {OUTPUT_FILE}")
    
    print("\n" + "=" * 60)
    
    # Return exit code
    if success_rate >= 90:
        print("[OK] Smoke test passed!")
        return 0
    else:
        print(f"[WARN] Success rate {success_rate:.0f}% below 90% threshold")
        return 1


if __name__ == "__main__":
    sys.exit(main())

