"""Test script for time series generation."""
import time
from src.generation import TimeSeriesGenerator


def test_single():
    """Test single time series generation."""
    print("=" * 50)
    print("Single Time Series Generation Test")
    print("=" * 50)

    # Initialize generator
    g = TimeSeriesGenerator.from_config()
    print(f"Generator ready with {g.series_type_count} series types")
    print(f"Domains: {g.available_domains}\n")

    # Generate single time series
    print("Generating single time series (24 hours, residential consumption)...")
    start = time.time()
    
    series = g.generate_single(
        series_type="residential_consumption",
        length=24,
        frequency="1H",
        complexity="medium",
        language="en"
    )
    
    elapsed = time.time() - start
    
    print(f"\n{'=' * 50}")
    print("RESULT")
    print(f"{'=' * 50}")
    print(f"Series ID: {series['series_id']}")
    print(f"Domain: {series['domain']}")
    print(f"Series Type: {series['series_type']}")
    print(f"Length: {series['length']} points")
    print(f"Frequency: {series['frequency']}")
    print(f"Complexity: {series['complexity']}")
    print(f"Seasonality: {series.get('seasonality_types', [])}")
    print(f"Trend: {series.get('trend_type', 'none')}")
    print(f"Time: {elapsed:.1f}s")
    
    # Show first few and last few data points
    target = series.get("target", [])
    if target:
        print(f"\nFirst 6 values: {target[:6]}")
        print(f"Last 6 values: {target[-6:]}")
    
    print(f"\n[OK] Single generation test complete!")
    return series


def test_batch(count: int = 10):
    """Test batch time series generation."""
    print("\n" + "=" * 50)
    print(f"Batch Time Series Generation Test ({count} series)")
    print("=" * 50)

    # Initialize generator
    g = TimeSeriesGenerator.from_config()
    
    # Generate batch
    start = time.time()
    series_list = g.generate_batch(
        count=count,
        length=24,
        frequency="1H",
        continue_on_error=True
    )
    elapsed = time.time() - start
    
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Generated: {len(series_list)} time series")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(series_list), 1):.1f}s avg)")
    print(f"Metrics: {g.get_metrics()}")
    
    # Sample output
    print(f"\n{'=' * 50}")
    print("SAMPLE TIME SERIES")
    print(f"{'=' * 50}")
    for i, s in enumerate(series_list[:5], 1):
        domain = s.get('domain', 'unknown')
        series_type = s.get('series_type', 'unknown')
        length = s.get('length', 0)
        complexity = s.get('complexity', 'unknown')
        print(f"{i}. [{domain}] {series_type}")
        print(f"   Length: {length}, Complexity: {complexity}")
    
    print(f"\n[OK] Batch generation test complete!")
    return series_list


if __name__ == "__main__":
    import sys
    
    # Default: run single test only
    # Pass "batch" argument to run batch test
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        test_batch(10)
    else:
        test_single()

