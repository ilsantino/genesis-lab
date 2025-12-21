"""
Throttling diagnostic script for AWS Bedrock.

Tests different delay settings to find optimal configuration for avoiding throttling.

Usage:
    uv run python scripts/diagnose_throttling.py
    uv run python scripts/diagnose_throttling.py --delays 2,5,10 --calls 5
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_with_delay(delay_seconds: float, num_calls: int) -> Tuple[int, int, float]:
    """
    Test Bedrock with a specific delay between calls.
    
    Args:
        delay_seconds: Delay between API calls
        num_calls: Number of calls to make
    
    Returns:
        Tuple of (successes, failures, avg_response_time)
    """
    from src.utils.aws_client import BedrockClient
    
    client = BedrockClient.from_config()
    
    successes = 0
    failures = 0
    response_times = []
    
    print(f"\n  Testing with {delay_seconds}s delay ({num_calls} calls)...")
    
    for i in range(num_calls):
        start_time = time.time()
        
        try:
            response = client.invoke_model(
                prompt="Say 'OK' only.",
                system_prompt="You are a test bot. Reply with exactly 'OK'.",
                max_tokens=5,
                temperature=0.0
            )
            
            elapsed = time.time() - start_time
            response_times.append(elapsed)
            successes += 1
            print(f"    Call {i+1}/{num_calls}: [OK] ({elapsed:.1f}s)")
            
        except Exception as e:
            elapsed = time.time() - start_time
            failures += 1
            error_type = "Throttled" if "Throttling" in str(e) else "Error"
            print(f"    Call {i+1}/{num_calls}: [FAIL] {error_type} ({elapsed:.1f}s)")
        
        # Wait before next call (except for last call)
        if i < num_calls - 1:
            time.sleep(delay_seconds)
    
    avg_response_time = sum(response_times) / len(response_times) if response_times else 0
    return successes, failures, avg_response_time


def calculate_optimal_settings(results: Dict[float, Tuple[int, int, float]]) -> Dict:
    """
    Calculate optimal settings based on test results.
    
    Args:
        results: Dict mapping delay -> (successes, failures, avg_time)
    
    Returns:
        Dict with recommended settings
    """
    recommendations = {}
    
    for delay, (successes, failures, avg_time) in results.items():
        total = successes + failures
        success_rate = successes / total if total > 0 else 0
        
        # Calculate effective requests per minute
        # Each request takes: avg_response_time + delay
        time_per_request = avg_time + delay
        requests_per_minute = 60 / time_per_request if time_per_request > 0 else 0
        
        recommendations[delay] = {
            "success_rate": success_rate,
            "successes": successes,
            "failures": failures,
            "avg_response_time": avg_time,
            "effective_rpm": requests_per_minute,
            "effective_rpm_adjusted": requests_per_minute * success_rate
        }
    
    return recommendations


def find_optimal_delay(recommendations: Dict) -> Tuple[float, Dict]:
    """
    Find the optimal delay that maximizes throughput while minimizing failures.
    
    Args:
        recommendations: Dict from calculate_optimal_settings
    
    Returns:
        Tuple of (optimal_delay, settings)
    """
    # Prioritize: 
    # 1. Success rate >= 80%
    # 2. Highest effective throughput (adjusted RPM)
    
    viable = {
        delay: settings 
        for delay, settings in recommendations.items() 
        if settings["success_rate"] >= 0.8
    }
    
    if not viable:
        # Fall back to highest success rate
        best_delay = max(recommendations.keys(), key=lambda d: recommendations[d]["success_rate"])
        return best_delay, recommendations[best_delay]
    
    # Among viable options, pick highest adjusted RPM
    best_delay = max(viable.keys(), key=lambda d: viable[d]["effective_rpm_adjusted"])
    return best_delay, viable[best_delay]


def get_current_env_settings() -> Dict:
    """Read current settings from .env file."""
    env_path = Path(".env")
    settings = {
        "MAX_REQUESTS_PER_MINUTE": None,
        "BEDROCK_DELAY_SECONDS": None
    }
    
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key in settings:
                            settings[key] = value
    
    return settings


def update_env_file(optimal_delay: float, optimal_rpm: float, dry_run: bool = False) -> bool:
    """
    Update .env file with recommended settings.
    
    Args:
        optimal_delay: Recommended delay in seconds
        optimal_rpm: Recommended requests per minute
        dry_run: If True, only show what would be changed
    
    Returns:
        True if file was updated
    """
    env_path = Path(".env")
    
    # Read existing content
    existing_lines = []
    if env_path.exists():
        with open(env_path, "r") as f:
            existing_lines = f.readlines()
    
    # Prepare new settings
    new_settings = {
        "BEDROCK_DELAY_SECONDS": str(optimal_delay),
        "MAX_REQUESTS_PER_MINUTE": str(int(optimal_rpm))
    }
    
    # Update or append settings
    updated_keys = set()
    new_lines = []
    
    for line in existing_lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in new_settings:
                new_lines.append(f"{key}={new_settings[key]}\n")
                updated_keys.add(key)
                continue
        new_lines.append(line)
    
    # Add missing settings
    for key, value in new_settings.items():
        if key not in updated_keys:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines.append("\n")
            new_lines.append(f"\n# Throttling settings (auto-configured)\n")
            new_lines.append(f"{key}={value}\n")
            break  # Only add header once
    
    for key, value in new_settings.items():
        if key not in updated_keys:
            new_lines.append(f"{key}={value}\n")
    
    if dry_run:
        print("\n  Would update .env with:")
        for key, value in new_settings.items():
            print(f"    {key}={value}")
        return False
    
    # Write updated content
    with open(env_path, "w") as f:
        f.writelines(new_lines)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Diagnose AWS Bedrock throttling")
    parser.add_argument(
        "--delays", 
        type=str, 
        default="2,5,10",
        help="Comma-separated delay values to test (default: 2,5,10)"
    )
    parser.add_argument(
        "--calls", 
        type=int, 
        default=5,
        help="Number of calls per delay setting (default: 5)"
    )
    parser.add_argument(
        "--update-env",
        action="store_true",
        help="Update .env file with recommended settings"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without actually changing"
    )
    
    args = parser.parse_args()
    
    # Parse delay values
    delays = [float(d.strip()) for d in args.delays.split(",")]
    num_calls = args.calls
    
    print("=" * 60)
    print("AWS Bedrock Throttling Diagnostic")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Delays to test: {delays}s")
    print(f"  Calls per delay: {num_calls}")
    print(f"  Total API calls: {len(delays) * num_calls}")
    
    # Get current settings
    current_settings = get_current_env_settings()
    print(f"\nCurrent .env settings:")
    print(f"  MAX_REQUESTS_PER_MINUTE: {current_settings['MAX_REQUESTS_PER_MINUTE'] or 'not set'}")
    print(f"  BEDROCK_DELAY_SECONDS: {current_settings['BEDROCK_DELAY_SECONDS'] or 'not set'}")
    
    # Run tests
    print("\n" + "-" * 60)
    print("Running throttling tests...")
    print("-" * 60)
    
    results = {}
    total_start = time.time()
    
    for delay in delays:
        successes, failures, avg_time = test_with_delay(delay, num_calls)
        results[delay] = (successes, failures, avg_time)
    
    total_elapsed = time.time() - total_start
    
    # Calculate recommendations
    recommendations = calculate_optimal_settings(results)
    optimal_delay, optimal_settings = find_optimal_delay(recommendations)
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    print("\nSuccess rates by delay:")
    print("-" * 40)
    print(f"{'Delay':>8} | {'Success':>8} | {'Rate':>8} | {'Eff. RPM':>10}")
    print("-" * 40)
    
    for delay in sorted(recommendations.keys()):
        r = recommendations[delay]
        rate_pct = r["success_rate"] * 100
        indicator = "[OK]" if r["success_rate"] >= 0.8 else "[!!]"
        print(f"{delay:>6.1f}s | {r['successes']:>4}/{r['successes']+r['failures']:<3} | {rate_pct:>6.0f}% | {r['effective_rpm_adjusted']:>8.1f} {indicator}")
    
    print("-" * 40)
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    optimal_rpm = optimal_settings["effective_rpm_adjusted"]
    
    print(f"\n  Optimal delay: {optimal_delay}s")
    print(f"  Expected success rate: {optimal_settings['success_rate']*100:.0f}%")
    print(f"  Effective throughput: {optimal_rpm:.1f} requests/min")
    print(f"  Avg response time: {optimal_settings['avg_response_time']:.1f}s")
    
    # Compare with current
    print("\n  Recommended .env settings:")
    print(f"    BEDROCK_DELAY_SECONDS={optimal_delay}")
    print(f"    MAX_REQUESTS_PER_MINUTE={int(optimal_rpm)}")
    
    # Update .env if requested
    if args.update_env or args.dry_run:
        print("\n" + "-" * 60)
        if args.dry_run:
            update_env_file(optimal_delay, optimal_rpm, dry_run=True)
        else:
            if update_env_file(optimal_delay, optimal_rpm):
                print("  [OK] .env file updated with recommended settings")
            else:
                print("  [FAIL] Could not update .env file")
    else:
        print("\n  Run with --update-env to apply these settings")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n  Total test time: {total_elapsed:.1f}s")
    print(f"  Total API calls: {sum(r[0]+r[1] for r in results.values())}")
    print(f"  Overall success rate: {sum(r[0] for r in results.values())}/{sum(r[0]+r[1] for r in results.values())}")
    
    if optimal_settings["success_rate"] >= 0.8:
        print(f"\n  [OK] Recommended: Use {optimal_delay}s delay for reliable generation")
    else:
        print(f"\n  [WARN] All delays show <80% success. Consider:")
        print("    - Waiting a few minutes before retrying")
        print("    - Requesting AWS quota increase")
        print("    - Using a different model (e.g., claude-3-haiku)")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

