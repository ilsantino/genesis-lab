"""
Smoke test script for GENESIS-LAB generators.

Generates small batches of customer service conversations and time series data
with throttling protection and Pydantic validation.

Usage:
    uv run python -m scripts.smoke_test
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from pydantic import ValidationError

from src.generation import CustomerServiceGenerator, TimeSeriesGenerator


# =============================================================================
# CONFIGURATION
# =============================================================================

TOTAL_CONVERSATIONS = 10
TOTAL_TIMESERIES = 10
BATCH_SIZE = 2
DELAY_BETWEEN_BATCHES = 3  # seconds

OUTPUT_DIR = Path("data/synthetic")
CS_OUTPUT_FILE = OUTPUT_DIR / "customer_service_smoke_test.json"
TS_OUTPUT_FILE = OUTPUT_DIR / "timeseries_smoke_test.json"


# =============================================================================
# LIGHTWEIGHT VALIDATION (matches generator output format)
# =============================================================================

def validate_conversation(conv: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate customer service conversation structure.
    
    Args:
        conv: Conversation dictionary from generator
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["conversation_id", "intent", "sentiment", "turns", "resolution_status"]
    
    for field in required_fields:
        if field not in conv:
            return False, f"Missing required field: {field}"
    
    # Validate turns
    turns = conv.get("turns", [])
    if not isinstance(turns, list) or len(turns) < 2:
        return False, f"Invalid turns: expected list with at least 2 items, got {len(turns) if isinstance(turns, list) else type(turns)}"
    
    for i, turn in enumerate(turns):
        if not isinstance(turn, dict):
            return False, f"Turn {i} is not a dict"
        if "speaker" not in turn or "text" not in turn:
            return False, f"Turn {i} missing speaker or text"
        if turn["speaker"] not in ["customer", "agent"]:
            return False, f"Turn {i} has invalid speaker: {turn['speaker']}"
    
    # First turn should be from customer
    if turns[0]["speaker"] != "customer":
        return False, "First turn must be from customer"
    
    return True, ""


def validate_timeseries(ts: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate time series structure.
    
    Args:
        ts: Time series dictionary from generator
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    required_fields = ["series_id", "domain", "series_type", "frequency", "target"]
    
    for field in required_fields:
        if field not in ts:
            return False, f"Missing required field: {field}"
    
    # Validate target array
    target = ts.get("target", [])
    if not isinstance(target, list):
        return False, f"Target is not a list: {type(target)}"
    
    if len(target) < 1:
        return False, "Target array is empty"
    
    # Validate all values are numeric
    for i, val in enumerate(target[:10]):  # Check first 10
        if not isinstance(val, (int, float)):
            return False, f"Target[{i}] is not numeric: {type(val)}"
    
    return True, ""


# =============================================================================
# BATCH GENERATION WITH THROTTLING
# =============================================================================

def generate_conversations_with_throttling(
    generator: CustomerServiceGenerator,
    total: int,
    batch_size: int,
    delay: int
) -> Tuple[List[Dict], int, int]:
    """
    Generate conversations in small batches with delays.
    
    Returns:
        Tuple of (successful_results, success_count, failure_count)
    """
    results = []
    success_count = 0
    failure_count = 0
    
    num_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_count = batch_end - batch_start
        
        print(f"  Generating conversations {batch_start + 1}-{batch_end}/{total}...")
        
        try:
            batch_results = generator.generate_batch(
                count=batch_count,
                continue_on_error=True
            )
            
            for conv in batch_results:
                is_valid, error = validate_conversation(conv)
                if is_valid:
                    results.append(conv)
                    success_count += 1
                else:
                    print(f"    [VALIDATION FAILED] {error}")
                    failure_count += 1
                    
        except Exception as e:
            print(f"    [BATCH ERROR] {e}")
            failure_count += batch_count
        
        # Delay before next batch (except for last batch)
        if batch_idx < num_batches - 1:
            print(f"  Waiting {delay}s to avoid throttling...")
            time.sleep(delay)
    
    return results, success_count, failure_count


def generate_timeseries_with_throttling(
    generator: TimeSeriesGenerator,
    total: int,
    batch_size: int,
    delay: int
) -> Tuple[List[Dict], int, int]:
    """
    Generate time series in small batches with delays.
    
    Returns:
        Tuple of (successful_results, success_count, failure_count)
    """
    results = []
    success_count = 0
    failure_count = 0
    
    num_batches = (total + batch_size - 1) // batch_size
    
    for batch_idx in range(num_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total)
        batch_count = batch_end - batch_start
        
        print(f"  Generating time series {batch_start + 1}-{batch_end}/{total}...")
        
        try:
            batch_results = generator.generate_batch(
                count=batch_count,
                length=24,  # 24 hours
                frequency="1H",
                continue_on_error=True
            )
            
            for ts in batch_results:
                is_valid, error = validate_timeseries(ts)
                if is_valid:
                    results.append(ts)
                    success_count += 1
                else:
                    print(f"    [VALIDATION FAILED] {error}")
                    failure_count += 1
                    
        except Exception as e:
            print(f"    [BATCH ERROR] {e}")
            failure_count += batch_count
        
        # Delay before next batch (except for last batch)
        if batch_idx < num_batches - 1:
            print(f"  Waiting {delay}s to avoid throttling...")
            time.sleep(delay)
    
    return results, success_count, failure_count


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run smoke test for all generators."""
    print("=" * 60)
    print("GENESIS-LAB Smoke Test")
    print("=" * 60)
    print(f"Config: {TOTAL_CONVERSATIONS} conversations, {TOTAL_TIMESERIES} time series")
    print(f"Batch size: {BATCH_SIZE}, Delay: {DELAY_BETWEEN_BATCHES}s")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    
    # =========================================================================
    # CUSTOMER SERVICE CONVERSATIONS
    # =========================================================================
    print("\n[1/2] Customer Service Conversations")
    print("-" * 40)
    
    cs_start = time.time()
    cs_generator = CustomerServiceGenerator.from_config()
    print(f"  Generator ready with {cs_generator.intent_count} intents")
    
    cs_results, cs_success, cs_failed = generate_conversations_with_throttling(
        generator=cs_generator,
        total=TOTAL_CONVERSATIONS,
        batch_size=BATCH_SIZE,
        delay=DELAY_BETWEEN_BATCHES
    )
    cs_elapsed = time.time() - cs_start
    
    # Save results
    if cs_results:
        with open(CS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cs_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved {len(cs_results)} conversations to {CS_OUTPUT_FILE}")
    
    print(f"  Time: {cs_elapsed:.1f}s | Success: {cs_success} | Failed: {cs_failed}")
    
    # =========================================================================
    # TIME SERIES
    # =========================================================================
    print("\n[2/2] Time Series")
    print("-" * 40)
    
    ts_start = time.time()
    ts_generator = TimeSeriesGenerator.from_config()
    print(f"  Generator ready with {ts_generator.series_type_count} series types")
    
    ts_results, ts_success, ts_failed = generate_timeseries_with_throttling(
        generator=ts_generator,
        total=TOTAL_TIMESERIES,
        batch_size=BATCH_SIZE,
        delay=DELAY_BETWEEN_BATCHES
    )
    ts_elapsed = time.time() - ts_start
    
    # Save results
    if ts_results:
        with open(TS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(ts_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"  Saved {len(ts_results)} time series to {TS_OUTPUT_FILE}")
    
    print(f"  Time: {ts_elapsed:.1f}s | Success: {ts_success} | Failed: {ts_failed}")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"Total time: {overall_elapsed:.1f}s")
    print(f"\nCustomer Service:")
    print(f"  - Generated: {cs_success}/{TOTAL_CONVERSATIONS}")
    print(f"  - Failed: {cs_failed}")
    print(f"  - Output: {CS_OUTPUT_FILE}")
    print(f"\nTime Series:")
    print(f"  - Generated: {ts_success}/{TOTAL_TIMESERIES}")
    print(f"  - Failed: {ts_failed}")
    print(f"  - Output: {TS_OUTPUT_FILE}")
    
    total_success = cs_success + ts_success
    total_failed = cs_failed + ts_failed
    total_attempted = TOTAL_CONVERSATIONS + TOTAL_TIMESERIES
    
    print(f"\nOverall: {total_success}/{total_attempted} successful ({total_success/total_attempted*100:.0f}%)")
    
    if total_failed == 0:
        print("\n[OK] All generations passed validation!")
    else:
        print(f"\n[WARN] {total_failed} generations failed")
    
    return total_failed == 0


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)

