"""
Smoke test script for GENESIS-LAB generators (v2).

Generates customer service conversations and time series data
with optimized throttling settings for >90% success rate.

Usage:
    uv run python -m scripts.smoke_test
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.generation import CustomerServiceGenerator, TimeSeriesGenerator


# =============================================================================
# CONFIGURATION (v2 - optimized for >90% success rate)
# =============================================================================

TOTAL_CONVERSATIONS = 10
TOTAL_TIMESERIES = 10
BATCH_SIZE = 1  # Sequential generation for reliability
DELAY_BETWEEN_CALLS = 5  # 5 seconds between calls (optimal from diagnostic)

SUCCESS_RATE_THRESHOLD = 0.90  # Target >90% success

OUTPUT_DIR = Path("data/synthetic")
CS_OUTPUT_FILE = OUTPUT_DIR / "cs_smoke_v2.json"
TS_OUTPUT_FILE = OUTPUT_DIR / "ts_smoke_v2.json"


# =============================================================================
# VALIDATION FUNCTIONS
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
# SEQUENTIAL GENERATION WITH THROTTLING
# =============================================================================

def generate_conversations_sequential(
    generator: CustomerServiceGenerator,
    total: int,
    delay: float,
    bilingual: bool = True
) -> Tuple[List[Dict], int, int]:
    """
    Generate conversations one at a time with delays.
    
    Args:
        generator: CustomerServiceGenerator instance
        total: Number of conversations to generate
        delay: Delay in seconds between calls
        bilingual: If True, alternate between EN and ES (50/50 split)
    
    Returns:
        Tuple of (successful_results, success_count, failure_count)
    """
    results = []
    success_count = 0
    failure_count = 0
    
    for i in range(total):
        # Alternate between EN and ES for bilingual mode
        language = "en" if (i % 2 == 0 or not bilingual) else "es"
        lang_label = f"[{language.upper()}]"
        
        print(f"  [{i+1}/{total}] {lang_label} Generating conversation...", end=" ", flush=True)
        
        try:
            conv = generator.generate_single(language=language)
            
            is_valid, error = validate_conversation(conv)
            if is_valid:
                results.append(conv)
                success_count += 1
                intent = conv.get("intent", "unknown")[:20]
                print(f"[OK] intent={intent}")
            else:
                print(f"[INVALID] {error}")
                failure_count += 1
                
        except Exception as e:
            error_type = "Throttled" if "Throttling" in str(e) else "Error"
            print(f"[FAIL] {error_type}")
            failure_count += 1
        
        # Delay before next call (except for last)
        if i < total - 1:
            time.sleep(delay)
    
    return results, success_count, failure_count


def generate_timeseries_sequential(
    generator: TimeSeriesGenerator,
    total: int,
    delay: float
) -> Tuple[List[Dict], int, int]:
    """
    Generate time series one at a time with delays.
    
    Returns:
        Tuple of (successful_results, success_count, failure_count)
    """
    results = []
    success_count = 0
    failure_count = 0
    
    for i in range(total):
        print(f"  [{i+1}/{total}] Generating time series...", end=" ", flush=True)
        
        try:
            ts = generator.generate_single(
                length=24,  # 24 hours
                frequency="1H"
            )
            
            is_valid, error = validate_timeseries(ts)
            if is_valid:
                results.append(ts)
                success_count += 1
                series_type = ts.get("series_type", "unknown")[:20]
                print(f"[OK] type={series_type}")
            else:
                print(f"[INVALID] {error}")
                failure_count += 1
                
        except Exception as e:
            error_type = "Throttled" if "Throttling" in str(e) else "Error"
            print(f"[FAIL] {error_type}")
            failure_count += 1
        
        # Delay before next call (except for last)
        if i < total - 1:
            time.sleep(delay)
    
    return results, success_count, failure_count


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run smoke test v2 with optimized throttling and bilingual support."""
    print("=" * 60)
    print("GENESIS-LAB Smoke Test v2 (Bilingual)")
    print("=" * 60)
    print(f"Target: {TOTAL_CONVERSATIONS} conversations + {TOTAL_TIMESERIES} time series")
    print(f"Mode: Sequential (batch size 1), Bilingual (EN/ES)")
    print(f"Delay: {DELAY_BETWEEN_CALLS}s between calls")
    print(f"Success threshold: {SUCCESS_RATE_THRESHOLD*100:.0f}%")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    overall_start = time.time()
    
    # =========================================================================
    # CUSTOMER SERVICE CONVERSATIONS (Bilingual EN/ES)
    # =========================================================================
    print("\n[1/2] Customer Service Conversations (Bilingual)")
    print("-" * 40)
    
    cs_start = time.time()
    cs_generator = CustomerServiceGenerator.from_config()
    print(f"  Generator ready with {cs_generator.intent_count} intents")
    print(f"  Language mode: Bilingual (50% EN, 50% ES)\n")
    
    cs_results, cs_success, cs_failed = generate_conversations_sequential(
        generator=cs_generator,
        total=TOTAL_CONVERSATIONS,
        delay=DELAY_BETWEEN_CALLS,
        bilingual=True
    )
    cs_elapsed = time.time() - cs_start
    cs_rate = cs_success / TOTAL_CONVERSATIONS if TOTAL_CONVERSATIONS > 0 else 0
    
    # Save results
    if cs_results:
        with open(CS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(cs_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved {len(cs_results)} conversations to {CS_OUTPUT_FILE}")
    
    print(f"  Time: {cs_elapsed:.1f}s | Success: {cs_success}/{TOTAL_CONVERSATIONS} ({cs_rate*100:.0f}%)")
    
    # =========================================================================
    # TIME SERIES
    # =========================================================================
    print("\n[2/2] Time Series")
    print("-" * 40)
    
    ts_start = time.time()
    ts_generator = TimeSeriesGenerator.from_config()
    print(f"  Generator ready with {ts_generator.series_type_count} series types\n")
    
    ts_results, ts_success, ts_failed = generate_timeseries_sequential(
        generator=ts_generator,
        total=TOTAL_TIMESERIES,
        delay=DELAY_BETWEEN_CALLS
    )
    ts_elapsed = time.time() - ts_start
    ts_rate = ts_success / TOTAL_TIMESERIES if TOTAL_TIMESERIES > 0 else 0
    
    # Save results
    if ts_results:
        with open(TS_OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(ts_results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved {len(ts_results)} time series to {TS_OUTPUT_FILE}")
    
    print(f"  Time: {ts_elapsed:.1f}s | Success: {ts_success}/{TOTAL_TIMESERIES} ({ts_rate*100:.0f}%)")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    overall_elapsed = time.time() - overall_start
    
    total_success = cs_success + ts_success
    total_attempted = TOTAL_CONVERSATIONS + TOTAL_TIMESERIES
    overall_rate = total_success / total_attempted if total_attempted > 0 else 0
    
    print("\n" + "=" * 60)
    print("SMOKE TEST v2 RESULTS")
    print("=" * 60)
    
    print(f"\nCustomer Service:")
    print(f"  - Generated: {cs_success}/{TOTAL_CONVERSATIONS} ({cs_rate*100:.0f}%)")
    print(f"  - Output: {CS_OUTPUT_FILE}")
    
    print(f"\nTime Series:")
    print(f"  - Generated: {ts_success}/{TOTAL_TIMESERIES} ({ts_rate*100:.0f}%)")
    print(f"  - Output: {TS_OUTPUT_FILE}")
    
    print(f"\n{'=' * 40}")
    print(f"OVERALL: {total_success}/{total_attempted} ({overall_rate*100:.0f}%)")
    print(f"{'=' * 40}")
    print(f"Time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} min)")
    
    # Check against threshold
    print("\n" + "-" * 40)
    if overall_rate >= SUCCESS_RATE_THRESHOLD:
        print(f"[PASS] Success rate {overall_rate*100:.0f}% >= {SUCCESS_RATE_THRESHOLD*100:.0f}% threshold")
        exit_code = 0
    else:
        print(f"[FAIL] Success rate {overall_rate*100:.0f}% < {SUCCESS_RATE_THRESHOLD*100:.0f}% threshold")
        exit_code = 1
    
    # Validation summary
    if cs_success == TOTAL_CONVERSATIONS and ts_success == TOTAL_TIMESERIES:
        print("[PASS] All generated items passed validation")
    else:
        failed = (TOTAL_CONVERSATIONS - cs_success) + (TOTAL_TIMESERIES - ts_success)
        print(f"[WARN] {failed} items failed generation or validation")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
