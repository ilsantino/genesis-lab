"""
Generate customer service conversations with configurable parameters.

Flexible generation script for both quick tests and overnight runs.

Features:
- Configurable total count, delay, and batch size
- Exponential backoff on throttling errors
- Auto-pause after consecutive failures
- Checkpointing and resume capability
- Cost estimation and ETA display
- Balanced distribution (language, sentiment, complexity)
- Real-time validation
- Auto-register in database

Usage:
    # Quick test (10 items)
    uv run python scripts/generate_data.py --total 10 --delay 3
    
    # Standard run (100 items)
    uv run python scripts/generate_data.py --total 100
    
    # Overnight run (500 items)
    uv run python scripts/generate_data.py --total 500 --delay 5 --max-failures 10
    
    # Resume interrupted run
    uv run python scripts/generate_data.py --resume
    
    # Dry run (show plan without generating)
    uv run python scripts/generate_data.py --total 100 --dry-run
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import CustomerServiceGenerator
from src.generation.templates.customer_service_prompts import ALL_INTENTS
from src.registry import DatasetRegistry
from src.validation import QualityValidator, BiasDetector


# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

DEFAULT_TOTAL = 100
DEFAULT_DELAY = 5  # seconds
DEFAULT_CHECKPOINT_INTERVAL = 10
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_FACTOR = 2.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
COST_PER_ITEM = 0.003  # USD estimate

# Distribution targets (percentages)
LANGUAGE_DIST = {"en": 50, "es": 50}
SENTIMENT_DIST = {"positive": 30, "neutral": 50, "negative": 20}
COMPLEXITY_DIST = {"simple": 30, "medium": 50, "complex": 20}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_eta(seconds: float) -> str:
    """Format ETA as time from now."""
    if seconds <= 0:
        return "now"
    eta_time = datetime.now() + timedelta(seconds=seconds)
    return eta_time.strftime("%H:%M")


def estimate_cost(count: int) -> float:
    """Estimate AWS Bedrock cost for generation."""
    return count * COST_PER_ITEM


def estimate_time(count: int, delay: float) -> float:
    """Estimate total time in seconds."""
    # Each item takes ~delay seconds + some processing overhead
    return count * (delay + 1)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_conversation(conv: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate conversation structure."""
    required_fields = ["conversation_id", "intent", "sentiment", "turns", "resolution_status"]
    
    for field in required_fields:
        if field not in conv:
            return False, f"Missing field: {field}"
    
    turns = conv.get("turns", [])
    if not isinstance(turns, list) or len(turns) < 2:
        return False, f"Invalid turns count: {len(turns) if isinstance(turns, list) else 0}"
    
    if turns[0].get("speaker") != "customer":
        return False, "First speaker must be customer"
    
    for i, turn in enumerate(turns):
        if "speaker" not in turn or "text" not in turn:
            return False, f"Turn {i} missing speaker or text"
        if not turn.get("text", "").strip():
            return False, f"Turn {i} has empty text"
    
    return True, ""


# =============================================================================
# DISTRIBUTION PLANNING
# =============================================================================

def create_generation_plan(total: int) -> List[Dict[str, str]]:
    """
    Create a balanced generation plan.
    
    Returns list of dicts with language, sentiment, complexity, intent for each item.
    """
    plan = []
    
    # Create intent cycle to cover as many intents as possible
    intent_cycle = list(ALL_INTENTS)  # 77 intents
    
    # Scale distributions to total
    lang_en = int(total * LANGUAGE_DIST["en"] / 100)
    lang_es = total - lang_en
    
    sent_pos = int(total * SENTIMENT_DIST["positive"] / 100)
    sent_neu = int(total * SENTIMENT_DIST["neutral"] / 100)
    sent_neg = total - sent_pos - sent_neu
    
    comp_simple = int(total * COMPLEXITY_DIST["simple"] / 100)
    comp_medium = int(total * COMPLEXITY_DIST["medium"] / 100)
    comp_complex = total - comp_simple - comp_medium
    
    # Create distribution queues
    languages = (["en"] * lang_en) + (["es"] * lang_es)
    sentiments = (["positive"] * sent_pos) + (["neutral"] * sent_neu) + (["negative"] * sent_neg)
    complexities = (["simple"] * comp_simple) + (["medium"] * comp_medium) + (["complex"] * comp_complex)
    
    # Shuffle for variety (but keep reproducible)
    import random
    random.seed(42)
    random.shuffle(languages)
    random.shuffle(sentiments)
    random.shuffle(complexities)
    
    for i in range(total):
        plan.append({
            "language": languages[i % len(languages)] if languages else "en",
            "sentiment": sentiments[i % len(sentiments)] if sentiments else "neutral",
            "complexity": complexities[i % len(complexities)] if complexities else "medium",
            "intent": intent_cycle[i % len(intent_cycle)]
        })
    
    return plan


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def get_checkpoint_path(total: int) -> Path:
    """Get checkpoint file path based on total."""
    return Path(f"data/synthetic/checkpoint_cs_{total}.json")


def get_output_path(total: int) -> Path:
    """Get output file path with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(f"data/synthetic/customer_service_{total}_{timestamp}.json")


def save_checkpoint(
    checkpoint_path: Path,
    conversations: List[Dict],
    current_index: int,
    failed_indices: List[int],
    start_time: float,
    total: int,
    config: Dict[str, Any]
) -> None:
    """Save current progress to checkpoint file."""
    checkpoint = {
        "conversations": conversations,
        "current_index": current_index,
        "failed_indices": failed_indices,
        "start_time": start_time,
        "checkpoint_time": datetime.now(timezone.utc).isoformat(),
        "total_target": total,
        "config": config
    }
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)


def load_checkpoint(checkpoint_path: Path) -> Optional[Dict]:
    """Load checkpoint if exists."""
    if checkpoint_path.exists():
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clear_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint file after successful completion."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_data(
    total: int = DEFAULT_TOTAL,
    delay: float = DEFAULT_DELAY,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
    resume: bool = False,
    dry_run: bool = False
) -> int:
    """
    Generate customer service conversations.
    
    Args:
        total: Total number of conversations to generate
        delay: Base delay between API calls (seconds)
        checkpoint_interval: Save checkpoint every N items
        max_retries: Max retries per item before skipping
        backoff_factor: Multiply delay by this on throttling
        max_consecutive_failures: Pause after this many consecutive failures
        resume: If True, resume from checkpoint
        dry_run: If True, show plan without generating
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    # Configuration summary
    config = {
        "total": total,
        "delay": delay,
        "checkpoint_interval": checkpoint_interval,
        "max_retries": max_retries,
        "backoff_factor": backoff_factor,
        "max_consecutive_failures": max_consecutive_failures
    }
    
    checkpoint_path = get_checkpoint_path(total)
    
    # Header
    print("=" * 70)
    print("GENESIS LAB - Customer Service Dataset Generation")
    print("=" * 70)
    print(f"  Target:      {total} conversations")
    print(f"  Languages:   {LANGUAGE_DIST['en']}% EN / {LANGUAGE_DIST['es']}% ES")
    print(f"  Sentiments:  {SENTIMENT_DIST}")
    print(f"  Delay:       {delay}s between calls (backoff: {backoff_factor}x)")
    print(f"  Checkpoint:  Every {checkpoint_interval} items")
    print(f"  Max retries: {max_retries} per item")
    print(f"  Auto-pause:  After {max_consecutive_failures} consecutive failures")
    print("-" * 70)
    
    # Estimates
    est_time = estimate_time(total, delay)
    est_cost = estimate_cost(total)
    print(f"  Est. time:   {format_duration(est_time)} (ETA: {format_eta(est_time)})")
    print(f"  Est. cost:   ${est_cost:.2f} USD")
    print("=" * 70)
    
    if dry_run:
        print("\n[DRY RUN] Showing generation plan...\n")
        plan = create_generation_plan(total)
        
        # Show first 10 items
        print("First 10 items:")
        for i, p in enumerate(plan[:10]):
            print(f"  {i+1:3d}. [{p['language'].upper()}] {p['intent'][:30]:30s} {p['sentiment']:10s} {p['complexity']}")
        
        if total > 10:
            print(f"  ... and {total - 10} more")
        
        # Distribution summary
        lang_counts = {"en": 0, "es": 0}
        sent_counts = {"positive": 0, "neutral": 0, "negative": 0}
        for p in plan:
            lang_counts[p["language"]] = lang_counts.get(p["language"], 0) + 1
            sent_counts[p["sentiment"]] = sent_counts.get(p["sentiment"], 0) + 1
        
        print(f"\nDistribution:")
        print(f"  Languages:  {lang_counts}")
        print(f"  Sentiments: {sent_counts}")
        print(f"  Intents:    {min(total, 77)} unique (cycles through 77)")
        
        return 0
    
    # Check for checkpoint
    conversations = []
    start_index = 0
    failed_indices = []
    start_time = time.time()
    
    if resume:
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            conversations = checkpoint["conversations"]
            start_index = checkpoint["current_index"]
            failed_indices = checkpoint.get("failed_indices", [])
            print(f"\n[RESUME] Loaded checkpoint at index {start_index}")
            print(f"         {len(conversations)} conversations already generated")
            print(f"         {len(failed_indices)} previous failures")
        else:
            print("\n[INFO] No checkpoint found, starting fresh")
    
    # Create generation plan
    plan = create_generation_plan(total)
    
    # Initialize generator
    print("\nInitializing generator...")
    try:
        generator = CustomerServiceGenerator.from_config()
        print(f"Generator ready with {generator.intent_count} intents\n")
    except Exception as e:
        print(f"[ERROR] Failed to initialize generator: {e}")
        return 1
    
    # Generate
    print("-" * 70)
    print("GENERATING...")
    print("-" * 70)
    
    consecutive_failures = 0
    current_delay = delay
    
    for i in range(start_index, total):
        params = plan[i]
        lang_label = params["language"].upper()
        
        # Progress info
        progress = (i + 1) / total * 100
        remaining = total - i - 1
        eta_seconds = remaining * (current_delay + 1)
        
        print(f"  [{i+1:4d}/{total}] ({progress:5.1f}%) [{lang_label}] {params['intent'][:25]:25s} ", end="", flush=True)
        
        # Retry loop with exponential backoff
        success = False
        retry_count = 0
        
        while retry_count < max_retries and not success:
            try:
                conv = generator.generate_single(
                    intent=params["intent"],
                    sentiment=params["sentiment"],
                    complexity=params["complexity"],
                    language=params["language"]
                )
                
                # Validate immediately
                is_valid, error = validate_conversation(conv)
                
                if is_valid:
                    conversations.append(conv)
                    print(f"[OK] (ETA: {format_eta(eta_seconds)})")
                    success = True
                    consecutive_failures = 0
                    current_delay = delay  # Reset delay on success
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"[INVALID: {error}] Retry {retry_count}...", end="", flush=True)
                        time.sleep(current_delay)
                    else:
                        failed_indices.append(i)
                        print(f"[INVALID] {error}")
                        consecutive_failures += 1
                    
            except Exception as e:
                error_str = str(e)
                is_throttling = "Throttling" in error_str or "throttl" in error_str.lower()
                
                retry_count += 1
                
                if is_throttling:
                    # Exponential backoff for throttling
                    current_delay = min(current_delay * backoff_factor, 60)  # Cap at 60s
                    if retry_count < max_retries:
                        print(f"[THROTTLED] Backoff {current_delay:.0f}s, retry {retry_count}...", end="", flush=True)
                        time.sleep(current_delay)
                    else:
                        failed_indices.append(i)
                        print(f"[THROTTLED] Max retries exceeded")
                        consecutive_failures += 1
                else:
                    if retry_count < max_retries:
                        print(f"[ERROR] Retry {retry_count}...", end="", flush=True)
                        time.sleep(current_delay)
                    else:
                        failed_indices.append(i)
                        print(f"[ERROR] {error_str[:50]}")
                        consecutive_failures += 1
        
        # Check for auto-pause
        if consecutive_failures >= max_consecutive_failures:
            print(f"\n[AUTO-PAUSE] {consecutive_failures} consecutive failures!")
            print(f"             Saving checkpoint and pausing...")
            save_checkpoint(checkpoint_path, conversations, i + 1, failed_indices, start_time, total, config)
            print(f"             Resume with: uv run python scripts/generate_data.py --total {total} --resume")
            print(f"\n             Waiting 60 seconds before continuing...")
            time.sleep(60)
            consecutive_failures = 0
            current_delay = delay * 2  # Increase base delay after pause
            print(f"             Resuming with delay={current_delay}s...")
        
        # Checkpoint every N items
        if (i + 1) % checkpoint_interval == 0:
            save_checkpoint(checkpoint_path, conversations, i + 1, failed_indices, start_time, total, config)
            print(f"         [Checkpoint: {len(conversations)} items, {len(failed_indices)} failed]")
        
        # Delay before next (except last)
        if i < total - 1 and success:
            time.sleep(current_delay)
    
    # Calculate stats
    elapsed = time.time() - start_time
    success_count = len(conversations)
    fail_count = len(failed_indices)
    actual_cost = estimate_cost(success_count)
    
    # Analyze coverage
    unique_intents = set(c.get("intent") for c in conversations)
    unique_sentiments = set(c.get("sentiment") for c in conversations)
    language_counts = {}
    for c in conversations:
        lang = c.get("language", "unknown")
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    # Print results
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    
    print(f"\n  Success:      {success_count}/{total} ({success_count/total*100:.0f}%)")
    print(f"  Failed:       {fail_count}")
    print(f"  Time:         {format_duration(elapsed)}")
    print(f"  Avg per item: {elapsed/total:.1f}s")
    print(f"  Est. cost:    ${actual_cost:.2f} USD")
    
    print(f"\n  Intent coverage: {len(unique_intents)}/77 ({len(unique_intents)/77*100:.0f}%)")
    print(f"  Sentiments:      {unique_sentiments}")
    print(f"  Languages:       {language_counts}")
    
    # Save final output
    if conversations:
        output_path = get_output_path(success_count)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n  Saved to: {output_path}")
        
        # Clear checkpoint
        clear_checkpoint(checkpoint_path)
        print("  [OK] Checkpoint cleared")
        
        # Run quality validation
        print("\n" + "-" * 70)
        print("Running quality validation...")
        try:
            validator = QualityValidator()
            quality = validator.compute_overall_score(conversations)
            print(f"  Quality score: {quality.overall_quality_score:.1f}/100")
            
            # Run bias detection
            print("Running bias detection...")
            detector = BiasDetector()
            bias = detector.detect_bias(conversations)
            print(f"  Bias detected: {bias.bias_detected} (severity: {bias.bias_severity})")
            
            # Register in database
            print("\n" + "-" * 70)
            print("Registering in database...")
            registry = DatasetRegistry()
            
            dataset_id = registry.register_dataset(
                domain="customer_service",
                size=len(conversations),
                file_path=str(output_path),
                file_format="json",
                notes=f"{len(conversations)} conversations ({language_counts}), {len(unique_intents)}/77 intents"
            )
            
            registry.update_quality_metrics(dataset_id, quality)
            registry.update_bias_metrics(dataset_id, bias)
            
            print(f"  [OK] Registered as: {dataset_id}")
            
            # Save report
            report_path = output_path.with_suffix(".report.json")
            report = {
                "dataset_id": dataset_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "config": config,
                "stats": {
                    "total_target": total,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "elapsed_seconds": elapsed,
                    "estimated_cost_usd": actual_cost
                },
                "coverage": {
                    "intents": len(unique_intents),
                    "languages": language_counts,
                    "sentiments": list(unique_sentiments)
                },
                "quality": {
                    "overall": quality.overall_quality_score,
                    "completeness": quality.completeness_score,
                    "consistency": quality.consistency_score,
                    "realism": quality.realism_score,
                    "diversity": quality.diversity_score
                },
                "bias": {
                    "detected": bias.bias_detected,
                    "severity": bias.bias_severity
                }
            }
            
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"  [OK] Report saved: {report_path}")
            
        except Exception as e:
            print(f"  [WARNING] Validation/registration failed: {e}")
            print(f"            Data saved but not validated/registered")
        
        # Final summary
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"  Items:    {len(conversations)}")
        print(f"  Quality:  {quality.overall_quality_score:.1f}/100" if 'quality' in dir() else "  Quality:  N/A")
        print(f"  Intents:  {len(unique_intents)}/77")
        print(f"  Time:     {format_duration(elapsed)}")
        print(f"  Cost:     ${actual_cost:.2f}")
        print(f"  File:     {output_path}")
        print("=" * 70)
        
        return 0
    else:
        print("\n[ERROR] No conversations generated!")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Generate customer service conversations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test (10 items)
    uv run python scripts/generate_data.py --total 10 --delay 3
    
    # Standard run (100 items)
    uv run python scripts/generate_data.py --total 100
    
    # Overnight run (500 items)
    uv run python scripts/generate_data.py --total 500 --delay 5 --max-failures 10
    
    # Resume interrupted run
    uv run python scripts/generate_data.py --total 500 --resume
    
    # Dry run (show plan)
    uv run python scripts/generate_data.py --total 100 --dry-run
        """
    )
    
    parser.add_argument(
        "--total", "-n",
        type=int,
        default=DEFAULT_TOTAL,
        help=f"Total conversations to generate (default: {DEFAULT_TOTAL})"
    )
    
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Base delay between API calls in seconds (default: {DEFAULT_DELAY})"
    )
    
    parser.add_argument(
        "--checkpoint-interval", "-c",
        type=int,
        default=DEFAULT_CHECKPOINT_INTERVAL,
        help=f"Save checkpoint every N items (default: {DEFAULT_CHECKPOINT_INTERVAL})"
    )
    
    parser.add_argument(
        "--max-retries", "-r",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per item before skipping (default: {DEFAULT_MAX_RETRIES})"
    )
    
    parser.add_argument(
        "--backoff-factor", "-b",
        type=float,
        default=DEFAULT_BACKOFF_FACTOR,
        help=f"Multiply delay by this on throttling (default: {DEFAULT_BACKOFF_FACTOR})"
    )
    
    parser.add_argument(
        "--max-failures", "-f",
        type=int,
        default=DEFAULT_MAX_CONSECUTIVE_FAILURES,
        help=f"Pause after this many consecutive failures (default: {DEFAULT_MAX_CONSECUTIVE_FAILURES})"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show generation plan without actually generating"
    )
    
    args = parser.parse_args()
    
    return generate_data(
        total=args.total,
        delay=args.delay,
        checkpoint_interval=args.checkpoint_interval,
        max_retries=args.max_retries,
        backoff_factor=args.backoff_factor,
        max_consecutive_failures=args.max_failures,
        resume=args.resume,
        dry_run=args.dry_run
    )


if __name__ == "__main__":
    sys.exit(main())
