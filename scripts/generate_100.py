"""
Generate 100 customer service conversations with checkpointing.

Features:
- 50 English + 50 Spanish (bilingual)
- Balanced intent distribution (covers 77 Banking77 intents)
- Mixed sentiment: 30 positive, 50 neutral, 20 negative
- Mixed complexity: 30 simple, 50 medium, 20 complex
- Checkpointing every 10 items
- Real-time validation
- Auto-register in database

Estimated time: ~10 minutes (100 items × 5s delay)

Usage:
    uv run python scripts/generate_100.py
    uv run python scripts/generate_100.py --resume  # Resume from checkpoint
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.generation import CustomerServiceGenerator
from src.generation.templates.customer_service_prompts import ALL_INTENTS
from src.registry import DatasetRegistry
from src.validation import QualityValidator, BiasDetector


# =============================================================================
# CONFIGURATION
# =============================================================================

TOTAL_CONVERSATIONS = 100
DELAY_BETWEEN_CALLS = 5  # seconds

# Distribution targets
LANGUAGE_DIST = {"en": 50, "es": 50}
SENTIMENT_DIST = {"positive": 30, "neutral": 50, "negative": 20}
COMPLEXITY_DIST = {"simple": 30, "medium": 50, "complex": 20}

CHECKPOINT_INTERVAL = 10
CHECKPOINT_FILE = Path("data/synthetic/checkpoint_cs_100.json")
OUTPUT_FILE = Path("data/synthetic/customer_service_100.json")


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
    
    # Create distribution queues
    languages = (["en"] * LANGUAGE_DIST["en"]) + (["es"] * LANGUAGE_DIST["es"])
    sentiments = (
        ["positive"] * SENTIMENT_DIST["positive"] +
        ["neutral"] * SENTIMENT_DIST["neutral"] +
        ["negative"] * SENTIMENT_DIST["negative"]
    )
    complexities = (
        ["simple"] * COMPLEXITY_DIST["simple"] +
        ["medium"] * COMPLEXITY_DIST["medium"] +
        ["complex"] * COMPLEXITY_DIST["complex"]
    )
    
    # Shuffle for variety (but keep reproducible)
    import random
    random.seed(42)
    random.shuffle(languages)
    random.shuffle(sentiments)
    random.shuffle(complexities)
    
    for i in range(total):
        plan.append({
            "language": languages[i % len(languages)],
            "sentiment": sentiments[i % len(sentiments)],
            "complexity": complexities[i % len(complexities)],
            "intent": intent_cycle[i % len(intent_cycle)]
        })
    
    return plan


# =============================================================================
# CHECKPOINT MANAGEMENT
# =============================================================================

def save_checkpoint(
    conversations: List[Dict],
    current_index: int,
    failed_indices: List[int],
    start_time: float
) -> None:
    """Save current progress to checkpoint file."""
    checkpoint = {
        "conversations": conversations,
        "current_index": current_index,
        "failed_indices": failed_indices,
        "start_time": start_time,
        "checkpoint_time": datetime.now(timezone.utc).isoformat(),
        "total_target": TOTAL_CONVERSATIONS
    }
    
    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)


def load_checkpoint() -> Optional[Dict]:
    """Load checkpoint if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def clear_checkpoint() -> None:
    """Remove checkpoint file after successful completion."""
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()


# =============================================================================
# MAIN GENERATION
# =============================================================================

def generate_100(resume: bool = False) -> int:
    """
    Generate 100 customer service conversations.
    
    Args:
        resume: If True, resume from checkpoint
    
    Returns:
        Exit code (0 = success, 1 = failure)
    """
    print("=" * 60)
    print("Customer Service Dataset Generation (100 items)")
    print("=" * 60)
    print(f"Languages: {LANGUAGE_DIST}")
    print(f"Sentiments: {SENTIMENT_DIST}")
    print(f"Complexities: {COMPLEXITY_DIST}")
    print(f"Delay: {DELAY_BETWEEN_CALLS}s between calls")
    print(f"Checkpoint: Every {CHECKPOINT_INTERVAL} items")
    print("=" * 60)
    
    # Check for checkpoint
    conversations = []
    start_index = 0
    failed_indices = []
    start_time = time.time()
    
    if resume:
        checkpoint = load_checkpoint()
        if checkpoint:
            conversations = checkpoint["conversations"]
            start_index = checkpoint["current_index"]
            failed_indices = checkpoint.get("failed_indices", [])
            print(f"\n[RESUME] Loaded checkpoint at index {start_index}")
            print(f"         {len(conversations)} conversations already generated")
        else:
            print("\n[INFO] No checkpoint found, starting fresh")
    
    # Create generation plan
    plan = create_generation_plan(TOTAL_CONVERSATIONS)
    
    # Initialize generator
    print("\nInitializing generator...")
    generator = CustomerServiceGenerator.from_config()
    print(f"Generator ready with {generator.intent_count} intents\n")
    
    # Generate
    print("-" * 60)
    print("GENERATING...")
    print("-" * 60)
    
    for i in range(start_index, TOTAL_CONVERSATIONS):
        params = plan[i]
        lang_label = params["language"].upper()
        
        print(f"  [{i+1:3d}/{TOTAL_CONVERSATIONS}] [{lang_label}] {params['intent'][:25]:25s} ", end="", flush=True)
        
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
                print("[OK]")
            else:
                failed_indices.append(i)
                print(f"[INVALID] {error}")
                
        except Exception as e:
            failed_indices.append(i)
            error_type = "Throttled" if "Throttling" in str(e) else "Error"
            print(f"[FAIL] {error_type}")
        
        # Checkpoint every N items
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(conversations, i + 1, failed_indices, start_time)
            print(f"         [Checkpoint saved: {len(conversations)} items]")
        
        # Delay before next (except last)
        if i < TOTAL_CONVERSATIONS - 1:
            time.sleep(DELAY_BETWEEN_CALLS)
    
    # Calculate stats
    elapsed = time.time() - start_time
    success_count = len(conversations)
    fail_count = len(failed_indices)
    
    # Analyze coverage
    unique_intents = set(c.get("intent") for c in conversations)
    unique_sentiments = set(c.get("sentiment") for c in conversations)
    language_counts = {}
    for c in conversations:
        lang = c.get("language", "unknown")
        language_counts[lang] = language_counts.get(lang, 0) + 1
    
    # Print results
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE")
    print("=" * 60)
    
    print(f"\nSuccess: {success_count}/{TOTAL_CONVERSATIONS} ({success_count/TOTAL_CONVERSATIONS*100:.0f}%)")
    print(f"Failed: {fail_count}")
    print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Avg per item: {elapsed/TOTAL_CONVERSATIONS:.1f}s")
    
    print(f"\nIntent coverage: {len(unique_intents)}/77 ({len(unique_intents)/77*100:.0f}%)")
    print(f"Sentiments: {unique_sentiments}")
    print(f"Languages: {language_counts}")
    
    # Save final output
    if conversations:
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved to: {OUTPUT_FILE}")
        
        # Clear checkpoint
        clear_checkpoint()
        print("[OK] Checkpoint cleared")
        
        # Run quality validation
        print("\n" + "-" * 40)
        print("Running quality validation...")
        validator = QualityValidator()
        quality = validator.compute_overall_score(conversations)
        print(f"Quality score: {quality.overall_quality_score:.1f}/100")
        
        # Run bias detection
        print("Running bias detection...")
        detector = BiasDetector()
        bias = detector.detect_bias(conversations)
        print(f"Bias detected: {bias.bias_detected} (severity: {bias.bias_severity})")
        
        # Register in database
        print("\n" + "-" * 40)
        print("Registering in database...")
        registry = DatasetRegistry()
        
        dataset_id = registry.register_dataset(
            domain="customer_service",
            size=len(conversations),
            file_path=str(OUTPUT_FILE),
            file_format="json",
            notes=f"100 conversations (50 EN + 50 ES), {len(unique_intents)}/77 intents covered"
        )
        
        registry.update_quality_metrics(dataset_id, quality)
        registry.update_bias_metrics(dataset_id, bias)
        
        print(f"[OK] Registered as: {dataset_id}")
        
        # Final summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"  Dataset ID: {dataset_id}")
        print(f"  Items: {len(conversations)}")
        print(f"  Quality: {quality.overall_quality_score:.1f}/100")
        print(f"  Intents: {len(unique_intents)}/77")
        print(f"  File: {OUTPUT_FILE}")
        print("=" * 60)
        
        return 0
    else:
        print("\n[ERROR] No conversations generated!")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Generate 100 CS conversations")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if exists"
    )
    
    args = parser.parse_args()
    
    return generate_100(resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())

