"""Test script for batch generation of customer service conversations."""
import time
from src.generation import CustomerServiceGenerator


def main():
    print("=" * 50)
    print("Batch Generation Test (10 conversations)")
    print("=" * 50)
    
    # Initialize generator
    g = CustomerServiceGenerator.from_config()
    print(f"Generator ready with {g.intent_count} intents\n")
    
    # Run batch generation
    start = time.time()
    conversations = g.generate_batch(count=10, continue_on_error=True)
    elapsed = time.time() - start
    
    # Results
    print(f"\n{'=' * 50}")
    print("RESULTS")
    print(f"{'=' * 50}")
    print(f"Generated: {len(conversations)} conversations")
    print(f"Time: {elapsed:.1f}s ({elapsed/max(len(conversations), 1):.1f}s avg)")
    print(f"Metrics: {g.get_metrics()}")
    
    # Sample output
    print(f"\n{'=' * 50}")
    print("SAMPLE CONVERSATIONS")
    print(f"{'=' * 50}")
    for i, c in enumerate(conversations[:5], 1):
        print(f"{i}. {c['intent']}")
        print(f"   Turns: {c['turn_count']}, Sentiment: {c['sentiment']}, Complexity: {c['complexity']}")
    
    print(f"\n✅ Batch generation test complete!")


if __name__ == "__main__":
    main()


