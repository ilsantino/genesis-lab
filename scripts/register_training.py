"""Register training run in the dataset registry."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.registry.database import DatasetRegistry


def main():
    print("=" * 50)
    print("Registering Training Run")
    print("=" * 50)
    
    registry = DatasetRegistry()
    
    # Find the customer_service_100 dataset
    datasets = registry.list_datasets(domain="customer_service")
    
    if not datasets:
        print("\n[ERROR] No customer_service datasets found")
        return 1
    
    # Get the most recent (largest) dataset
    dataset = max(datasets, key=lambda d: d.get("size", 0))
    dataset_id = dataset["id"]
    
    print(f"\nDataset: {dataset_id}")
    print(f"Size: {dataset.get('size', 0)} items")
    
    # Register training run with actual metrics from training
    run_id = registry.register_training_run(
        dataset_id=dataset_id,
        model_type="tfidf_logistic_regression",
        results={
            "accuracy": 0.15,
            "f1_score": 0.1034,  # f1_macro
            "f1_weighted": 0.15,
            "train_size": 80,
            "test_size": 20,
            "unique_intents": 77,
            "model_path": "models/trained/intent_classifier.pkl",
            "notes": "Baseline model - low accuracy expected due to 77 classes with only 100 samples"
        }
    )
    
    print(f"\n[OK] Training run registered: {run_id}")
    
    # Show training history
    print("\n" + "-" * 40)
    print("Training History")
    print("-" * 40)
    
    history = registry.get_training_history(dataset_id)
    print(f"Total runs: {len(history)}")
    
    for run in history:
        model = run.get("model_type", "unknown")
        acc = run.get("accuracy") or 0
        f1 = run.get("f1_score") or 0
        print(f"  - {model}: accuracy={acc:.1%}, f1={f1:.1%}")
    
    print("\n" + "=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())

