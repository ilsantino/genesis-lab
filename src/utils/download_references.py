"""
Download reference datasets from HuggingFace for validation comparisons.

Datasets:
- Conversations: PolyAI/banking77 (customer service intents)
- Time-series: ETDataset/ett (electricity transformer temperature)

Compatible with datasets>=4.0.0
"""

from datasets import load_dataset
import json
from pathlib import Path
from typing import Optional
import sys
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import REFERENCE_DATA_DIR


# =============================================================================
# DATASET INFO (for documentation and logging)
# =============================================================================

DATASET_INFO = {
    "conversations": {
        "name": "PolyAI/banking77",
        "description": "Banking customer service intent classification",
        "size": "13,083 examples (77 intent categories)",
        "source": "https://huggingface.co/datasets/PolyAI/banking77",
        "license": "CC BY 4.0"
    },
    "timeseries": {
        "name": "LeoTungAnh/electricity_hourly",
        "description": "Electricity Transformer Temperature - hourly readings",
        "size": "~17,420 hourly observations (2 years)",
        "source": "https://huggingface.co/datasets/LeoTungAnh/electricity_hourly",
        "license": "Open RAIL"
    }
}


def print_dataset_info(domain: str) -> None:
    """Print dataset information for transparency."""
    info = DATASET_INFO.get(domain, {})
    if info:
        print(f"   📋 {info.get('name', 'Unknown')}")
        print(f"   📝 {info.get('description', '')}")
        print(f"   📊 {info.get('size', '')}")


# =============================================================================
# DOWNLOAD FUNCTIONS
# =============================================================================

def download_customer_service_reference(
    force: bool = False,
    sample_size: int = 500
) -> bool:
    """
    Download Banking77 dataset for customer service domain.
    
    Args:
        force: Re-download even if file exists
        sample_size: Number of samples to save
    
    Returns:
        True if successful, False otherwise
    """
    output_path = REFERENCE_DATA_DIR / "customer_service_reference.json"
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"⏭️  Customer service dataset already exists: {output_path}")
        print("   Use --force to re-download")
        return True
    
    print("📥 Downloading: PolyAI/banking77")
    print_dataset_info("conversations")
    
    try:
        dataset = load_dataset("PolyAI/banking77", split="train", trust_remote_code=True)
        
        # Efficient slicing with select()
        sample_size = min(sample_size, len(dataset))
        subset = dataset.select(range(sample_size))
        
        # Convert to our format
        samples = [
            {
                "text": item["text"],
                "label": item["label"],
                "intent": dataset.features["label"].int2str(item["label"])
            }
            for item in subset
        ]
        
        # Save with UTF-8 encoding
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)
        
        print(f"   ✅ Saved {len(samples)} examples → {output_path.name}")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False


def download_timeseries_reference(
    force: bool = False,
    num_series: int = 100,
    points_per_series: int = 500
) -> bool:
    """
    Download Electricity Hourly dataset for time-series domain.
    
    Uses 370 independent client electricity consumption series (Portugal 2012-2014).
    Each series represents a different household/client with independent consumption patterns.
    
    Args:
        force: Re-download even if file exists
        num_series: Number of time series to save (max 370)
        points_per_series: Data points per series to extract
    
    Returns:
        True if successful, False otherwise
    """
    output_path = REFERENCE_DATA_DIR / "timeseries_reference.json"
    
    # Check if already exists
    if output_path.exists() and not force:
        print(f"⏭️  Time-series dataset already exists: {output_path}")
        print("   Use --force to re-download")
        return True
    
    print("📥 Downloading: LeoTungAnh/electricity_hourly")
    print_dataset_info("timeseries")
    
    try:
        # Load electricity consumption dataset
        # This dataset has 370 independent series (one per client)
        dataset = load_dataset("LeoTungAnh/electricity_hourly", split="train")
        
        # Limit to requested number of series (max 370)
        num_series = min(num_series, len(dataset))
        subset = dataset.select(range(num_series))
        
        # Extract each client's time series
        samples = []
        
        for i, item in enumerate(subset):
            target = item["target"]
            start_time = str(item["start"]) if "start" in item else "2012-01-01T01:00:00"
            item_id = item.get("item_id", f"client_{i:03d}")
            
            # Safe slicing - get first N points
            actual_points = min(points_per_series, len(target))
            
            samples.append({
                "series_id": item_id,
                "values": list(target[:actual_points]),
                "length": actual_points,
                "frequency": "1H",
                "start": start_time,
                "source": "LeoTungAnh/electricity_hourly",
                "client_id": item_id,
                "description": f"Electricity consumption for Portuguese client {i+1}"
            })
        
        # Save
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2)
        
        print(f"   ✅ Saved {len(samples)} independent series → {output_path.name}")
        print(f"   📊 Each series represents a different client/household")
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        print(f"   💡 Tip: Ensure 'datasets' library is installed")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Download all reference datasets with CLI options."""
    parser = argparse.ArgumentParser(
        description="Download reference datasets for GENESIS-LAB validation"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--domain",
        choices=["all", "conversations", "timeseries"],
        default="all",
        help="Which domain to download (default: all)"
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Show dataset information and exit"
    )
    args = parser.parse_args()
    
    # Info mode
    if args.info:
        print("\n📚 GENESIS-LAB Reference Datasets\n")
        print("=" * 60)
        for domain, info in DATASET_INFO.items():
            print(f"\n🔹 {domain.upper()}")
            for key, value in info.items():
                print(f"   {key}: {value}")
        print("\n" + "=" * 60)
        return
    
    print("=" * 60)
    print("🧬 GENESIS-LAB Reference Dataset Downloader")
    print("=" * 60 + "\n")
    
    # Create reference directory
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output: {REFERENCE_DATA_DIR}\n")
    
    # Download based on domain selection
    results = {}
    
    if args.domain in ["all", "conversations"]:
        results["conversations"] = download_customer_service_reference(
            force=args.force
        )
    
    if args.domain in ["all", "timeseries"]:
        print()  # Spacing
        results["timeseries"] = download_timeseries_reference(
            force=args.force
        )
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Summary:")
    for domain, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {domain}: {status}")
    print("=" * 60)
    
    # Exit code for CI/CD
    if all(results.values()):
        print("\n🎉 All reference datasets ready!")
        sys.exit(0)
    else:
        print("\n⚠️  Some downloads failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()