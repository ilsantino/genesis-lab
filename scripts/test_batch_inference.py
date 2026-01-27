#!/usr/bin/env python
"""
Test Batch Inference Script for Genesis Lab.

This script tests the AWS Bedrock batch inference workflow for generating
large-scale synthetic customer service conversations.

Features:
- Configurable total count and EN/ES split
- Prepares prompts using the generator
- Builds JSONL input file
- Uploads to S3 and submits batch job
- Polls for completion with timeout
- Downloads and processes results
- Validates quality and registers dataset

Usage:
    # Dry run (show plan without executing)
    uv run python scripts/test_batch_inference.py --total 1000 --dry-run
    
    # Run batch inference for 1000 items
    uv run python scripts/test_batch_inference.py --total 1000
    
    # Custom EN/ES split (70% English, 30% Spanish)
    uv run python scripts/test_batch_inference.py --total 1000 --en-percent 70
    
    # With custom timeout
    uv run python scripts/test_batch_inference.py --total 1000 --timeout 3600
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()


def estimate_time(count: int) -> int:
    """Estimate batch job time in minutes."""
    if count <= 100:
        return 5
    elif count <= 500:
        return 15
    elif count <= 1000:
        return 25
    elif count <= 2000:
        return 45
    else:
        return int(count / 1000 * 25)


def estimate_cost(count: int) -> float:
    """Estimate cost in USD."""
    return count * 0.003


def print_banner(title: str, char: str = "="):
    """Print a banner."""
    width = 70
    print(char * width)
    print(f"  {title}")
    print(char * width)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test AWS Bedrock batch inference for synthetic data generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Dry run (show plan only)
    uv run python scripts/test_batch_inference.py --total 1000 --dry-run
    
    # Generate 1000 conversations (50/50 EN/ES)
    uv run python scripts/test_batch_inference.py --total 1000
    
    # Generate 500 conversations (70% English)
    uv run python scripts/test_batch_inference.py --total 500 --en-percent 70
    
    # With custom polling interval
    uv run python scripts/test_batch_inference.py --total 1000 --poll-interval 60
        """
    )
    
    parser.add_argument(
        "--total", "-n",
        type=int,
        default=1000,
        help="Total conversations to generate (default: 1000)"
    )
    parser.add_argument(
        "--en-percent",
        type=int,
        default=50,
        help="Percentage of English conversations (default: 50)"
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Seconds between status checks (default: 30)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Maximum seconds to wait for completion (default: 7200 = 2 hours)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show plan without executing"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=os.getenv("S3_BUCKET", "genesis-lab-batch-inference"),
        help="S3 bucket for batch inference"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip quality validation after generation"
    )
    
    args = parser.parse_args()
    
    # Calculate split
    en_count = int(args.total * args.en_percent / 100)
    es_count = args.total - en_count
    
    # Estimates
    est_time = estimate_time(args.total)
    est_cost = estimate_cost(args.total)
    
    # Print plan
    print_banner("GENESIS LAB - BATCH INFERENCE TEST")
    print(f"""
  Configuration:
    Total conversations: {args.total:,}
    English:            {en_count:,} ({args.en_percent}%)
    Spanish:            {es_count:,} ({100 - args.en_percent}%)
    
  AWS Settings:
    S3 Bucket:          {args.s3_bucket}
    Poll Interval:      {args.poll_interval}s
    Timeout:            {args.timeout}s ({args.timeout // 60} min)
    
  Estimates:
    Time:               ~{est_time} min
    Cost:               ~${est_cost:.2f} USD
""")
    
    if args.dry_run:
        print("-" * 70)
        print("  DRY RUN - No batch job will be submitted")
        print("-" * 70)
        print("""
  Workflow Preview:
    1. Prepare prompts using CustomerServiceGenerator
    2. Build JSONL input file (Bedrock batch format)
    3. Upload to s3://{bucket}/batch-inference/input/
    4. Submit batch job to Bedrock
    5. Poll status every {poll}s until completion
    6. Download results from S3
    7. Process JSONL output
    8. Validate quality and detect bias
    9. Register dataset in database
    10. Save final JSON output
""".format(bucket=args.s3_bucket, poll=args.poll_interval))
        return 0
    
    # Confirm before proceeding
    print("-" * 70)
    response = input("  Proceed with batch inference? [y/N]: ").strip().lower()
    if response != 'y':
        print("  Cancelled.")
        return 0
    
    print()
    
    # Import backend modules
    try:
        from src.pipelines.customer_service_pipeline import CustomerServicePipeline, PipelineConfig
        from src.generation.generator import CustomerServiceGenerator
        from src.generation.batch_inference import (
            BatchInputBuilder,
            BatchJobManager,
            BatchResultProcessor
        )
        from src.validation.quality import QualityValidator
        from src.validation.bias import BiasDetector
        from src.registry.database import DatasetRegistry
        from src.utils.config.loader import get_config
    except ImportError as e:
        print(f"[ERROR] Failed to import backend modules: {e}")
        print("        Make sure you're running from the project root.")
        return 1
    
    # Start batch inference
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        # =====================================================================
        # STEP 1: Initialize components
        # =====================================================================
        print_banner("STEP 1/8: Initializing", "-")
        
        config = get_config()
        model_id = config.aws.get_model_id()
        
        generator = CustomerServiceGenerator.from_config()
        print(f"  Generator ready with {generator.intent_count} intents")
        print(f"  Model: {model_id}")
        
        # =====================================================================
        # STEP 2: Prepare prompts
        # =====================================================================
        print_banner("STEP 2/8: Preparing prompts", "-")
        
        # Split between EN and ES
        en_prompts = generator.prepare_batch_prompts(count=en_count, language="en")
        es_prompts = generator.prepare_batch_prompts(count=es_count, language="es")
        all_prompts = en_prompts + es_prompts
        
        print(f"  Prepared {len(all_prompts)} prompts")
        print(f"    - English: {len(en_prompts)}")
        print(f"    - Spanish: {len(es_prompts)}")
        
        # =====================================================================
        # STEP 3: Build JSONL input file
        # =====================================================================
        print_banner("STEP 3/8: Building input file", "-")
        
        builder = BatchInputBuilder(
            model_id=model_id,
            max_tokens=2000,
            temperature=0.7
        )
        
        for i, p in enumerate(all_prompts):
            builder.add_prompt(
                prompt=p["prompt"],
                system=p["system"],
                record_id=str(i + 1),
                metadata=p["metadata"]
            )
        
        # Save metadata map for later
        metadata_map = builder.get_metadata_map()
        
        # Save to local file
        local_input_path = f"data/batch/input_{timestamp}.jsonl"
        Path(local_input_path).parent.mkdir(parents=True, exist_ok=True)
        builder.save_to_jsonl(local_input_path)
        
        print(f"  Saved input to: {local_input_path}")
        print(f"  Records: {len(builder.records)}")
        
        # =====================================================================
        # STEP 4: Upload to S3 and submit job
        # =====================================================================
        print_banner("STEP 4/8: Uploading to S3", "-")
        
        manager = BatchJobManager(
            s3_bucket=args.s3_bucket,
            region=config.aws.region
        )
        
        s3_input_key = f"batch-inference/input/{timestamp}/input.jsonl"
        s3_output_prefix = f"batch-inference/output/{timestamp}/"
        
        input_s3_uri = manager.upload_input(local_input_path, s3_input_key)
        output_s3_uri = f"s3://{args.s3_bucket}/{s3_output_prefix}"
        
        print(f"  Input URI:  {input_s3_uri}")
        print(f"  Output URI: {output_s3_uri}")
        
        print_banner("STEP 5/8: Submitting batch job", "-")
        
        job_id = manager.create_job(
            input_s3_uri=input_s3_uri,
            output_s3_prefix=output_s3_uri,
            model_id=model_id,
            job_name=f"genesis-lab-{timestamp}"
        )
        
        print(f"  Job ID: {job_id}")
        print(f"  Status: Submitted")
        
        # =====================================================================
        # STEP 5: Poll for completion
        # =====================================================================
        print_banner("STEP 6/8: Waiting for completion", "-")
        print(f"  Polling every {args.poll_interval}s (timeout: {args.timeout}s)")
        print()
        
        def progress_callback(status):
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
            print(f"  [{elapsed_str}] Status: {status.status}, "
                  f"Processed: {status.processed_records}/{status.total_records}")
        
        final_status = manager.wait_for_completion(
            job_id=job_id,
            poll_interval=args.poll_interval,
            timeout=args.timeout,
            progress_callback=progress_callback
        )
        
        if final_status.status != "Completed":
            print(f"\n[ERROR] Batch job failed with status: {final_status.status}")
            print(f"        Message: {final_status.message}")
            return 1
        
        print(f"\n  Job completed!")
        print(f"  Processed: {final_status.processed_records}")
        print(f"  Failed: {final_status.failed_records}")
        
        # =====================================================================
        # STEP 6: Download results
        # =====================================================================
        print_banner("STEP 7/8: Downloading results", "-")
        
        local_output_dir = f"data/batch/output_{timestamp}"
        Path(local_output_dir).mkdir(parents=True, exist_ok=True)
        
        output_files = manager.download_results(
            output_s3_prefix=f"{s3_output_prefix}",
            local_dir=local_output_dir
        )
        
        print(f"  Downloaded {len(output_files)} output file(s)")
        for f in output_files:
            print(f"    - {f}")
        
        # =====================================================================
        # STEP 7: Process results
        # =====================================================================
        print_banner("STEP 8/8: Processing results", "-")
        
        processor = BatchResultProcessor()
        
        # Parse output files
        results = processor.parse_multiple_files(output_files)
        
        print(f"  Parsed {len(results)} results")
        
        # Extract conversations
        conversations = generator.process_batch_results(
            results=results,
            metadata_map=metadata_map
        )
        
        print(f"  Extracted {len(conversations)} conversations")
        
        # Save final output
        output_path = f"data/synthetic/customer_service_{args.total}_{timestamp}.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved to: {output_path}")
        
        # =====================================================================
        # Validation (optional)
        # =====================================================================
        if not args.skip_validation:
            print()
            print_banner("POST-PROCESSING: Validation", "-")
            
            # Quality validation
            validator = QualityValidator()
            quality_result = validator.validate(conversations)
            quality_score = quality_result.overall_score
            
            print(f"  Quality Score: {quality_score:.1f}/100")
            
            # Bias detection
            detector = BiasDetector()
            bias_result = detector.detect(conversations)
            
            print(f"  Bias Severity: {bias_result.bias_severity}")
            print(f"  Bias Detected: {bias_result.bias_detected}")
            
            # Register in database
            print()
            print("  Registering in database...")
            
            registry = DatasetRegistry()
            dataset_id = registry.create_dataset(
                name=f"customer_service_{args.total}",
                domain="customer_service",
                language="mixed" if args.en_percent not in [0, 100] else ("en" if args.en_percent == 100 else "es"),
                sample_count=len(conversations),
                file_path=output_path,
                quality_score=quality_score,
                bias_score=1.0 if bias_result.bias_severity == "none" else 0.5,
                metadata={
                    "generation_method": "batch_inference",
                    "en_count": en_count,
                    "es_count": es_count,
                    "batch_job_id": job_id,
                    "timestamp": timestamp
                }
            )
            
            print(f"  Registered as: {dataset_id}")
            
            # Save report
            report_path = output_path.replace(".json", ".report.json")
            report = {
                "dataset_id": dataset_id,
                "timestamp": timestamp,
                "generation": {
                    "method": "batch_inference",
                    "total": args.total,
                    "en_count": en_count,
                    "es_count": es_count,
                    "batch_job_id": job_id,
                    "duration_seconds": int(time.time() - start_time)
                },
                "quality": {
                    "overall": quality_score,
                    "completeness": quality_result.completeness,
                    "consistency": quality_result.consistency,
                    "diversity": quality_result.diversity
                },
                "bias": {
                    "detected": bias_result.bias_detected,
                    "severity": bias_result.bias_severity,
                    "recommendations": bias_result.recommendations[:5] if bias_result.recommendations else []
                }
            }
            
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            
            print(f"  Report saved: {report_path}")
        
        # =====================================================================
        # Summary
        # =====================================================================
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        print()
        print_banner("BATCH INFERENCE COMPLETE")
        print(f"""
  Results:
    Conversations:    {len(conversations):,}
    Duration:         {elapsed_str}
    Output File:      {output_path}
    
  Statistics:
    English:          {sum(1 for c in conversations if c.get('language') == 'en'):,}
    Spanish:          {sum(1 for c in conversations if c.get('language') == 'es'):,}
    Unique Intents:   {len(set(c.get('intent', 'unknown') for c in conversations))}
""")
        
        if not args.skip_validation:
            print(f"""  Quality:
    Score:            {quality_score:.1f}/100
    Bias Severity:    {bias_result.bias_severity}
    Dataset ID:       {dataset_id}
""")
        
        print("=" * 70)
        
        return 0
        
    except Exception as e:
        print(f"\n[ERROR] Batch inference failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
