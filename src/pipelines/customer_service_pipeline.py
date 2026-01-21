"""
Customer Service Pipeline - End-to-end workflow for conversation generation.

This pipeline orchestrates:
1. Data generation (using CustomerServiceGenerator)
2. Quality validation (using QualityValidator)
3. Bias detection (using BiasDetector)
4. Intent classification training (using IntentClassifier)
5. Dataset registration (using DatasetRegistry)

Usage:
    uv run python -m src.pipelines.customer_service_pipeline --count 100
    uv run python -m src.pipelines.customer_service_pipeline --count 100 --quality-threshold 80 --bias-threshold medium
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from src.generation.generator import CustomerServiceGenerator
from src.generation.batch_inference import (
    BatchInputBuilder,
    BatchJobManager,
    BatchResultProcessor,
    BatchJobStatus
)
# DatasetMetadata is defined in schemas but registry uses individual args
from src.registry.database import DatasetRegistry
from src.training.intent_classifier import IntentClassifier, compare_models
from src.utils.aws_client import BedrockClient, S3Client
from src.utils.config import get_config
from src.validation.bias import BiasDetector
from src.validation.quality import QualityValidator

# Use loguru for better logging
try:
    from loguru import logger
    LOGURU_AVAILABLE = True
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO"
    )
except ImportError:
    # Fallback to standard logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    logger = logging.getLogger(__name__)
    LOGURU_AVAILABLE = False


__all__ = ["CustomerServicePipeline", "PipelineConfig"]


# Type for bias severity threshold
BiasSeverityThreshold = Literal["none", "low", "medium", "high"]


class PipelineConfig:
    """Configuration for pipeline thresholds and settings."""
    
    def __init__(
        self,
        quality_threshold: float = 70.0,
        bias_threshold: BiasSeverityThreshold = "medium",
        accuracy_threshold: float = 0.70,
        use_nlp_sentiment: bool = False,
        use_batch_mode: bool = False,
        s3_bucket: Optional[str] = None,
        batch_role_arn: Optional[str] = None
    ):
        """
        Initialize pipeline configuration.
        
        Args:
            quality_threshold: Minimum quality score (0-100) for production readiness
            bias_threshold: Maximum acceptable bias severity ("none", "low", "medium", "high")
            accuracy_threshold: Minimum model accuracy for production readiness
            use_nlp_sentiment: Whether to use TextBlob for sentiment analysis
            use_batch_mode: Whether to use batch inference instead of real-time
            s3_bucket: S3 bucket for batch inference (required if use_batch_mode=True)
            batch_role_arn: IAM role ARN for Bedrock batch inference (optional)
        """
        self.quality_threshold = quality_threshold
        self.bias_threshold = bias_threshold
        self.accuracy_threshold = accuracy_threshold
        self.use_nlp_sentiment = use_nlp_sentiment
        self.use_batch_mode = use_batch_mode
        self.s3_bucket = s3_bucket
        self.batch_role_arn = batch_role_arn
    
    @property
    def bias_severity_order(self) -> Dict[str, int]:
        """Get severity order for comparison."""
        return {"none": 0, "low": 1, "medium": 2, "high": 3}
    
    def is_bias_acceptable(self, severity: str) -> bool:
        """Check if bias severity is within acceptable threshold."""
        order = self.bias_severity_order
        return order.get(severity, 4) <= order.get(self.bias_threshold, 2)


class CustomerServicePipeline:
    """
    End-to-end pipeline for customer service data generation and training.
    
    Stages:
    1. generate: Create synthetic conversations using AWS Bedrock
    2. validate: Run quality validation (completeness, coherence, realism, diversity)
    3. detect_bias: Check for sentiment, intent, language imbalances
    4. train: Train intent classification models
    5. register: Save dataset metadata to SQLite registry
    
    Example:
        >>> config = PipelineConfig(quality_threshold=80, bias_threshold="low")
        >>> pipeline = CustomerServicePipeline(config=config)
        >>> results = pipeline.run(count=100, save_path="data/synthetic/cs_100.json")
        >>> print(f"Quality: {results['quality_score']:.1f}/100")
        >>> print(f"Production Ready: {results['production_ready']}")
    """
    
    def __init__(
        self,
        reference_path: str = "data/reference/customer_service_reference.json",
        registry_path: str = "data/registry.db",
        config: Optional[PipelineConfig] = None
    ):
        """
        Initialize pipeline components.
        
        Args:
            reference_path: Path to reference dataset for validation
            registry_path: Path to SQLite registry database
            config: Pipeline configuration with thresholds
        """
        self._reference_path = reference_path
        self._registry_path = registry_path
        self._config = config or PipelineConfig()
        
        # Components (lazy loaded)
        self._client: Optional[BedrockClient] = None
        self._generator: Optional[CustomerServiceGenerator] = None
        self._validator: Optional[QualityValidator] = None
        self._bias_detector: Optional[BiasDetector] = None
        self._registry: Optional[DatasetRegistry] = None
        
        # Results
        self._results: Dict[str, Any] = {}
    
    @property
    def config(self) -> PipelineConfig:
        """Get pipeline configuration."""
        return self._config
    
    @property
    def client(self) -> BedrockClient:
        """Get or create BedrockClient."""
        if self._client is None:
            self._client = BedrockClient.from_config()
        return self._client
    
    @property
    def generator(self) -> CustomerServiceGenerator:
        """Get or create CustomerServiceGenerator."""
        if self._generator is None:
            self._generator = CustomerServiceGenerator(
                client=self.client,
                domain="customer_service"
            )
        return self._generator
    
    @property
    def validator(self) -> QualityValidator:
        """Get or create QualityValidator."""
        if self._validator is None:
            self._validator = QualityValidator(reference_path=self._reference_path)
        return self._validator
    
    @property
    def bias_detector(self) -> BiasDetector:
        """Get or create BiasDetector."""
        if self._bias_detector is None:
            self._bias_detector = BiasDetector()
        return self._bias_detector
    
    @property
    def registry(self) -> DatasetRegistry:
        """Get or create DatasetRegistry."""
        if self._registry is None:
            self._registry = DatasetRegistry(db_path=self._registry_path)
        return self._registry
    
    # =========================================================================
    # STAGE 1: GENERATION
    # =========================================================================
    
    def generate(
        self,
        count: int,
        batch_size: int = 5,
        delay_between_batches: float = 2.0,
        language: str = "en"
    ) -> List[Dict]:
        """
        Generate synthetic conversations.
        
        Args:
            count: Number of conversations to generate
            batch_size: Number of conversations per batch (used for progress display)
            delay_between_batches: Seconds to wait between batches (rate limiting)
            language: Target language ("en" or "es")
        
        Returns:
            List of generated conversation dictionaries
        """
        print("\n" + "=" * 60)
        print("  STAGE 1: GENERATION")
        print("=" * 60)
        
        start_time = time.time()
        
        print(f"\n  Target: {count} conversations")
        print(f"  Language: {language}")
        
        # The generator handles batching internally
        conversations = self.generator.generate_batch(
            count=count,
            language=language,
            continue_on_error=True
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  Generated: {len(conversations)} conversations")
        print(f"  Time: {elapsed:.1f}s ({elapsed/max(1,len(conversations)):.2f}s per item)")
        
        self._results["generation"] = {
            "count": len(conversations),
            "elapsed_seconds": elapsed,
            "success_rate": len(conversations) / count if count > 0 else 0
        }
        
        return conversations
    
    def generate_batch_mode(
        self,
        count: int,
        language: str = "en",
        poll_interval: int = 30,
        timeout: int = 7200
    ) -> List[Dict]:
        """
        Generate synthetic conversations using batch inference.
        
        This method uses AWS Bedrock batch inference instead of real-time API calls,
        which avoids throttling issues and is more efficient for large datasets.
        
        Args:
            count: Number of conversations to generate
            language: Target language ("en" or "es")
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait for completion
        
        Returns:
            List of generated conversation dictionaries
        """
        print("\n" + "=" * 60)
        print("  STAGE 1: GENERATION (BATCH MODE)")
        print("=" * 60)
        
        start_time = time.time()
        
        # Get S3 bucket
        s3_bucket = self._config.s3_bucket
        if not s3_bucket:
            config = get_config()
            s3_bucket = config.aws.s3_bucket
        
        if not s3_bucket:
            raise ValueError(
                "No S3 bucket configured for batch inference. "
                "Set S3_BUCKET environment variable or pass s3_bucket to PipelineConfig."
            )
        
        print(f"\n  Target: {count} conversations")
        print(f"  Language: {language}")
        print(f"  S3 Bucket: {s3_bucket}")
        print(f"  Mode: Batch Inference")
        
        # Step 1: Prepare prompts
        print("\n  [1/5] Preparing prompts...")
        prompts = self.generator.prepare_batch_prompts(
            count=count,
            language=language
        )
        print(f"        Prepared {len(prompts)} prompts")
        
        # Step 2: Build input file
        print("\n  [2/5] Building batch input file...")
        config = get_config()
        model_id = config.aws.get_model_id()
        
        builder = BatchInputBuilder(
            model_id=model_id,
            max_tokens=2000,
            temperature=0.7
        )
        
        for i, p in enumerate(prompts):
            builder.add_prompt(
                prompt=p["prompt"],
                system=p["system"],
                record_id=str(i + 1),
                metadata=p["metadata"]
            )
        
        # Save metadata map for later
        metadata_map = builder.get_metadata_map()
        
        # Save to local file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_input_path = f"data/batch/input_{timestamp}.jsonl"
        Path(local_input_path).parent.mkdir(parents=True, exist_ok=True)
        builder.save_to_jsonl(local_input_path)
        print(f"        Saved input to {local_input_path}")
        
        # Step 3: Upload to S3 and submit job
        print("\n  [3/5] Uploading to S3 and submitting job...")
        
        manager = BatchJobManager(
            s3_bucket=s3_bucket,
            region=config.aws.region,
            role_arn=self._config.batch_role_arn
        )
        
        s3_input_key = f"batch-inference/input/{timestamp}/input.jsonl"
        s3_output_prefix = f"batch-inference/output/{timestamp}/"
        
        input_s3_uri = manager.upload_input(local_input_path, s3_input_key)
        output_s3_uri = f"s3://{s3_bucket}/{s3_output_prefix}"
        
        job_id = manager.create_job(
            input_s3_uri=input_s3_uri,
            output_s3_prefix=output_s3_uri,
            model_id=model_id,
            job_name=f"genesis-batch-{timestamp}"
        )
        print(f"        Job submitted: {job_id}")
        
        # Step 4: Wait for completion
        print("\n  [4/5] Waiting for job completion...")
        
        def progress_callback(status: BatchJobStatus):
            pct = (status.processed_records / max(1, status.total_records)) * 100
            print(
                f"        Status: {status.status} - "
                f"{status.processed_records}/{status.total_records} ({pct:.1f}%)"
            )
        
        try:
            final_status = manager.wait_for_completion(
                job_id=job_id,
                poll_interval=poll_interval,
                timeout=timeout,
                progress_callback=progress_callback
            )
            print(f"        Job completed: {final_status.processed_records} processed")
        except TimeoutError as e:
            print(f"        [ERROR] {e}")
            raise
        except RuntimeError as e:
            print(f"        [ERROR] Job failed: {e}")
            raise
        
        # Step 5: Download and process results
        print("\n  [5/5] Downloading and processing results...")
        
        local_output_dir = f"data/batch/output_{timestamp}"
        downloaded_files = manager.download_results(
            output_s3_prefix=output_s3_uri,
            local_dir=local_output_dir
        )
        print(f"        Downloaded {len(downloaded_files)} files")
        
        # Process results
        processor = BatchResultProcessor()
        all_results = processor.parse_multiple_files([str(f) for f in downloaded_files])
        
        # Convert to conversations
        conversations = self.generator.process_batch_results(
            results=all_results,
            metadata_map=metadata_map
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n  Generated: {len(conversations)} conversations")
        print(f"  Failed: {processor.error_count} errors")
        print(f"  Time: {elapsed:.1f}s ({elapsed/max(1,len(conversations)):.2f}s per item)")
        
        self._results["generation"] = {
            "count": len(conversations),
            "elapsed_seconds": elapsed,
            "success_rate": len(conversations) / count if count > 0 else 0,
            "batch_mode": True,
            "job_id": job_id,
            "errors": processor.error_count
        }
        
        return conversations
    
    # =========================================================================
    # STAGE 2: QUALITY VALIDATION
    # =========================================================================
    
    def validate(self, conversations: List[Dict]) -> Dict[str, Any]:
        """
        Run quality validation on generated conversations.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Quality metrics dictionary
        """
        print("\n" + "=" * 60)
        print("  STAGE 2: QUALITY VALIDATION")
        print("=" * 60)
        
        metrics = self.validator.compute_overall_score(conversations)
        
        print(f"\n  Completeness: {metrics.completeness_score:.2f}")
        print(f"  Consistency:  {metrics.consistency_score:.2f}")
        print(f"  Realism:      {metrics.realism_score:.2f}")
        print(f"  Diversity:    {metrics.diversity_score:.2f}")
        print(f"\n  OVERALL: {metrics.overall_quality_score:.1f}/100")
        
        if metrics.overall_quality_score >= 85:
            print("  [PASS] Quality meets production threshold")
        elif metrics.overall_quality_score >= 70:
            print("  [WARN] Quality is acceptable but could improve")
        else:
            print("  [FAIL] Quality below acceptable threshold")
        
        if metrics.issues_found:
            print(f"\n  Issues: {len(metrics.issues_found)}")
        if metrics.warnings:
            print(f"  Warnings: {len(metrics.warnings)}")
        
        self._results["quality"] = {
            "completeness": metrics.completeness_score,
            "consistency": metrics.consistency_score,
            "realism": metrics.realism_score,
            "diversity": metrics.diversity_score,
            "overall": metrics.overall_quality_score,
            "issues_count": len(metrics.issues_found),
            "warnings_count": len(metrics.warnings)
        }
        
        return self._results["quality"]
    
    # =========================================================================
    # STAGE 3: BIAS DETECTION
    # =========================================================================
    
    def detect_bias(self, conversations: List[Dict]) -> Dict[str, Any]:
        """
        Run bias detection on generated conversations.
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Bias metrics dictionary
        """
        print("\n" + "=" * 60)
        print("  STAGE 3: BIAS DETECTION")
        print("=" * 60)
        
        # Use NLP sentiment if configured
        metrics = self.bias_detector.detect_bias(
            conversations,
            use_nlp_sentiment=self._config.use_nlp_sentiment
        )
        
        sent = metrics.sentiment_distribution
        print(f"\n  Sentiment: P={sent.get('positive',0):.0%} N={sent.get('neutral',0):.0%} G={sent.get('negative',0):.0%}")
        
        if self._config.use_nlp_sentiment:
            print("  (Analyzed using TextBlob NLP)")
        
        intent_coverage = metrics.metadata.get("intent_coverage_pct", 0)
        print(f"  Intent Coverage: {intent_coverage:.1f}%")
        
        lang = metrics.demographic_balance.get("language", {})
        print(f"  Language: EN={lang.get('en',0):.0%} ES={lang.get('es',0):.0%}")
        
        # Check against threshold
        bias_acceptable = self._config.is_bias_acceptable(metrics.bias_severity)
        
        if metrics.bias_detected:
            print(f"\n  [BIAS DETECTED] Severity: {metrics.bias_severity.upper()}")
            if bias_acceptable:
                print(f"  [OK] Within acceptable threshold ({self._config.bias_threshold})")
            else:
                print(f"  [FAIL] Exceeds threshold ({self._config.bias_threshold})")
            for rec in metrics.recommendations[:3]:
                print(f"    - {rec[:60]}...")
        else:
            print("\n  [OK] No significant bias detected")
        
        self._results["bias"] = {
            "detected": metrics.bias_detected,
            "severity": metrics.bias_severity,
            "acceptable": bias_acceptable,
            "sentiment": dict(metrics.sentiment_distribution),
            "intent_coverage_pct": intent_coverage,
            "recommendations": metrics.recommendations,
            "underrepresented_intents": metrics.demographic_balance.get("underrepresented_intents", [])
        }
        
        return self._results["bias"]
    
    # =========================================================================
    # STAGE 4: TRAINING
    # =========================================================================
    
    def train(
        self,
        conversations: List[Dict],
        model_type: str = "logistic_regression",
        save_model: bool = True
    ) -> Dict[str, Any]:
        """
        Train intent classifier on generated conversations.
        
        Args:
            conversations: List of conversation dictionaries
            model_type: Model to use ("logistic_regression", "random_forest", "xgboost")
            save_model: Whether to save the trained model
        
        Returns:
            Training results dictionary
        """
        print("\n" + "=" * 60)
        print("  STAGE 4: INTENT CLASSIFICATION TRAINING")
        print("=" * 60)
        
        print(f"\n  Model: {model_type}")
        print(f"  Training samples: {len(conversations)}")
        
        # Compare models first
        print("\n  Comparing models...")
        comparison = compare_models(conversations, test_size=0.2)
        
        # Use best model
        valid_models = {k: v for k, v in comparison.items() if "error" not in v}
        if not valid_models:
            print("  [ERROR] All models failed!")
            self._results["training"] = {"error": "All models failed"}
            return self._results["training"]
        
        best_model = max(valid_models.keys(), key=lambda k: valid_models[k]["accuracy"])
        
        print(f"\n  Best model: {best_model}")
        
        # Train best model
        classifier = IntentClassifier(model_type=best_model)
        results = classifier.train(conversations, test_size=0.2, verbose=False)
        
        print(f"\n  Accuracy: {results['accuracy']:.1%}")
        print(f"  F1 (macro): {results['f1_macro']:.1%}")
        print(f"  F1 (weighted): {results['f1_weighted']:.1%}")
        
        if save_model:
            model_path = "models/trained/intent_classifier.pkl"
            classifier.save(model_path)
            print(f"\n  Model saved: {model_path}")
        
        self._results["training"] = {
            "model_type": best_model,
            "accuracy": results["accuracy"],
            "f1_macro": results["f1_macro"],
            "f1_weighted": results["f1_weighted"],
            "train_size": results["train_size"],
            "test_size": results["test_size"],
            "unique_intents": results["unique_intents"],
            "model_comparison": {k: v.get("accuracy", 0) for k, v in comparison.items()}
        }
        
        return self._results["training"]
    
    # =========================================================================
    # STAGE 5: REGISTRATION
    # =========================================================================
    
    def register(
        self,
        conversations: List[Dict],
        file_path: str,
        notes: Optional[str] = None
    ) -> str:
        """
        Register dataset in SQLite registry.
        
        Args:
            conversations: List of conversation dictionaries
            file_path: Path where dataset was saved
            notes: Optional notes about the dataset
        
        Returns:
            Dataset ID
        """
        print("\n" + "=" * 60)
        print("  STAGE 5: DATASET REGISTRATION")
        print("=" * 60)
        
        # Get results from previous stages
        quality = self._results.get("quality", {})
        bias = self._results.get("bias", {})
        training = self._results.get("training", {})
        
        # Register dataset
        dataset_id = self.registry.register_dataset(
            domain="customer_service",
            size=len(conversations),
            file_path=file_path,
            file_format="json",
            model_used="anthropic.claude-3-5-sonnet-20241022-v2:0",
            notes=notes
        )
        
        # Update quality metrics if available
        if quality:
            from src.generation.schemas import QualityMetrics
            qm = QualityMetrics(
                completeness_score=quality.get("completeness", 0),
                consistency_score=quality.get("consistency", 0),
                realism_score=quality.get("realism", 0),
                diversity_score=quality.get("diversity", 0),
                overall_quality_score=quality.get("overall", 0),
                issues_found=[],
                warnings=[],
                metadata={"source": "pipeline"}
            )
            self.registry.update_quality_metrics(dataset_id, qm)
        
        # Update bias info if available
        if bias:
            from src.generation.schemas import BiasMetrics
            bm = BiasMetrics(
                sentiment_distribution=bias.get("sentiment", {}),
                bias_detected=bias.get("detected", False),
                bias_severity=bias.get("severity", "none"),
                recommendations=bias.get("recommendations", [])
            )
            self.registry.update_bias_metrics(dataset_id, bm)
        
        # Register training run if available
        if training and "error" not in training:
            self.registry.register_training_run(
                dataset_id=dataset_id,
                model_type=training.get("model_type", "unknown"),
                results={
                    "accuracy": training.get("accuracy", 0),
                    "f1_score": training.get("f1_macro", 0),
                    "f1_weighted": training.get("f1_weighted", 0),
                    "model_path": "models/trained/intent_classifier.pkl"
                }
            )
        
        print(f"\n  Dataset ID: {dataset_id}")
        print(f"  File: {file_path}")
        print(f"  Size: {len(conversations)} conversations")
        print(f"  Quality: {quality.get('overall', 0):.1f}/100")
        
        self._results["registration"] = {
            "dataset_id": dataset_id,
            "file_path": file_path
        }
        
        return dataset_id
    
    # =========================================================================
    # FULL PIPELINE
    # =========================================================================
    
    def run(
        self,
        count: int,
        save_path: Optional[str] = None,
        batch_size: int = 5,
        delay_between_batches: float = 2.0,
        language: str = "en",
        notes: Optional[str] = None,
        save_combined_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run the full pipeline end-to-end.
        
        Args:
            count: Number of conversations to generate
            save_path: Path to save generated data (default: auto-generated)
            batch_size: Conversations per batch (for display purposes)
            delay_between_batches: Seconds between batches (for display purposes)
            language: Target language ("en" or "es")
            notes: Optional notes for registry
            save_combined_report: If True, save all reports in a single JSON file
        
        Returns:
            Complete results dictionary with all metrics and production_ready flag
        """
        print("\n")
        print("=" * 70)
        print("  CUSTOMER SERVICE PIPELINE - FULL RUN")
        print("=" * 70)
        
        # Print configuration
        print(f"\n  Configuration:")
        print(f"    Quality Threshold: {self._config.quality_threshold}")
        print(f"    Bias Threshold: {self._config.bias_threshold}")
        print(f"    Accuracy Threshold: {self._config.accuracy_threshold:.0%}")
        print(f"    Batch Mode: {self._config.use_batch_mode}")
        
        start_time = time.time()
        
        # Default save path
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"data/synthetic/customer_service_{count}_{timestamp}.json"
        
        # Stage 1: Generate (use batch mode if configured)
        if self._config.use_batch_mode:
            conversations = self.generate_batch_mode(
                count=count,
                language=language
            )
        else:
            conversations = self.generate(
                count=count,
                batch_size=batch_size,
                delay_between_batches=delay_between_batches,
                language=language
            )
        
        # Save data immediately after generation
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(conversations, f, indent=2, ensure_ascii=False)
        
        print(f"\n  Data saved to: {save_path}")
        
        # Stage 2: Validate
        self.validate(conversations)
        
        # Stage 3: Bias Detection
        self.detect_bias(conversations)
        
        # Stage 4: Training
        self.train(conversations)
        
        # Stage 5: Register
        self.register(conversations, save_path, notes)
        
        total_time = time.time() - start_time
        
        # Compute production_ready flag
        quality_score = self._results.get('quality', {}).get('overall', 0)
        bias_acceptable = self._results.get('bias', {}).get('acceptable', True)
        accuracy = self._results.get('training', {}).get('accuracy', 0)
        
        quality_ok = quality_score >= self._config.quality_threshold
        accuracy_ok = accuracy >= self._config.accuracy_threshold
        
        production_ready = quality_ok and bias_acceptable and accuracy_ok
        
        # Final Summary
        print("\n" + "=" * 70)
        print("  PIPELINE COMPLETE")
        print("=" * 70)
        
        print(f"\n  Total time: {total_time:.1f}s")
        print(f"  Conversations: {len(conversations)}")
        print(f"  Quality Score: {quality_score:.1f}/100 {'[OK]' if quality_ok else '[FAIL]'}")
        print(f"  Bias: {self._results.get('bias', {}).get('severity', 'none')} {'[OK]' if bias_acceptable else '[FAIL]'}")
        print(f"  Model Accuracy: {accuracy:.1%} {'[OK]' if accuracy_ok else '[FAIL]'}")
        print(f"  Dataset ID: {self._results.get('registration', {}).get('dataset_id', 'N/A')}")
        
        print("\n" + "-" * 50)
        if production_ready:
            print("  [PRODUCTION READY] All thresholds met!")
        else:
            print("  [NOT PRODUCTION READY] Some thresholds not met:")
            if not quality_ok:
                print(f"    - Quality: {quality_score:.1f} < {self._config.quality_threshold}")
            if not bias_acceptable:
                print(f"    - Bias: {self._results.get('bias', {}).get('severity', 'high')} > {self._config.bias_threshold}")
            if not accuracy_ok:
                print(f"    - Accuracy: {accuracy:.1%} < {self._config.accuracy_threshold:.0%}")
        print("-" * 50)
        
        print("\n" + "=" * 70)
        
        self._results["summary"] = {
            "total_time_seconds": total_time,
            "conversation_count": len(conversations),
            "save_path": save_path,
            "production_ready": production_ready,
            "thresholds": {
                "quality": self._config.quality_threshold,
                "bias": self._config.bias_threshold,
                "accuracy": self._config.accuracy_threshold
            },
            "checks": {
                "quality_ok": quality_ok,
                "bias_acceptable": bias_acceptable,
                "accuracy_ok": accuracy_ok
            }
        }
        
        # Save combined report if requested
        if save_combined_report:
            report_path = Path(save_path).with_suffix('.report.json')
            combined_report = {
                "metadata": {
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "data_path": save_path,
                    "conversation_count": len(conversations),
                    "production_ready": production_ready
                },
                "quality": self._results.get("quality", {}),
                "bias": self._results.get("bias", {}),
                "training": self._results.get("training", {}),
                "registration": self._results.get("registration", {}),
                "summary": self._results.get("summary", {})
            }
            
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(combined_report, f, indent=2, ensure_ascii=False)
            
            print(f"  Report saved to: {report_path}")
        
        return self._results
    
    def run_from_existing(self, data_path: str) -> Dict[str, Any]:
        """
        Run validation, bias detection, and training on existing data.
        
        Args:
            data_path: Path to existing JSON data file
        
        Returns:
            Results dictionary
        """
        print("\n")
        print("=" * 70)
        print("  CUSTOMER SERVICE PIPELINE - FROM EXISTING DATA")
        print("=" * 70)
        
        # Load data
        print(f"\n  Loading: {data_path}")
        with open(data_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)
        
        print(f"  Loaded: {len(conversations)} conversations")
        
        # Run stages
        self.validate(conversations)
        self.detect_bias(conversations)
        self.train(conversations)
        self.register(conversations, data_path, notes="Loaded from existing file")
        
        return self._results


def main():
    """Run the customer service pipeline from command line."""
    parser = argparse.ArgumentParser(
        description="Customer Service Pipeline - End-to-end synthetic data generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate 100 conversations with default thresholds
  uv run python -m src.pipelines.customer_service_pipeline --count 100

  # Generate with custom thresholds
  uv run python -m src.pipelines.customer_service_pipeline -c 100 --quality-threshold 80 --bias-threshold low

  # Run on existing data
  uv run python -m src.pipelines.customer_service_pipeline --existing data/synthetic/customer_service_100.json
        """
    )
    parser.add_argument(
        "--count", "-c", 
        type=int, 
        default=100,
        help="Number of conversations to generate (default: 100)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file path (default: auto-generated)"
    )
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=5,
        help="Batch size for generation (default: 5)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=2.0,
        help="Delay between batches in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--existing", "-e",
        type=str,
        default=None,
        help="Path to existing data file (skip generation)"
    )
    parser.add_argument(
        "--notes", "-n",
        type=str,
        default=None,
        help="Notes to add to registry entry"
    )
    parser.add_argument(
        "--language", "-l",
        type=str,
        choices=["en", "es"],
        default="en",
        help="Target language for generation (default: en)"
    )
    
    # Threshold arguments
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=70.0,
        help="Minimum quality score for production readiness (0-100, default: 70)"
    )
    parser.add_argument(
        "--bias-threshold",
        type=str,
        choices=["none", "low", "medium", "high"],
        default="medium",
        help="Maximum acceptable bias severity (default: medium)"
    )
    parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.70,
        help="Minimum model accuracy for production readiness (0-1, default: 0.70)"
    )
    parser.add_argument(
        "--use-nlp-sentiment",
        action="store_true",
        help="Use TextBlob for NLP-based sentiment analysis"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Don't save the combined report JSON file"
    )
    
    # Batch mode arguments
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use batch inference instead of real-time API calls (requires S3_BUCKET)"
    )
    parser.add_argument(
        "--s3-bucket",
        type=str,
        default=None,
        help="S3 bucket for batch inference (overrides S3_BUCKET env var)"
    )
    parser.add_argument(
        "--batch-role-arn",
        type=str,
        default=None,
        help="IAM role ARN for Bedrock batch inference"
    )
    
    args = parser.parse_args()
    
    # Get S3 bucket from args or environment
    s3_bucket = args.s3_bucket
    if not s3_bucket and args.batch_mode:
        import os
        s3_bucket = os.getenv("S3_BUCKET")
        if not s3_bucket:
            print("[ERROR] --batch-mode requires S3_BUCKET environment variable or --s3-bucket argument")
            return 1
    
    # Create config from arguments
    config = PipelineConfig(
        quality_threshold=args.quality_threshold,
        bias_threshold=args.bias_threshold,
        accuracy_threshold=args.accuracy_threshold,
        use_nlp_sentiment=args.use_nlp_sentiment,
        use_batch_mode=args.batch_mode,
        s3_bucket=s3_bucket,
        batch_role_arn=args.batch_role_arn
    )
    
    pipeline = CustomerServicePipeline(config=config)
    
    if args.existing:
        # Run on existing data
        results = pipeline.run_from_existing(args.existing)
    else:
        # Full pipeline with generation
        results = pipeline.run(
            count=args.count,
            save_path=args.output,
            batch_size=args.batch_size,
            delay_between_batches=args.delay,
            language=args.language,
            notes=args.notes,
            save_combined_report=not args.no_report
        )
    
    # Exit with appropriate code based on production_ready
    production_ready = results.get("summary", {}).get("production_ready", False)
    if production_ready:
        return 0
    else:
        # Still return 0 if quality is acceptable (backward compat)
        quality = results.get("quality", {}).get("overall", 0)
        return 0 if quality >= 70 else 1


if __name__ == "__main__":
    sys.exit(main())
