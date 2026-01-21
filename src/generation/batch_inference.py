"""
Batch Inference module for AWS Bedrock.

This module provides classes for managing batch inference jobs:
- BatchInputBuilder: Prepares JSONL input from prompts
- BatchJobManager: Manages batch job lifecycle
- BatchResultProcessor: Parses batch output

Usage:
    from src.generation.batch_inference import BatchInputBuilder, BatchJobManager
    
    # Build input
    builder = BatchInputBuilder(model_id="anthropic.claude-3-5-sonnet...")
    builder.add_prompt("Generate a conversation...", system="You are...")
    input_path = builder.save_to_jsonl("batch_input.jsonl")
    
    # Submit job
    manager = BatchJobManager(s3_bucket="my-bucket", region="us-east-1")
    job_id = manager.create_job(input_s3_uri, output_s3_uri, model_id)
    manager.wait_for_completion(job_id)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import boto3
from botocore.exceptions import ClientError

__all__ = [
    "BatchInputBuilder",
    "BatchJobManager",
    "BatchResultProcessor",
    "BatchJobStatus",
]

logger = logging.getLogger(__name__)


@dataclass
class BatchJobStatus:
    """Status information for a batch inference job."""
    job_id: str
    status: str  # Submitted, InProgress, Completed, Failed, Stopping, Stopped
    message: Optional[str] = None
    input_data_config: Optional[Dict] = None
    output_data_config: Optional[Dict] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_records: int = 0
    processed_records: int = 0
    total_records: int = 0


class BatchInputBuilder:
    """
    Builds JSONL input files for Bedrock batch inference.
    
    Creates records in the format required by Bedrock batch API:
    {"recordId": "1", "modelInput": {...}}
    
    Example:
        >>> builder = BatchInputBuilder(
        ...     model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        ...     max_tokens=2000
        ... )
        >>> builder.add_prompt("Generate a conversation", system="You are an expert")
        >>> path = builder.save_to_jsonl("input.jsonl")
    """
    
    def __init__(
        self,
        model_id: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """
        Initialize the batch input builder.
        
        Args:
            model_id: Bedrock model ID
            max_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        """
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.records: List[Dict[str, Any]] = []
        self._record_counter = 0
    
    def add_prompt(
        self,
        prompt: str,
        system: Optional[str] = None,
        record_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Add a prompt to the batch.
        
        Args:
            prompt: User prompt text
            system: Optional system prompt
            record_id: Optional custom record ID (auto-generated if None)
            metadata: Optional metadata to include (stored but not sent to model)
        
        Returns:
            The record ID for this prompt
        """
        self._record_counter += 1
        rid = record_id or str(self._record_counter)
        
        # Build model input in Anthropic format
        model_input: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        if system:
            model_input["system"] = system
        
        record = {
            "recordId": rid,
            "modelInput": model_input
        }
        
        # Store metadata separately (for our use, not sent to Bedrock)
        if metadata:
            record["_metadata"] = metadata
        
        self.records.append(record)
        return rid
    
    def add_prompts_batch(
        self,
        prompts: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Add multiple prompts at once.
        
        Args:
            prompts: List of dicts with keys: prompt, system (optional), metadata (optional)
        
        Returns:
            List of record IDs
        """
        record_ids = []
        for p in prompts:
            rid = self.add_prompt(
                prompt=p["prompt"],
                system=p.get("system"),
                metadata=p.get("metadata")
            )
            record_ids.append(rid)
        return record_ids
    
    def save_to_jsonl(self, path: str) -> Path:
        """
        Save records to a JSONL file.
        
        Args:
            path: Output file path
        
        Returns:
            Path to the saved file
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for record in self.records:
                # Remove internal metadata before saving
                save_record = {k: v for k, v in record.items() if not k.startswith("_")}
                f.write(json.dumps(save_record, ensure_ascii=False) + "\n")
        
        logger.info(f"Saved {len(self.records)} records to {output_path}")
        return output_path
    
    def get_metadata_map(self) -> Dict[str, Dict]:
        """
        Get mapping of record IDs to their metadata.
        
        Returns:
            Dict mapping record_id -> metadata
        """
        return {
            r["recordId"]: r.get("_metadata", {})
            for r in self.records
        }
    
    def clear(self) -> None:
        """Clear all records."""
        self.records = []
        self._record_counter = 0
    
    @property
    def count(self) -> int:
        """Number of records in the batch."""
        return len(self.records)


class BatchJobManager:
    """
    Manages Bedrock batch inference job lifecycle.
    
    Handles job creation, status monitoring, and result retrieval.
    
    Example:
        >>> manager = BatchJobManager(
        ...     s3_bucket="my-bucket",
        ...     region="us-east-1"
        ... )
        >>> job_id = manager.create_job(
        ...     input_s3_uri="s3://bucket/input.jsonl",
        ...     output_s3_prefix="s3://bucket/output/",
        ...     model_id="anthropic.claude-3-5-sonnet..."
        ... )
        >>> status = manager.wait_for_completion(job_id)
    """
    
    def __init__(
        self,
        s3_bucket: str,
        region: str = "us-east-1",
        role_arn: Optional[str] = None
    ):
        """
        Initialize the batch job manager.
        
        Args:
            s3_bucket: S3 bucket for input/output
            region: AWS region
            role_arn: IAM role ARN for Bedrock to access S3 (optional)
        """
        self.s3_bucket = s3_bucket
        self.region = region
        self.role_arn = role_arn
        
        # Initialize clients
        self.bedrock_client = boto3.client("bedrock", region_name=region)
        self.s3_client = boto3.client("s3", region_name=region)
    
    def upload_input(self, local_path: str, s3_key: str) -> str:
        """
        Upload input file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key (without bucket)
        
        Returns:
            S3 URI (s3://bucket/key)
        """
        self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
        s3_uri = f"s3://{self.s3_bucket}/{s3_key}"
        logger.info(f"Uploaded {local_path} to {s3_uri}")
        return s3_uri
    
    def create_job(
        self,
        input_s3_uri: str,
        output_s3_prefix: str,
        model_id: str,
        job_name: Optional[str] = None
    ) -> str:
        """
        Create a batch inference job.
        
        Args:
            input_s3_uri: S3 URI of input JSONL file
            output_s3_prefix: S3 URI prefix for output
            model_id: Bedrock model ID
            job_name: Optional job name (auto-generated if None)
        
        Returns:
            Job ARN/ID
        """
        if job_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            job_name = f"genesis-batch-{timestamp}"
        
        # Build job configuration
        job_params: Dict[str, Any] = {
            "jobName": job_name,
            "modelId": model_id,
            "inputDataConfig": {
                "s3InputDataConfig": {
                    "s3Uri": input_s3_uri
                }
            },
            "outputDataConfig": {
                "s3OutputDataConfig": {
                    "s3Uri": output_s3_prefix
                }
            }
        }
        
        if self.role_arn:
            job_params["roleArn"] = self.role_arn
        
        try:
            response = self.bedrock_client.create_model_invocation_job(**job_params)
            job_arn = response["jobArn"]
            logger.info(f"Created batch job: {job_arn}")
            return job_arn
        except ClientError as e:
            logger.error(f"Failed to create batch job: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> BatchJobStatus:
        """
        Get current status of a batch job.
        
        Args:
            job_id: Job ARN or ID
        
        Returns:
            BatchJobStatus with current state
        """
        try:
            response = self.bedrock_client.get_model_invocation_job(jobIdentifier=job_id)
            
            return BatchJobStatus(
                job_id=response.get("jobArn", job_id),
                status=response.get("status", "Unknown"),
                message=response.get("message"),
                input_data_config=response.get("inputDataConfig"),
                output_data_config=response.get("outputDataConfig"),
                created_at=response.get("submitTime"),
                completed_at=response.get("endTime"),
                failed_records=response.get("failedRecordCount", 0),
                processed_records=response.get("processedRecordCount", 0),
                total_records=response.get("totalRecordCount", 0)
            )
        except ClientError as e:
            logger.error(f"Failed to get job status: {e}")
            raise
    
    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 7200,
        progress_callback: Optional[Callable[[BatchJobStatus], None]] = None
    ) -> BatchJobStatus:
        """
        Wait for a batch job to complete.
        
        Args:
            job_id: Job ARN or ID
            poll_interval: Seconds between status checks
            timeout: Maximum seconds to wait
            progress_callback: Optional callback for progress updates
        
        Returns:
            Final BatchJobStatus
        
        Raises:
            TimeoutError: If job doesn't complete within timeout
            RuntimeError: If job fails
        """
        start_time = time.time()
        terminal_states = {"Completed", "Failed", "Stopped"}
        
        while True:
            status = self.get_job_status(job_id)
            
            if progress_callback:
                progress_callback(status)
            else:
                logger.info(
                    f"Job {status.status}: "
                    f"{status.processed_records}/{status.total_records} processed"
                )
            
            if status.status in terminal_states:
                if status.status == "Failed":
                    raise RuntimeError(f"Batch job failed: {status.message}")
                return status
            
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job did not complete within {timeout}s")
            
            time.sleep(poll_interval)
    
    def download_results(
        self,
        output_s3_prefix: str,
        local_dir: str
    ) -> List[Path]:
        """
        Download batch job results from S3.
        
        Args:
            output_s3_prefix: S3 URI prefix where results are stored
            local_dir: Local directory to save results
        
        Returns:
            List of downloaded file paths
        """
        # Parse S3 prefix
        if output_s3_prefix.startswith("s3://"):
            parts = output_s3_prefix[5:].split("/", 1)
            bucket = parts[0]
            prefix = parts[1] if len(parts) > 1 else ""
        else:
            bucket = self.s3_bucket
            prefix = output_s3_prefix
        
        # List objects
        response = self.s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        downloaded = []
        local_path = Path(local_dir)
        local_path.mkdir(parents=True, exist_ok=True)
        
        for obj in response.get("Contents", []):
            key = obj["Key"]
            filename = Path(key).name
            local_file = local_path / filename
            
            self.s3_client.download_file(bucket, key, str(local_file))
            downloaded.append(local_file)
            logger.info(f"Downloaded {key} to {local_file}")
        
        return downloaded
    
    def stop_job(self, job_id: str) -> None:
        """
        Stop a running batch job.
        
        Args:
            job_id: Job ARN or ID
        """
        try:
            self.bedrock_client.stop_model_invocation_job(jobIdentifier=job_id)
            logger.info(f"Stopped job: {job_id}")
        except ClientError as e:
            logger.error(f"Failed to stop job: {e}")
            raise


class BatchResultProcessor:
    """
    Processes results from Bedrock batch inference.
    
    Parses JSONL output files and extracts generated content.
    
    Example:
        >>> processor = BatchResultProcessor()
        >>> results = processor.parse_jsonl_output("output.jsonl.out")
        >>> for r in results:
        ...     print(r["recordId"], r["generated_text"][:100])
    """
    
    def __init__(self):
        """Initialize the result processor."""
        self.results: List[Dict[str, Any]] = []
        self.errors: List[Dict[str, Any]] = []
    
    def parse_jsonl_output(self, path: str) -> List[Dict[str, Any]]:
        """
        Parse a JSONL output file from batch inference.
        
        Args:
            path: Path to output JSONL file
        
        Returns:
            List of parsed results with record_id and generated_text
        """
        self.results = []
        self.errors = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    record = json.loads(line)
                    parsed = self._parse_record(record)
                    if parsed.get("error"):
                        self.errors.append(parsed)
                    else:
                        self.results.append(parsed)
                except json.JSONDecodeError as e:
                    self.errors.append({
                        "error": f"JSON parse error: {e}",
                        "raw_line": line[:200]
                    })
        
        logger.info(
            f"Parsed {len(self.results)} results, {len(self.errors)} errors"
        )
        return self.results
    
    def _parse_record(self, record: Dict) -> Dict[str, Any]:
        """Parse a single output record."""
        result = {
            "record_id": record.get("recordId"),
            "generated_text": None,
            "error": None,
            "raw": record
        }
        
        # Check for error in record
        if "error" in record:
            result["error"] = record["error"]
            return result
        
        # Extract generated text from modelOutput
        model_output = record.get("modelOutput", {})
        
        # Anthropic format: content[0].text
        content = model_output.get("content", [])
        if content and isinstance(content, list) and len(content) > 0:
            result["generated_text"] = content[0].get("text", "")
        
        return result
    
    def parse_multiple_files(self, paths: List[str]) -> List[Dict[str, Any]]:
        """
        Parse multiple output files.
        
        Args:
            paths: List of file paths
        
        Returns:
            Combined list of results
        """
        all_results = []
        for path in paths:
            results = self.parse_jsonl_output(path)
            all_results.extend(results)
        return all_results
    
    def extract_conversations(
        self,
        results: List[Dict[str, Any]],
        metadata_map: Optional[Dict[str, Dict]] = None
    ) -> List[Dict[str, Any]]:
        """
        Extract conversation dictionaries from batch results.
        
        Parses the generated JSON from each result and combines with metadata.
        
        Args:
            results: Parsed batch results
            metadata_map: Optional mapping of record_id -> metadata
        
        Returns:
            List of conversation dictionaries
        """
        conversations = []
        
        for result in results:
            if result.get("error") or not result.get("generated_text"):
                continue
            
            try:
                # Parse generated JSON
                text = result["generated_text"]
                
                # Try to extract JSON from markdown code blocks
                if "```json" in text:
                    start = text.find("```json") + 7
                    end = text.find("```", start)
                    if end > start:
                        text = text[start:end].strip()
                elif "```" in text:
                    start = text.find("```") + 3
                    end = text.find("```", start)
                    if end > start:
                        text = text[start:end].strip()
                
                conversation = json.loads(text)
                
                # Add metadata if available
                record_id = result["record_id"]
                if metadata_map and record_id in metadata_map:
                    conversation.update(metadata_map[record_id])
                
                conversations.append(conversation)
                
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Failed to parse conversation from record {result.get('record_id')}: {e}"
                )
                self.errors.append({
                    "record_id": result.get("record_id"),
                    "error": f"JSON parse error: {e}",
                    "text_preview": result.get("generated_text", "")[:200]
                })
        
        logger.info(f"Extracted {len(conversations)} conversations")
        return conversations
    
    @property
    def error_count(self) -> int:
        """Number of errors encountered."""
        return len(self.errors)
    
    def get_error_summary(self) -> Dict[str, int]:
        """Get summary of errors by type."""
        summary: Dict[str, int] = {}
        for err in self.errors:
            error_type = str(err.get("error", "unknown"))[:50]
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary
