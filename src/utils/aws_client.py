"""
AWS Bedrock client with rate limiting, retry logic, and error handling.
"""

import time
import json
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from collections import deque

import boto3
from botocore.exceptions import ClientError

from .config import AWSConfig, GenerationConfig, get_config


__all__ = ["BedrockClient", "RateLimiter", "S3Client"]

logger = logging.getLogger(__name__)

# Bedrock errors that warrant retry
RETRYABLE_ERRORS = frozenset({
    "ThrottlingException",
    "TooManyRequestsException",
    "ServiceUnavailableException",
    "ModelStreamErrorException",
})


class RateLimiter:
    """Simple rate limiter using sliding window."""
    
    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.requests: deque = deque()
    
    def acquire(self) -> None:
        """Wait if necessary to respect rate limit."""
        while True:
            now = datetime.now()
            cutoff = now - timedelta(minutes=1)
            
            # Remove requests older than 1 minute
            while self.requests and self.requests[0] < cutoff:
                self.requests.popleft()
            
            # If under limit, record and return
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return
            
            # At limit - wait until oldest request expires
            sleep_until = self.requests[0] + timedelta(minutes=1)
            sleep_time = (sleep_until - now).total_seconds()
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping {sleep_time:.2f}s")
                time.sleep(sleep_time)


class BedrockClient:
    """
    AWS Bedrock client with production-ready features:
    - Rate limiting
    - Exponential backoff retry
    - Error handling
    - Request logging
    """
    
    def __init__(
        self,
        aws_config: AWSConfig,
        generation_config: GenerationConfig
    ):
        """
        Initialize Bedrock client.
        
        Args:
            aws_config: AWS configuration
            generation_config: Generation configuration (rate limits, retries)
        """
        self.aws_config = aws_config
        self.generation_config = generation_config
        
        # Initialize boto3 client
        session_kwargs: Dict[str, Any] = {
            "region_name": aws_config.region
        }
        
        # Add credentials if provided (otherwise uses default credential chain)
        if aws_config.access_key_id and aws_config.secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_config.access_key_id
            session_kwargs["aws_secret_access_key"] = aws_config.secret_access_key
        
        self.client = boto3.client("bedrock-runtime", **session_kwargs)
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            generation_config.max_requests_per_minute
        )
        
        # Metrics
        self.total_requests = 0
        self.total_retries = 0
        self.total_errors = 0
        
        logger.info(
            f"BedrockClient initialized: "
            f"region={aws_config.region}, "
            f"model={aws_config.default_model}, "
            f"rate_limit={generation_config.max_requests_per_minute}/min"
        )
    
    @classmethod
    def from_config(cls) -> "BedrockClient":
        """Create client from global configuration."""
        config = get_config()
        return cls(
            aws_config=config.aws,
            generation_config=config.generation
        )
    
    def invoke_model(
        self,
        prompt: str,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Invoke Bedrock model with retry logic.
        
        Args:
            prompt: User prompt
            model_name: Model identifier (uses default if None)
            temperature: Sampling temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            system_prompt: Optional system prompt
        
        Returns:
            Generated text
        
        Raises:
            RuntimeError: If all retry attempts fail
        """
        model_id = self.aws_config.get_model_id(model_name)
        
        # Prepare messages
        messages = [{"role": "user", "content": prompt}]
        
        # Build request body (Anthropic format for Claude models)
        body: Dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        if system_prompt:
            body["system"] = system_prompt
        
        # Retry loop with exponential backoff
        last_error: Optional[Exception] = None
        
        for attempt in range(self.generation_config.retry_attempts):
            try:
                # Respect rate limit
                self.rate_limiter.acquire()
                
                # Make request
                self.total_requests += 1
                response = self.client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(body)
                )
                
                # Parse response
                response_body = json.loads(response["body"].read())
                
                # Extract text from Anthropic response format
                if "content" in response_body and len(response_body["content"]) > 0:
                    generated_text = response_body["content"][0]["text"]
                    
                    logger.debug(
                        f"Generated {len(generated_text)} chars "
                        f"(attempt {attempt + 1})"
                    )
                    
                    return generated_text
                else:
                    raise RuntimeError("Unexpected response format from Bedrock")
            
            except ClientError as e:
                error_code = e.response.get("Error", {}).get("Code", "Unknown")
                error_message = e.response.get("Error", {}).get("Message", str(e))
                last_error = e
                self.total_errors += 1
                
                is_last_attempt = (attempt == self.generation_config.retry_attempts - 1)
                is_retryable = error_code in RETRYABLE_ERRORS
                
                if is_retryable and not is_last_attempt:
                    self.total_retries += 1
                    delay = self.generation_config.retry_delay_seconds * (2 ** attempt)
                    logger.warning(
                        f"{error_code}: Retrying in {delay:.1f}s "
                        f"(attempt {attempt + 1}/{self.generation_config.retry_attempts})"
                    )
                    time.sleep(delay)
                    continue
                
                # Non-retryable or last attempt
                logger.error(f"Bedrock error: {error_code} - {error_message}")
                raise RuntimeError(
                    f"Bedrock API error: {error_code} - {error_message}"
                ) from e
            
            except Exception as e:
                last_error = e
                self.total_errors += 1
                
                is_last_attempt = (attempt == self.generation_config.retry_attempts - 1)
                
                if is_last_attempt:
                    logger.error(f"Unexpected error: {e}")
                    raise RuntimeError(f"Unexpected error: {e}") from e
                
                self.total_retries += 1
                delay = self.generation_config.retry_delay_seconds * (2 ** attempt)
                logger.warning(f"Error: {e}. Retrying in {delay:.1f}s")
                time.sleep(delay)
        
        # Should never reach here, but satisfy type checker
        raise RuntimeError(f"All retries exhausted: {last_error}")
    
    def invoke_batch(
        self,
        prompts: List[str],
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        system_prompt: Optional[str] = None
    ) -> List[str]:
        """
        Invoke model for multiple prompts (sequential with rate limiting).
        
        Args:
            prompts: List of prompts
            model_name: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens per generation
            top_p: Nucleus sampling
            system_prompt: Optional system prompt
        
        Returns:
            List of generated texts (same order as prompts)
        """
        results: List[str] = []
        
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"Generating {i}/{len(prompts)}...")
            
            result = self.invoke_model(
                prompt=prompt,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                system_prompt=system_prompt
            )
            
            results.append(result)
        
        return results
    
    def get_metrics(self) -> Dict[str, int]:
        """Get client metrics."""
        return {
            "total_requests": self.total_requests,
            "total_retries": self.total_retries,
            "total_errors": self.total_errors
        }
    
    def get_bedrock_client(self) -> Any:
        """
        Get boto3 bedrock client (for job management).
        
        Note: This is different from bedrock-runtime (used for invoke_model).
        The bedrock client is used for batch job operations.
        
        Returns:
            boto3 bedrock client
        """
        session_kwargs: Dict[str, Any] = {
            "region_name": self.aws_config.region
        }
        
        if self.aws_config.access_key_id and self.aws_config.secret_access_key:
            session_kwargs["aws_access_key_id"] = self.aws_config.access_key_id
            session_kwargs["aws_secret_access_key"] = self.aws_config.secret_access_key
        
        return boto3.client("bedrock", **session_kwargs)
    
    def get_s3_client(self) -> Any:
        """
        Get boto3 S3 client.
        
        Returns:
            boto3 S3 client
        """
        session_kwargs: Dict[str, Any] = {
            "region_name": self.aws_config.region
        }
        
        if self.aws_config.access_key_id and self.aws_config.secret_access_key:
            session_kwargs["aws_access_key_id"] = self.aws_config.access_key_id
            session_kwargs["aws_secret_access_key"] = self.aws_config.secret_access_key
        
        return boto3.client("s3", **session_kwargs)
    
    @property
    def s3_bucket(self) -> Optional[str]:
        """Get configured S3 bucket."""
        return self.aws_config.s3_bucket


class S3Client:
    """
    Simple S3 client for batch inference file operations.
    
    Handles upload/download of batch input/output files.
    """
    
    def __init__(self, aws_config: AWSConfig):
        """
        Initialize S3 client.
        
        Args:
            aws_config: AWS configuration
        """
        self.aws_config = aws_config
        
        session_kwargs: Dict[str, Any] = {
            "region_name": aws_config.region
        }
        
        if aws_config.access_key_id and aws_config.secret_access_key:
            session_kwargs["aws_access_key_id"] = aws_config.access_key_id
            session_kwargs["aws_secret_access_key"] = aws_config.secret_access_key
        
        self.client = boto3.client("s3", **session_kwargs)
        self._bucket = aws_config.s3_bucket
    
    @classmethod
    def from_config(cls) -> "S3Client":
        """Create client from global configuration."""
        config = get_config()
        return cls(aws_config=config.aws)
    
    @property
    def bucket(self) -> Optional[str]:
        """Get configured S3 bucket."""
        return self._bucket
    
    def upload_file(
        self,
        local_path: str,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> str:
        """
        Upload a file to S3.
        
        Args:
            local_path: Local file path
            s3_key: S3 object key
            bucket: S3 bucket (uses configured bucket if None)
        
        Returns:
            S3 URI (s3://bucket/key)
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise ValueError("No S3 bucket configured. Set S3_BUCKET env var.")
        
        self.client.upload_file(local_path, bucket, s3_key)
        s3_uri = f"s3://{bucket}/{s3_key}"
        logger.info(f"Uploaded {local_path} to {s3_uri}")
        return s3_uri
    
    def download_file(
        self,
        s3_key: str,
        local_path: str,
        bucket: Optional[str] = None
    ) -> str:
        """
        Download a file from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Local file path
            bucket: S3 bucket (uses configured bucket if None)
        
        Returns:
            Local file path
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise ValueError("No S3 bucket configured. Set S3_BUCKET env var.")
        
        self.client.download_file(bucket, s3_key, local_path)
        logger.info(f"Downloaded s3://{bucket}/{s3_key} to {local_path}")
        return local_path
    
    def list_objects(
        self,
        prefix: str,
        bucket: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List objects in S3 with a prefix.
        
        Args:
            prefix: S3 key prefix
            bucket: S3 bucket (uses configured bucket if None)
        
        Returns:
            List of object metadata dictionaries
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise ValueError("No S3 bucket configured. Set S3_BUCKET env var.")
        
        response = self.client.list_objects_v2(Bucket=bucket, Prefix=prefix)
        return response.get("Contents", [])
    
    def delete_object(
        self,
        s3_key: str,
        bucket: Optional[str] = None
    ) -> None:
        """
        Delete an object from S3.
        
        Args:
            s3_key: S3 object key
            bucket: S3 bucket (uses configured bucket if None)
        """
        bucket = bucket or self._bucket
        if not bucket:
            raise ValueError("No S3 bucket configured. Set S3_BUCKET env var.")
        
        self.client.delete_object(Bucket=bucket, Key=s3_key)
        logger.info(f"Deleted s3://{bucket}/{s3_key}")