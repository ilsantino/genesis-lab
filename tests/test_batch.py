"""
Unit tests for Batch Inference module.

Tests BatchInputBuilder, BatchJobManager, and BatchResultProcessor.
Uses mocks to avoid AWS calls.
"""

import pytest
import json
import time
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime

from botocore.exceptions import ClientError

from src.generation.batch_inference import (
    BatchInputBuilder,
    BatchJobManager,
    BatchResultProcessor,
    BatchJobStatus
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def builder():
    """BatchInputBuilder with default config."""
    return BatchInputBuilder(
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        max_tokens=2000,
        temperature=0.7,
        top_p=0.9
    )


@pytest.fixture
def mock_boto3_clients():
    """Mock both bedrock and s3 clients."""
    with patch("src.generation.batch_inference.boto3.client") as mock:
        # Create separate mocks for bedrock and s3
        bedrock_mock = MagicMock()
        s3_mock = MagicMock()
        
        def client_factory(service_name, **kwargs):
            if service_name == "bedrock":
                return bedrock_mock
            elif service_name == "s3":
                return s3_mock
            return MagicMock()
        
        mock.side_effect = client_factory
        yield {"bedrock": bedrock_mock, "s3": s3_mock, "factory": mock}


@pytest.fixture
def sample_batch_output(tmp_path):
    """Create sample batch output JSONL file."""
    output_path = tmp_path / "batch_output.jsonl"
    
    records = [
        {
            "recordId": "1",
            "modelOutput": {
                "content": [{"text": '{"conversation_id": "conv_1", "intent": "billing"}'}]
            }
        },
        {
            "recordId": "2",
            "modelOutput": {
                "content": [{"text": '{"conversation_id": "conv_2", "intent": "support"}'}]
            }
        },
        {
            "recordId": "3",
            "error": "Throttling: Rate exceeded"
        }
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    return output_path


@pytest.fixture
def sample_batch_output_with_markdown(tmp_path):
    """Create sample batch output with markdown code blocks."""
    output_path = tmp_path / "batch_output_md.jsonl"
    
    records = [
        {
            "recordId": "1",
            "modelOutput": {
                "content": [{"text": '```json\n{"conversation_id": "conv_1"}\n```'}]
            }
        },
        {
            "recordId": "2",
            "modelOutput": {
                "content": [{"text": '```\n{"conversation_id": "conv_2"}\n```'}]
            }
        }
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    return output_path


# ============================================================================
# BATCH INPUT BUILDER TESTS
# ============================================================================

class TestBatchInputBuilder:
    """Tests for BatchInputBuilder class."""
    
    def test_add_prompt_basic(self, builder):
        """Should add prompt with auto-generated recordId."""
        record_id = builder.add_prompt("Generate a conversation about billing")
        
        assert record_id == "1"
        assert builder.count == 1
        assert len(builder.records) == 1
        
        record = builder.records[0]
        assert record["recordId"] == "1"
        assert "modelInput" in record
        assert record["modelInput"]["messages"][0]["content"] == "Generate a conversation about billing"
    
    def test_add_prompt_with_system(self, builder):
        """Should include system prompt in modelInput."""
        builder.add_prompt(
            prompt="Generate a conversation",
            system="You are a customer service expert."
        )
        
        record = builder.records[0]
        assert record["modelInput"]["system"] == "You are a customer service expert."
    
    def test_add_prompt_with_custom_record_id(self, builder):
        """Should use custom recordId when provided."""
        record_id = builder.add_prompt(
            prompt="Test prompt",
            record_id="custom_id_123"
        )
        
        assert record_id == "custom_id_123"
        assert builder.records[0]["recordId"] == "custom_id_123"
    
    def test_add_prompt_with_metadata(self, builder):
        """Should store metadata but not include in saved output."""
        builder.add_prompt(
            prompt="Test prompt",
            metadata={"intent": "billing", "language": "en"}
        )
        
        record = builder.records[0]
        assert "_metadata" in record
        assert record["_metadata"]["intent"] == "billing"
    
    def test_add_prompts_batch(self, builder):
        """Should add multiple prompts at once."""
        prompts = [
            {"prompt": "Prompt 1", "system": "System 1"},
            {"prompt": "Prompt 2", "metadata": {"id": 2}},
            {"prompt": "Prompt 3"}
        ]
        
        record_ids = builder.add_prompts_batch(prompts)
        
        assert len(record_ids) == 3
        assert record_ids == ["1", "2", "3"]
        assert builder.count == 3
    
    def test_save_to_jsonl(self, builder, tmp_path):
        """Should save records to JSONL file without metadata."""
        builder.add_prompt("Prompt 1", metadata={"internal": "data"})
        builder.add_prompt("Prompt 2")
        
        output_path = tmp_path / "test_input.jsonl"
        result_path = builder.save_to_jsonl(str(output_path))
        
        assert result_path == output_path
        assert output_path.exists()
        
        # Read and verify format
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        assert len(lines) == 2
        
        # Verify metadata is NOT in saved output
        record1 = json.loads(lines[0])
        assert "_metadata" not in record1
        assert "recordId" in record1
        assert "modelInput" in record1
    
    def test_get_metadata_map(self, builder):
        """Should return mapping of recordId to metadata."""
        builder.add_prompt("Prompt 1", metadata={"intent": "billing"})
        builder.add_prompt("Prompt 2", metadata={"intent": "support"})
        builder.add_prompt("Prompt 3")  # No metadata
        
        metadata_map = builder.get_metadata_map()
        
        assert metadata_map["1"]["intent"] == "billing"
        assert metadata_map["2"]["intent"] == "support"
        assert metadata_map["3"] == {}
    
    def test_clear(self, builder):
        """Should clear records and reset counter."""
        builder.add_prompt("Prompt 1")
        builder.add_prompt("Prompt 2")
        
        assert builder.count == 2
        
        builder.clear()
        
        assert builder.count == 0
        assert len(builder.records) == 0
        
        # Counter should be reset
        new_id = builder.add_prompt("New prompt")
        assert new_id == "1"
    
    def test_count_property(self, builder):
        """Should return correct count."""
        assert builder.count == 0
        
        builder.add_prompt("Prompt 1")
        assert builder.count == 1
        
        builder.add_prompt("Prompt 2")
        assert builder.count == 2
    
    def test_model_input_structure(self, builder):
        """Should create correct Anthropic format modelInput."""
        builder.add_prompt("Test prompt", system="System prompt")
        
        model_input = builder.records[0]["modelInput"]
        
        assert model_input["anthropic_version"] == "bedrock-2023-05-31"
        assert model_input["max_tokens"] == 2000
        assert model_input["temperature"] == 0.7
        assert model_input["top_p"] == 0.9
        assert model_input["messages"] == [{"role": "user", "content": "Test prompt"}]
        assert model_input["system"] == "System prompt"


# ============================================================================
# BATCH JOB MANAGER TESTS
# ============================================================================

class TestBatchJobManager:
    """Tests for BatchJobManager class - all AWS calls mocked."""
    
    def test_upload_input(self, mock_boto3_clients, tmp_path):
        """Should upload file to S3 and return URI."""
        # Create a test file
        test_file = tmp_path / "input.jsonl"
        test_file.write_text('{"test": "data"}')
        
        manager = BatchJobManager(
            s3_bucket="test-bucket",
            region="us-east-1"
        )
        
        s3_uri = manager.upload_input(str(test_file), "batch/input.jsonl")
        
        assert s3_uri == "s3://test-bucket/batch/input.jsonl"
        mock_boto3_clients["s3"].upload_file.assert_called_once_with(
            str(test_file), "test-bucket", "batch/input.jsonl"
        )
    
    def test_create_job_success(self, mock_boto3_clients):
        """Should create job and return job ARN."""
        mock_boto3_clients["bedrock"].create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789:job/test-job-id"
        }
        
        manager = BatchJobManager(
            s3_bucket="test-bucket",
            region="us-east-1"
        )
        
        job_arn = manager.create_job(
            input_s3_uri="s3://test-bucket/input.jsonl",
            output_s3_prefix="s3://test-bucket/output/",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        
        assert job_arn == "arn:aws:bedrock:us-east-1:123456789:job/test-job-id"
        mock_boto3_clients["bedrock"].create_model_invocation_job.assert_called_once()
    
    def test_create_job_with_role_arn(self, mock_boto3_clients):
        """Should include roleArn when provided."""
        mock_boto3_clients["bedrock"].create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789:job/test-job-id"
        }
        
        manager = BatchJobManager(
            s3_bucket="test-bucket",
            region="us-east-1",
            role_arn="arn:aws:iam::123456789:role/BedrockRole"
        )
        
        manager.create_job(
            input_s3_uri="s3://test-bucket/input.jsonl",
            output_s3_prefix="s3://test-bucket/output/",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0"
        )
        
        call_kwargs = mock_boto3_clients["bedrock"].create_model_invocation_job.call_args.kwargs
        assert call_kwargs["roleArn"] == "arn:aws:iam::123456789:role/BedrockRole"
    
    def test_create_job_with_custom_name(self, mock_boto3_clients):
        """Should use custom job name when provided."""
        mock_boto3_clients["bedrock"].create_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789:job/custom-job"
        }
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        manager.create_job(
            input_s3_uri="s3://test-bucket/input.jsonl",
            output_s3_prefix="s3://test-bucket/output/",
            model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
            job_name="my-custom-job-name"
        )
        
        call_kwargs = mock_boto3_clients["bedrock"].create_model_invocation_job.call_args.kwargs
        assert call_kwargs["jobName"] == "my-custom-job-name"
    
    def test_get_job_status(self, mock_boto3_clients):
        """Should return BatchJobStatus with correct fields."""
        mock_boto3_clients["bedrock"].get_model_invocation_job.return_value = {
            "jobArn": "arn:aws:bedrock:us-east-1:123456789:job/test-job",
            "status": "InProgress",
            "message": "Processing",
            "processedRecordCount": 50,
            "totalRecordCount": 100,
            "failedRecordCount": 2
        }
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        status = manager.get_job_status("test-job-id")
        
        assert isinstance(status, BatchJobStatus)
        assert status.status == "InProgress"
        assert status.processed_records == 50
        assert status.total_records == 100
        assert status.failed_records == 2
    
    def test_wait_for_completion_success(self, mock_boto3_clients):
        """Should return when job completes successfully."""
        # Simulate job progressing to completion
        mock_boto3_clients["bedrock"].get_model_invocation_job.side_effect = [
            {"jobArn": "job-1", "status": "InProgress", "processedRecordCount": 50, "totalRecordCount": 100, "failedRecordCount": 0},
            {"jobArn": "job-1", "status": "Completed", "processedRecordCount": 100, "totalRecordCount": 100, "failedRecordCount": 0}
        ]
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        with patch("src.generation.batch_inference.time.sleep"):
            status = manager.wait_for_completion("job-1", poll_interval=1)
        
        assert status.status == "Completed"
        assert status.processed_records == 100
    
    def test_wait_for_completion_failed(self, mock_boto3_clients):
        """Should raise RuntimeError when job fails."""
        mock_boto3_clients["bedrock"].get_model_invocation_job.return_value = {
            "jobArn": "job-1",
            "status": "Failed",
            "message": "Model error: Invalid input",
            "processedRecordCount": 0,
            "totalRecordCount": 100,
            "failedRecordCount": 100
        }
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        with pytest.raises(RuntimeError, match="Batch job failed"):
            manager.wait_for_completion("job-1")
    
    def test_wait_for_completion_timeout(self, mock_boto3_clients):
        """Should raise TimeoutError when timeout exceeded."""
        mock_boto3_clients["bedrock"].get_model_invocation_job.return_value = {
            "jobArn": "job-1",
            "status": "InProgress",
            "processedRecordCount": 10,
            "totalRecordCount": 100,
            "failedRecordCount": 0
        }
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        with patch("src.generation.batch_inference.time.sleep"):
            with patch("src.generation.batch_inference.time.time") as mock_time:
                # Simulate time passing beyond timeout
                mock_time.side_effect = [0, 0, 100, 200]  # Start, first check, after timeout
                
                with pytest.raises(TimeoutError, match="did not complete within"):
                    manager.wait_for_completion("job-1", timeout=50, poll_interval=1)
    
    def test_stop_job(self, mock_boto3_clients):
        """Should call stop_model_invocation_job."""
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        manager.stop_job("job-1")
        
        mock_boto3_clients["bedrock"].stop_model_invocation_job.assert_called_once_with(
            jobIdentifier="job-1"
        )
    
    def test_download_results(self, mock_boto3_clients, tmp_path):
        """Should download result files from S3."""
        # Mock S3 list_objects_v2
        mock_boto3_clients["s3"].list_objects_v2.return_value = {
            "Contents": [
                {"Key": "output/result1.jsonl"},
                {"Key": "output/result2.jsonl"}
            ]
        }
        
        manager = BatchJobManager(s3_bucket="test-bucket", region="us-east-1")
        
        downloaded = manager.download_results(
            output_s3_prefix="s3://test-bucket/output/",
            local_dir=str(tmp_path)
        )
        
        assert len(downloaded) == 2
        assert mock_boto3_clients["s3"].download_file.call_count == 2


# ============================================================================
# BATCH RESULT PROCESSOR TESTS
# ============================================================================

class TestBatchResultProcessor:
    """Tests for BatchResultProcessor class."""
    
    def test_parse_jsonl_output(self, sample_batch_output):
        """Should parse JSONL file and return results."""
        processor = BatchResultProcessor()
        
        results = processor.parse_jsonl_output(str(sample_batch_output))
        
        # 2 successful, 1 error
        assert len(results) == 2
        assert processor.error_count == 1
    
    def test_parse_record_with_error(self, sample_batch_output):
        """Should track records with errors separately."""
        processor = BatchResultProcessor()
        
        processor.parse_jsonl_output(str(sample_batch_output))
        
        assert len(processor.errors) == 1
        assert "Throttling" in str(processor.errors[0]["error"])
    
    def test_parse_extracts_generated_text(self, sample_batch_output):
        """Should extract generated_text from modelOutput."""
        processor = BatchResultProcessor()
        
        results = processor.parse_jsonl_output(str(sample_batch_output))
        
        assert results[0]["generated_text"] == '{"conversation_id": "conv_1", "intent": "billing"}'
        assert results[1]["generated_text"] == '{"conversation_id": "conv_2", "intent": "support"}'
    
    def test_extract_conversations(self, sample_batch_output):
        """Should extract JSON conversations from generated_text."""
        processor = BatchResultProcessor()
        
        results = processor.parse_jsonl_output(str(sample_batch_output))
        conversations = processor.extract_conversations(results)
        
        assert len(conversations) == 2
        assert conversations[0]["conversation_id"] == "conv_1"
        assert conversations[1]["conversation_id"] == "conv_2"
    
    def test_extract_conversations_from_markdown(self, sample_batch_output_with_markdown):
        """Should handle ```json code blocks in generated text."""
        processor = BatchResultProcessor()
        
        results = processor.parse_jsonl_output(str(sample_batch_output_with_markdown))
        conversations = processor.extract_conversations(results)
        
        assert len(conversations) == 2
        assert conversations[0]["conversation_id"] == "conv_1"
        assert conversations[1]["conversation_id"] == "conv_2"
    
    def test_extract_conversations_with_metadata(self, sample_batch_output):
        """Should merge metadata when provided."""
        processor = BatchResultProcessor()
        
        results = processor.parse_jsonl_output(str(sample_batch_output))
        
        metadata_map = {
            "1": {"language": "en", "complexity": "simple"},
            "2": {"language": "es", "complexity": "medium"}
        }
        
        conversations = processor.extract_conversations(results, metadata_map)
        
        assert conversations[0]["language"] == "en"
        assert conversations[1]["language"] == "es"
    
    def test_error_count_property(self, sample_batch_output):
        """Should return correct error count."""
        processor = BatchResultProcessor()
        
        assert processor.error_count == 0
        
        processor.parse_jsonl_output(str(sample_batch_output))
        
        assert processor.error_count == 1
    
    def test_get_error_summary(self, tmp_path):
        """Should return error summary grouped by type."""
        # Create file with multiple error types
        output_path = tmp_path / "errors.jsonl"
        records = [
            {"recordId": "1", "error": "Throttling: Rate exceeded"},
            {"recordId": "2", "error": "Throttling: Rate exceeded"},
            {"recordId": "3", "error": "ValidationError: Invalid input"},
            {"recordId": "4", "modelOutput": {"content": [{"text": "{}"}]}}
        ]
        
        with open(output_path, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        
        processor = BatchResultProcessor()
        processor.parse_jsonl_output(str(output_path))
        
        summary = processor.get_error_summary()
        
        assert "Throttling: Rate exceeded" in summary
        assert summary["Throttling: Rate exceeded"] == 2
        assert "ValidationError: Invalid input" in summary
        assert summary["ValidationError: Invalid input"] == 1
    
    def test_parse_multiple_files(self, tmp_path):
        """Should combine results from multiple files."""
        # Create two output files
        file1 = tmp_path / "output1.jsonl"
        file2 = tmp_path / "output2.jsonl"
        
        with open(file1, "w", encoding="utf-8") as f:
            f.write(json.dumps({"recordId": "1", "modelOutput": {"content": [{"text": "{}"}]}}) + "\n")
        
        with open(file2, "w", encoding="utf-8") as f:
            f.write(json.dumps({"recordId": "2", "modelOutput": {"content": [{"text": "{}"}]}}) + "\n")
            f.write(json.dumps({"recordId": "3", "modelOutput": {"content": [{"text": "{}"}]}}) + "\n")
        
        processor = BatchResultProcessor()
        results = processor.parse_multiple_files([str(file1), str(file2)])
        
        assert len(results) == 3
    
    def test_handles_malformed_json_in_generated_text(self, tmp_path):
        """Should handle invalid JSON in generated_text gracefully."""
        output_path = tmp_path / "malformed.jsonl"
        
        with open(output_path, "w", encoding="utf-8") as f:
            # Valid record structure but invalid JSON in generated text
            f.write(json.dumps({
                "recordId": "1",
                "modelOutput": {"content": [{"text": "not valid json {"}]}
            }) + "\n")
        
        processor = BatchResultProcessor()
        results = processor.parse_jsonl_output(str(output_path))
        
        # Should parse the record successfully
        assert len(results) == 1
        assert results[0]["generated_text"] == "not valid json {"
        
        # extract_conversations should handle the error
        conversations = processor.extract_conversations(results)
        assert len(conversations) == 0
        assert processor.error_count == 1  # Logged as error during extraction
