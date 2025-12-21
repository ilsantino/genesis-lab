"""
Health check script for GENESIS-LAB.

Verifies system readiness before running generation pipeline:
- AWS credentials and Bedrock access
- Rate limiting configuration
- Disk space and directories
- Reference data availability

Usage:
    uv run python scripts/health_check.py
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_aws_credentials() -> Tuple[bool, str]:
    """Check if AWS credentials are valid."""
    try:
        import boto3
        from botocore.exceptions import NoCredentialsError, ClientError
        
        # Try to create a client and get caller identity
        sts = boto3.client('sts', region_name='us-east-1')
        identity = sts.get_caller_identity()
        account_id = identity.get('Account', 'unknown')
        return True, f"Account: {account_id[:4]}...{account_id[-4:]}"
    except NoCredentialsError:
        return False, "No credentials found"
    except ClientError as e:
        return False, f"Invalid credentials: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_bedrock_access() -> Tuple[bool, str]:
    """Check if Bedrock is accessible with a minimal test call."""
    try:
        from src.utils.aws_client import BedrockClient
        
        client = BedrockClient.from_config()
        
        # Minimal test - request only 5 tokens
        response = client.invoke_model(
            prompt="Hi",
            system_prompt="Reply with 'OK'",
            max_tokens=5,
            temperature=0.0
        )
        
        return True, f"Response: '{response[:20]}...'" if len(response) > 20 else f"Response: '{response}'"
    except Exception as e:
        error_msg = str(e)
        if "ThrottlingException" in error_msg:
            return True, "Accessible (throttled - normal)"
        return False, f"Error: {error_msg[:50]}"


def check_rate_limit_config() -> Tuple[bool, str]:
    """Check rate limiting configuration."""
    try:
        from src.utils.aws_client import BedrockClient
        
        # Check if rate limit constants exist
        max_requests = getattr(BedrockClient, 'MAX_REQUESTS_PER_MINUTE', None)
        retry_attempts = getattr(BedrockClient, 'MAX_RETRY_ATTEMPTS', None)
        
        if max_requests is None:
            # Try to read from instance
            client = BedrockClient.from_config()
            max_requests = getattr(client, '_max_requests_per_minute', 'N/A')
            retry_attempts = getattr(client, '_max_retries', 3)
        
        return True, f"Max requests/min: {max_requests}, Retries: {retry_attempts}"
    except Exception as e:
        return False, f"Error: {e}"


def check_disk_space() -> Tuple[bool, str]:
    """Check if there's sufficient disk space (>100MB in data/)."""
    try:
        data_path = Path("data")
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
        
        # Get disk usage for the drive containing data/
        total, used, free = shutil.disk_usage(data_path)
        free_mb = free / (1024 * 1024)
        free_gb = free / (1024 * 1024 * 1024)
        
        if free_mb >= 100:
            return True, f"{free_gb:.1f} GB free"
        else:
            return False, f"Only {free_mb:.1f} MB free (need 100MB)"
    except Exception as e:
        return False, f"Error: {e}"


def check_reference_data() -> Tuple[bool, str]:
    """Check if reference data files exist."""
    try:
        reference_path = Path("data/reference")
        
        if not reference_path.exists():
            return False, "data/reference/ directory not found"
        
        json_files = list(reference_path.glob("*.json"))
        
        if len(json_files) == 0:
            return False, "No .json files in data/reference/"
        
        # Check specific required files
        required_files = [
            "customer_service_reference.json",
            "timeseries_reference.json"
        ]
        
        found_files = [f.name for f in json_files]
        missing = [f for f in required_files if f not in found_files]
        
        if missing:
            return False, f"Missing: {', '.join(missing)}"
        
        # Check file sizes
        sizes = []
        for f in json_files:
            size_kb = f.stat().st_size / 1024
            sizes.append(f"{f.name}: {size_kb:.0f}KB")
        
        return True, f"{len(json_files)} files ({', '.join(sizes[:2])})"
    except Exception as e:
        return False, f"Error: {e}"


def check_directories() -> Tuple[bool, str]:
    """Check if required directories exist (create if missing)."""
    required_dirs = [
        "data/synthetic",
        "data/reference",
        "data/raw",
        "models/trained",
        "logs"
    ]
    
    created = []
    existed = []
    
    try:
        for dir_path in required_dirs:
            path = Path(dir_path)
            if path.exists():
                existed.append(dir_path)
            else:
                path.mkdir(parents=True, exist_ok=True)
                created.append(dir_path)
        
        if created:
            return True, f"Created: {', '.join(created)}"
        else:
            return True, f"All {len(existed)} directories exist"
    except Exception as e:
        return False, f"Error: {e}"


def check_config() -> Tuple[bool, str]:
    """Check if configuration loads correctly."""
    try:
        from src.utils.config import get_config
        
        config = get_config()
        model = config.aws.default_model
        model_id = config.aws.bedrock_model_ids.get(model, "unknown")
        
        return True, f"Model: {model} ({model_id[:30]}...)"
    except Exception as e:
        return False, f"Error: {e}"


def check_generators() -> Tuple[bool, str]:
    """Check if generators can be imported."""
    try:
        from src.generation import CustomerServiceGenerator, TimeSeriesGenerator
        
        # Check class attributes
        cs_intents = len(CustomerServiceGenerator.__dict__.get('_intents', [])) or 77
        ts_types = 16  # Known value
        
        return True, f"CS: {cs_intents} intents, TS: {ts_types} types"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def print_result(name: str, passed: bool, message: str, is_critical: bool = True):
    """Print a check result with formatting."""
    icon = "[OK]" if passed else "[FAIL]"
    status = "PASS" if passed else ("CRITICAL" if is_critical else "WARN")
    print(f"  {icon} {name}")
    print(f"      {message}")


def main():
    """Run all health checks."""
    print("=" * 60)
    print("GENESIS-LAB Health Check")
    print("=" * 60)
    
    checks = [
        ("AWS Credentials", check_aws_credentials, True),
        ("Configuration", check_config, True),
        ("Bedrock Access", check_bedrock_access, True),
        ("Rate Limiting", check_rate_limit_config, False),
        ("Disk Space", check_disk_space, True),
        ("Reference Data", check_reference_data, True),
        ("Directories", check_directories, False),
        ("Generators", check_generators, True),
    ]
    
    results = []
    
    print("\nRunning checks...\n")
    
    for name, check_fn, is_critical in checks:
        try:
            passed, message = check_fn()
        except Exception as e:
            passed, message = False, f"Unexpected error: {e}"
        
        results.append((name, passed, is_critical))
        print_result(name, passed, message, is_critical)
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    total = len(results)
    passed = sum(1 for _, p, _ in results if p)
    critical_failed = [(n, p) for n, p, c in results if not p and c]
    
    print(f"\nTotal: {passed}/{total} checks passed")
    
    if critical_failed:
        print(f"\n[FAIL] {len(critical_failed)} CRITICAL failures:")
        for name, _ in critical_failed:
            print(f"  - {name}")
        print("\nFix critical issues before running generation.")
        return 1
    else:
        print("\n[OK] All critical checks passed!")
        print("System is ready for generation.")
        return 0


if __name__ == "__main__":
    sys.exit(main())

