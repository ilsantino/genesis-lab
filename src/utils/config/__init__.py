"""
Configuration module for GENESIS-LAB.
Provides backward-compatible API while using new Pydantic-based structure.
"""

from pathlib import Path
from typing import Optional, Literal, Dict, Any

from .schemas import (
    AppConfig,
    PathConfig,
    AWSConfig,
    DomainConfig,
    GenerationConfig,
    ValidationConfig,
    TrainingConfig,
    RegistryConfig,
    UIConfig,
    LoggingConfig,
    DomainType,
    ModelName
)
from .loader import load_config

# Global config instance (lazy-loaded)
_config: Optional[AppConfig] = None


def get_config(
    project_root: Optional[Path] = None,
    initialize_dirs: bool = True,
    reload: bool = False
) -> AppConfig:
    """
    Get or load application configuration (singleton pattern).
    
    Args:
        project_root: Root directory of the project. If None, inferred.
        initialize_dirs: Whether to create directories if they don't exist.
        reload: Force reload of configuration even if already loaded.
    
    Returns:
        AppConfig instance
    """
    global _config
    if _config is None or reload:
        _config = load_config(project_root=project_root, initialize_dirs=initialize_dirs)
    return _config


# Initialize config on first import (with directory creation for backward compatibility)
try:
    _config = load_config(initialize_dirs=True)
except Exception:
    # If loading fails, config will be None and will be loaded on first use
    _config = None


# ============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# ============================================================================
# Export all constants and functions that the old config.py provided
# This ensures backward compatibility when importing from utils.config

if _config is not None:
    # Project paths
    PROJECT_ROOT = _config.paths.project_root
    DATA_DIR = _config.paths.data_dir
    RAW_DATA_DIR = _config.paths.raw_data_dir
    SYNTHETIC_DATA_DIR = _config.paths.synthetic_data_dir
    REFERENCE_DATA_DIR = _config.paths.reference_data_dir
    MODELS_DIR = _config.paths.models_dir
    LOGS_DIR = _config.paths.logs_dir
    
    # AWS configuration
    AWS_REGION = _config.aws.region
    AWS_ACCESS_KEY_ID = _config.aws.access_key_id
    AWS_SECRET_ACCESS_KEY = _config.aws.secret_access_key
    BEDROCK_MODEL_IDS: Dict[str, str] = dict(_config.aws.bedrock_model_ids)
    DEFAULT_MODEL: str = _config.aws.default_model
    
    # Domain configurations (convert to dicts for backward compatibility)
    DOMAIN_CONFIGS: Dict[str, Dict[str, Any]] = {}
    for domain_type, domain_config in _config.domains.items():
        domain_dict = domain_config.model_dump()
        if hasattr(domain_config.generation_params, 'model_dump'):
            domain_dict['generation_params'] = domain_config.generation_params.model_dump()
        DOMAIN_CONFIGS[domain_type] = domain_dict
    
    # Generation configuration
    MAX_REQUESTS_PER_MINUTE = _config.generation.max_requests_per_minute
    RETRY_ATTEMPTS = _config.generation.retry_attempts
    RETRY_DELAY_SECONDS = _config.generation.retry_delay_seconds
    BATCH_SIZE = _config.generation.batch_size
    ENABLE_PROMPT_CACHING = _config.generation.enable_prompt_caching
    
    # Validation configuration
    VALIDATION_THRESHOLDS = {
        "completeness_min": _config.validation.completeness_min,
        "consistency_min": _config.validation.consistency_min,
        "realism_min": _config.validation.realism_min,
        "diversity_min": _config.validation.diversity_min,
        "overall_quality_min": _config.validation.overall_quality_min
    }
    
    BIAS_DETECTION_THRESHOLDS = {
        "sentiment_imbalance_max": _config.validation.sentiment_imbalance_max,
        "topic_coverage_min": _config.validation.topic_coverage_min
    }
    
    # Training configuration
    TRAINING_CONFIG = {
        "test_size": _config.training.test_size,
        "val_size": _config.training.val_size,
        "random_state": _config.training.random_state,
        "n_jobs": _config.training.n_jobs,
        "models": _config.training.models
    }
    
    # Registry configuration
    REGISTRY_DB_PATH = _config.registry.db_path
    REGISTRY_TABLE_NAME = _config.registry.table_name
    
    # UI configuration
    UI_CONFIG = {
        "page_title": _config.ui.page_title,
        "page_icon": _config.ui.page_icon,
        "layout": _config.ui.layout,
        "initial_sidebar_state": _config.ui.initial_sidebar_state
    }
    
    # Logging configuration
    LOG_LEVEL = _config.logging.level
    LOG_FORMAT = _config.logging.format
else:
    # Fallback values if config failed to load
    PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
    REFERENCE_DATA_DIR = DATA_DIR / "reference"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    AWS_REGION = "us-east-1"
    AWS_ACCESS_KEY_ID = None
    AWS_SECRET_ACCESS_KEY = None
    BEDROCK_MODEL_IDS = {}
    DEFAULT_MODEL = "claude_35_sonnet"
    DOMAIN_CONFIGS = {}
    MAX_REQUESTS_PER_MINUTE = 10
    RETRY_ATTEMPTS = 3
    RETRY_DELAY_SECONDS = 2
    BATCH_SIZE = 10
    ENABLE_PROMPT_CACHING = True
    VALIDATION_THRESHOLDS = {}
    BIAS_DETECTION_THRESHOLDS = {}
    TRAINING_CONFIG = {}
    REGISTRY_DB_PATH = DATA_DIR / "registry.db"
    REGISTRY_TABLE_NAME = "datasets"
    UI_CONFIG = {}
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# Helper functions for backward compatibility
def _ensure_config() -> AppConfig:
    """Ensure config is loaded."""
    global _config
    if _config is None:
        _config = get_config()
    return _config


def get_domain_config(domain: DomainType) -> Dict[str, Any]:
    """
    Get configuration for a specific domain.
    
    Args:
        domain: Domain type identifier
    
    Returns:
        Dictionary with domain configuration
    
    Raises:
        ValueError: If domain is unknown
    """
    return _ensure_config().get_domain_config(domain).model_dump()


def get_bedrock_model_id(model_name: Optional[str] = None) -> str:
    """
    Get Bedrock model ID for a given model name.
    
    Args:
        model_name: Model name identifier. If None, uses default model.
    
    Returns:
        Bedrock model ID string
    
    Raises:
        ValueError: If model name is unknown
    """
    return _ensure_config().aws.get_model_id(model_name)  # type: ignore


def validate_aws_credentials() -> bool:
    """
    Validate that AWS credentials are present.
    
    Note: This only checks if credentials are set, not if they are valid.
    For actual validation, use boto3 to test the credentials.
    
    Returns:
        True if both access key ID and secret access key are set
    """
    return _ensure_config().aws.validate_credentials_present()


def initialize_directories() -> None:
    """
    Initialize project directories explicitly.
    
    This function can be called to ensure all required directories exist.
    Useful when you want to control when directories are created.
    
    Raises:
        RuntimeError: If directory creation fails
    """
    _ensure_config().paths.initialize_directories()


# Export all schemas, types, constants, and functions
__all__ = [
    # New API
    "AppConfig",
    "PathConfig",
    "AWSConfig",
    "DomainConfig",
    "GenerationConfig",
    "ValidationConfig",
    "TrainingConfig",
    "RegistryConfig",
    "UIConfig",
    "LoggingConfig",
    "DomainType",
    "ModelName",
    "load_config",
    "get_config",
    # Backward compatibility
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "SYNTHETIC_DATA_DIR",
    "REFERENCE_DATA_DIR",
    "MODELS_DIR",
    "LOGS_DIR",
    "AWS_REGION",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "BEDROCK_MODEL_IDS",
    "DEFAULT_MODEL",
    "DOMAIN_CONFIGS",
    "MAX_REQUESTS_PER_MINUTE",
    "RETRY_ATTEMPTS",
    "RETRY_DELAY_SECONDS",
    "BATCH_SIZE",
    "ENABLE_PROMPT_CACHING",
    "VALIDATION_THRESHOLDS",
    "BIAS_DETECTION_THRESHOLDS",
    "TRAINING_CONFIG",
    "REGISTRY_DB_PATH",
    "REGISTRY_TABLE_NAME",
    "UI_CONFIG",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "get_domain_config",
    "get_bedrock_model_id",
    "validate_aws_credentials",
    "initialize_directories",
]

