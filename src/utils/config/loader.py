"""
Configuration loader with validation and initialization.
Handles loading environment variables and creating configuration objects.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

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
    GenerationParams,
    DomainType
)


def load_path_config(project_root: Optional[Path] = None) -> PathConfig:
    """
    Load and create path configuration.
    
    Args:
        project_root: Root directory of the project. If None, inferred from file location.
    
    Returns:
        PathConfig instance
    """
    if project_root is None:
        # Infer from this file's location: config/loader.py -> utils/config/loader.py
        # So we go up 3 levels to get to project root
        project_root = Path(__file__).parent.parent.parent.parent
    
    return PathConfig(
        project_root=project_root,
        data_dir=project_root / "data",
        raw_data_dir=project_root / "data" / "raw",
        synthetic_data_dir=project_root / "data" / "synthetic",
        reference_data_dir=project_root / "data" / "reference",
        models_dir=project_root / "models",
        logs_dir=project_root / "logs"
    )


def load_aws_config() -> AWSConfig:
    """
    Load AWS configuration from environment variables.
    
    Returns:
        AWSConfig instance
    """
    # Allow model IDs to be overridden via env vars (format: MODEL_NAME=model_id)
    default_model_ids = {
        "claude_35_sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        "claude_3_sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude_haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "llama_32_90b": "meta.llama3-2-90b-instruct-v1:0",
        "nova_pro": "us.amazon.nova-pro-v1:0"
    }
    
    # Override with env vars if present
    model_ids = {}
    for model_name in default_model_ids:
        env_key = f"BEDROCK_MODEL_ID_{model_name.upper()}"
        model_ids[model_name] = os.getenv(env_key, default_model_ids[model_name])
    
    default_model_env = os.getenv("DEFAULT_MODEL", "claude_35_sonnet")
    # Validate default model is in the model_ids dict
    if default_model_env not in model_ids:
        default_model_env = "claude_35_sonnet"  # Fallback to safe default
    
    return AWSConfig(
        region=os.getenv("AWS_REGION", "us-east-1"),
        access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        s3_bucket=os.getenv("S3_BUCKET"),
        bedrock_model_ids=model_ids,
        default_model=default_model_env  # type: ignore
    )


def load_domain_configs() -> dict[DomainType, DomainConfig]:
    """
    Load domain configurations.
    
    Returns:
        Dictionary mapping domain types to DomainConfig instances
    """
    return {
        "customer_service": DomainConfig(
            name="Customer Service Conversations",
            description="Multi-turn dialogues between customers and support agents",
            default_size=1000,
            reference_dataset="banking77",
            implemented=True,
            intents=[
                "account_inquiry",
                "billing_issue",
                "technical_support",
                "product_information",
                "complaint",
                "cancellation_request",
                "feature_request",
                "password_reset",
                "refund_request",
                "general_inquiry"
            ],
            generation_params=GenerationParams(
                temperature=0.7,
                max_tokens=1000,
                top_p=0.9
            )
        ),
        "time_series": DomainConfig(
            name="Time-Series Data",
            description="Temporal data with trends, seasonality, and anomalies",
            default_size=1000,
            reference_dataset="electricity",
            implemented=True,
            series_types=[
                "sensor_temperature",
                "sensor_pressure",
                "stock_price",
                "energy_consumption",
                "network_traffic",
                "manufacturing_output"
            ],
            frequencies=["1min", "5min", "1hour", "1day"],
            default_length=100,
            generation_params=GenerationParams(
                temperature=0.5,
                max_tokens=2000,
                top_p=0.85
            )
        ),
        "financial": DomainConfig(
            name="Financial Transactions",
            description="Synthetic financial transactions with fraud labels (ARCHITECTURE ONLY)",
            default_size=1000,
            reference_dataset="card_fraud",
            implemented=False,
            documentation_path="docs/DOMAIN3_FINANCIAL.md"
        )
    }


def load_generation_config() -> GenerationConfig:
    """
    Load generation configuration.
    
    Returns:
        GenerationConfig instance
    """
    return GenerationConfig(
        max_requests_per_minute=int(os.getenv("MAX_REQUESTS_PER_MINUTE", "10")),
        retry_attempts=int(os.getenv("RETRY_ATTEMPTS", "3")),
        retry_delay_seconds=float(os.getenv("RETRY_DELAY_SECONDS", "2.0")),
        batch_size=int(os.getenv("BATCH_SIZE", "10")),
        enable_prompt_caching=os.getenv("ENABLE_PROMPT_CACHING", "true").lower() == "true"
    )


def load_validation_config() -> ValidationConfig:
    """
    Load validation configuration.
    
    Returns:
        ValidationConfig instance
    """
    return ValidationConfig(
        completeness_min=float(os.getenv("COMPLETENESS_MIN", "0.95")),
        consistency_min=float(os.getenv("CONSISTENCY_MIN", "0.90")),
        realism_min=float(os.getenv("REALISM_MIN", "0.85")),
        diversity_min=float(os.getenv("DIVERSITY_MIN", "0.80")),
        overall_quality_min=float(os.getenv("OVERALL_QUALITY_MIN", "85.0")),
        sentiment_imbalance_max=float(os.getenv("SENTIMENT_IMBALANCE_MAX", "0.3")),
        topic_coverage_min=float(os.getenv("TOPIC_COVERAGE_MIN", "0.7"))
    )


def load_training_config() -> TrainingConfig:
    """
    Load training configuration.
    
    Returns:
        TrainingConfig instance
    """
    return TrainingConfig(
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        val_size=float(os.getenv("VAL_SIZE", "0.1")),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        n_jobs=int(os.getenv("N_JOBS", "-1")),
        models={
            "logistic_regression": {
                "enabled": os.getenv("LR_ENABLED", "true").lower() == "true",
                "hyperparams": {
                    "C": float(os.getenv("LR_C", "1.0")),
                    "max_iter": int(os.getenv("LR_MAX_ITER", "1000")),
                    "solver": os.getenv("LR_SOLVER", "lbfgs")
                }
            },
            "xgboost": {
                "enabled": os.getenv("XGB_ENABLED", "true").lower() == "true",
                "hyperparams": {
                    "n_estimators": int(os.getenv("XGB_N_ESTIMATORS", "100")),
                    "max_depth": int(os.getenv("XGB_MAX_DEPTH", "5")),
                    "learning_rate": float(os.getenv("XGB_LEARNING_RATE", "0.1"))
                }
            },
            "random_forest": {
                "enabled": os.getenv("RF_ENABLED", "false").lower() == "true",
                "hyperparams": {
                    "n_estimators": int(os.getenv("RF_N_ESTIMATORS", "100")),
                    "max_depth": int(os.getenv("RF_MAX_DEPTH", "10"))
                }
            }
        }
    )


def load_registry_config(data_dir: Path) -> RegistryConfig:
    """
    Load registry configuration.
    
    Args:
        data_dir: Data directory path
    
    Returns:
        RegistryConfig instance
    """
    return RegistryConfig(
        db_path=data_dir / "registry.db",
        table_name=os.getenv("REGISTRY_TABLE_NAME", "datasets")
    )


def load_ui_config() -> UIConfig:
    """
    Load UI configuration.
    
    Returns:
        UIConfig instance
    """
    return UIConfig(
        page_title=os.getenv("UI_PAGE_TITLE", "GENESIS-LAB"),
        page_icon=os.getenv("UI_PAGE_ICON", "🧬"),
        layout=os.getenv("UI_LAYOUT", "wide"),  # type: ignore
        initial_sidebar_state=os.getenv("UI_SIDEBAR_STATE", "expanded")  # type: ignore
    )


def load_logging_config() -> LoggingConfig:
    """
    Load logging configuration.
    
    Returns:
        LoggingConfig instance
    """
    return LoggingConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),  # type: ignore
        format=os.getenv(
            "LOG_FORMAT",
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )


def load_config(
    project_root: Optional[Path] = None,
    initialize_dirs: bool = True
) -> AppConfig:
    """
    Load complete application configuration.
    
    Args:
        project_root: Root directory of the project. If None, inferred.
        initialize_dirs: Whether to create directories if they don't exist.
    
    Returns:
        AppConfig instance with all configurations loaded and validated
    
    Raises:
        RuntimeError: If directory initialization fails
        ValidationError: If configuration validation fails
    """
    # Load environment variables
    load_dotenv()
    
    # Load path configuration
    paths = load_path_config(project_root)
    
    # Initialize directories if requested
    if initialize_dirs:
        paths.initialize_directories()
    
    # Load all other configurations
    aws = load_aws_config()
    domains = load_domain_configs()
    generation = load_generation_config()
    validation = load_validation_config()
    training = load_training_config()
    registry = load_registry_config(paths.data_dir)
    ui = load_ui_config()
    logging = load_logging_config()
    
    # Create and return main config
    return AppConfig(
        paths=paths,
        aws=aws,
        domains=domains,
        generation=generation,
        validation=validation,
        training=training,
        registry=registry,
        ui=ui,
        logging=logging
    )

