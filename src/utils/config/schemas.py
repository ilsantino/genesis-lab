"""
Pydantic schemas for configuration management.
Provides type-safe, validated configuration models.
"""

from pathlib import Path
from typing import Literal, Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator


# Type aliases
DomainType = Literal["customer_service", "time_series", "financial"]
ModelName = Literal[
    "claude_35_sonnet",
    "claude_3_sonnet",
    "claude_haiku",
    "llama_32_90b",
    "nova_pro"
]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class GenerationParams(BaseModel):
    """Parameters for LLM generation."""
    
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0-2.0)"
    )
    max_tokens: int = Field(
        default=1000,
        gt=0,
        description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling parameter (0.0-1.0)"
    )


class DomainConfig(BaseModel):
    """Configuration for a specific data domain."""
    
    name: str = Field(description="Human-readable domain name")
    description: str = Field(description="Domain description")
    default_size: int = Field(gt=0, description="Default dataset size")
    reference_dataset: str = Field(description="Reference dataset identifier")
    implemented: bool = Field(default=True, description="Whether domain is implemented")
    generation_params: GenerationParams = Field(default_factory=GenerationParams)
    
    # Optional domain-specific fields
    intents: Optional[List[str]] = Field(default=None, description="Available intents (customer_service)")
    series_types: Optional[List[str]] = Field(default=None, description="Series types (time_series)")
    frequencies: Optional[List[str]] = Field(default=None, description="Time frequencies (time_series)")
    default_length: Optional[int] = Field(default=None, gt=0, description="Default series length (time_series)")
    documentation_path: Optional[str] = Field(default=None, description="Path to domain documentation")


class PathConfig(BaseModel):
    """Project path configuration."""
    
    project_root: Path = Field(description="Root directory of the project")
    data_dir: Path = Field(description="Data directory")
    raw_data_dir: Path = Field(description="Raw data directory")
    synthetic_data_dir: Path = Field(description="Synthetic data directory")
    reference_data_dir: Path = Field(description="Reference data directory")
    models_dir: Path = Field(description="Models directory")
    logs_dir: Path = Field(description="Logs directory")
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v
    
    def initialize_directories(self) -> None:
        """Create all directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.raw_data_dir,
            self.synthetic_data_dir,
            self.reference_data_dir,
            self.models_dir,
            self.logs_dir
        ]
        for dir_path in dirs:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise RuntimeError(
                    f"Failed to create directory {dir_path}: {e}"
                ) from e


class AWSConfig(BaseModel):
    """AWS configuration with validation."""
    
    region: str = Field(
        default="us-east-1",
        description="AWS region"
    )
    access_key_id: Optional[str] = Field(
        default=None,
        description="AWS access key ID"
    )
    secret_access_key: Optional[str] = Field(
        default=None,
        description="AWS secret access key"
    )
    bedrock_model_ids: Dict[ModelName, str] = Field(
        default_factory=lambda: {
            "claude_35_sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
            "claude_3_sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
            "claude_haiku": "anthropic.claude-3-haiku-20240307-v1:0",
            "llama_32_90b": "meta.llama3-2-90b-instruct-v1:0",
            "nova_pro": "us.amazon.nova-pro-v1:0"
        },
        description="Bedrock model IDs"
    )
    default_model: ModelName = Field(
        default="claude_35_sonnet",
        description="Default model for generation"
    )
    
    @field_validator('access_key_id', 'secret_access_key')
    @classmethod
    def validate_credential_format(cls, v: Optional[str]) -> Optional[str]:
        """Validate AWS credential format (basic checks)."""
        if v is not None:
            v = v.strip()
            if not v:
                return None
            # Basic format validation
            if len(v) < 16:
                raise ValueError("AWS credentials appear to be too short")
        return v
    
    def validate_credentials_present(self) -> bool:
        """Check if AWS credentials are present."""
        return bool(self.access_key_id and self.secret_access_key)
    
    def get_model_id(self, model_name: Optional[str] = None) -> str:
        """
        Get Bedrock model ID for a given model name.
        
        Args:
            model_name: Model name identifier. If None, uses default model.
        
        Returns:
            Bedrock model ID string
        
        Raises:
            ValueError: If model name is unknown
        """
        if model_name is None:
            model_name = self.default_model
        # Type narrowing: model_name is now str, but we need to check it's a valid ModelName
        if model_name not in self.bedrock_model_ids:
            raise ValueError(f"Unknown model: {model_name}")
        return self.bedrock_model_ids[model_name]  # type: ignore


class GenerationConfig(BaseModel):
    """Configuration for data generation."""
    
    max_requests_per_minute: int = Field(
        default=10,
        gt=0,
        description="Maximum API requests per minute"
    )
    retry_attempts: int = Field(
        default=3,
        ge=0,
        description="Number of retry attempts"
    )
    retry_delay_seconds: float = Field(
        default=2.0,
        ge=0.0,
        description="Base delay for exponential backoff (seconds)"
    )
    batch_size: int = Field(
        default=10,
        gt=0,
        description="Batch size for API calls"
    )
    enable_prompt_caching: bool = Field(
        default=True,
        description="Enable prompt caching"
    )


class ValidationConfig(BaseModel):
    """Configuration for data validation."""
    
    completeness_min: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Minimum completeness threshold"
    )
    consistency_min: float = Field(
        default=0.90,
        ge=0.0,
        le=1.0,
        description="Minimum consistency threshold"
    )
    realism_min: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Minimum realism threshold"
    )
    diversity_min: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Minimum diversity threshold"
    )
    overall_quality_min: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Minimum overall quality score"
    )
    sentiment_imbalance_max: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum sentiment imbalance"
    )
    topic_coverage_min: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum topic coverage"
    )


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    test_size: float = Field(
        default=0.2,
        gt=0.0,
        lt=1.0,
        description="Test set size ratio"
    )
    val_size: float = Field(
        default=0.1,
        gt=0.0,
        lt=1.0,
        description="Validation set size ratio"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )
    n_jobs: int = Field(
        default=-1,
        description="Number of parallel jobs (-1 for all cores)"
    )
    models: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Model configurations"
    )
    
    @model_validator(mode='after')
    def validate_split_sizes(self) -> 'TrainingConfig':
        """Validate that test_size + val_size < 1.0."""
        if self.test_size + self.val_size >= 1.0:
            raise ValueError(
                f"test_size ({self.test_size}) + val_size ({self.val_size}) "
                f"must be less than 1.0"
            )
        return self


class RegistryConfig(BaseModel):
    """Configuration for dataset registry."""
    
    db_path: Path = Field(description="Path to registry database")
    table_name: str = Field(
        default="datasets",
        description="Registry table name"
    )
    
    @field_validator('db_path', mode='before')
    @classmethod
    def convert_to_path(cls, v: Any) -> Path:
        """Convert string paths to Path objects."""
        if isinstance(v, str):
            return Path(v)
        return v


class UIConfig(BaseModel):
    """Configuration for Streamlit UI."""
    
    page_title: str = Field(
        default="GENESIS-LAB",
        description="Page title"
    )
    page_icon: str = Field(
        default="🧬",
        description="Page icon"
    )
    layout: Literal["centered", "wide"] = Field(
        default="wide",
        description="Page layout"
    )
    initial_sidebar_state: Literal["auto", "expanded", "collapsed"] = Field(
        default="expanded",
        description="Initial sidebar state"
    )


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    
    level: LogLevel = Field(
        default="INFO",
        description="Logging level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )


class AppConfig(BaseModel):
    """Main application configuration."""
    
    paths: PathConfig
    aws: AWSConfig
    domains: Dict[DomainType, DomainConfig]
    generation: GenerationConfig
    validation: ValidationConfig
    training: TrainingConfig
    registry: RegistryConfig
    ui: UIConfig
    logging: LoggingConfig
    config_version: str = Field(
        default="1.0.0",
        description="Configuration schema version"
    )
    
    def get_domain_config(self, domain: DomainType) -> DomainConfig:
        """Get configuration for a specific domain."""
        if domain not in self.domains:
            raise ValueError(f"Unknown domain: {domain}")
        return self.domains[domain]

