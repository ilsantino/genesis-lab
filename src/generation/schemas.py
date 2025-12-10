"""
Data schemas for synthetic data generation.
Defines the structure and validation rules for each domain.

Compatibility: Pydantic v1.10+ and v2.x
"""

from __future__ import annotations

import sys
from typing import List, Optional, Dict, Union, Any
from datetime import datetime, timezone

# ============================================================================
# PYDANTIC VERSION COMPATIBILITY LAYER
# ============================================================================

try:
    from pydantic import __version__ as PYDANTIC_VERSION
    PYDANTIC_V2 = PYDANTIC_VERSION.startswith("2.")
except ImportError:
    PYDANTIC_V2 = False

if PYDANTIC_V2:
    from pydantic import BaseModel, Field, field_validator, ConfigDict
    from typing import Literal
else:
    from pydantic import BaseModel, Field, validator
    from typing_extensions import Literal

# ============================================================================
# CONSTANTS (Centralized, maintainable)
# ============================================================================

class SchemaConstants:
    """Centralized constants for schema validation."""
    
    # Conversation limits
    MIN_TURN_LENGTH: int = 10
    MAX_TURN_LENGTH: int = 500
    MIN_TURNS: int = 2
    MAX_TURNS: int = 20
    
    # Time series limits
    MIN_SERIES_POINTS: int = 10
    VALUE_LOWER_BOUND: float = -1e10
    VALUE_UPPER_BOUND: float = 1e10
    
    # Supported languages (ISO 639-1)
    SUPPORTED_LANGUAGES: frozenset = frozenset({
        "en", "es", "fr", "de", "pt", "it", "nl", "pl", "ru", "zh", 
        "ja", "ko", "ar", "hi", "tr"
    })

# Type aliases for better type safety
Metadata = Dict[str, Union[str, int, float, bool, None]]
IntentType = Literal[
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
]
SentimentType = Literal["positive", "neutral", "negative"]
ResolutionType = Literal["resolved", "escalated", "unresolved"]
SpeakerType = Literal["customer", "agent"]
FrequencyType = Literal["1min", "5min", "1hour", "1day"]
SeriesType = Literal[
    "sensor_temperature",
    "sensor_pressure",
    "stock_price",
    "energy_consumption",
    "network_traffic",
    "manufacturing_output"
]
DomainType = Literal["customer_service", "time_series", "financial"]
FileFormatType = Literal["json", "jsonl", "parquet", "csv"]
BiasSeverityType = Literal["none", "low", "medium", "high"]

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def ensure_timezone_aware(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware (UTC if naive)."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt

# ============================================================================
# BASE MODEL WITH CONFIG
# ============================================================================

if PYDANTIC_V2:
    class BaseSchema(BaseModel):
        """Base schema with common configuration."""
        model_config = ConfigDict(
            str_strip_whitespace=True,
            validate_assignment=True,
            extra="forbid",
            frozen=False,
        )
else:
    class BaseSchema(BaseModel):
        """Base schema with common configuration."""
        class Config:
            anystr_strip_whitespace = True
            validate_assignment = True
            extra = "forbid"

# ============================================================================
# DOMAIN 1: CUSTOMER SERVICE CONVERSATIONS
# ============================================================================

class ConversationTurn(BaseSchema):
    """Single turn in a customer service conversation."""
    
    speaker: SpeakerType
    text: str = Field(
        ..., 
        min_length=SchemaConstants.MIN_TURN_LENGTH, 
        max_length=SchemaConstants.MAX_TURN_LENGTH
    )
    timestamp: Optional[datetime] = None
    
    if PYDANTIC_V2:
        @field_validator('text')
        @classmethod
        def text_not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("Text cannot be empty or whitespace only")
            return v.strip()
        
        @field_validator('timestamp')
        @classmethod
        def ensure_tz_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
            return ensure_timezone_aware(v) if v else None
    else:
        @validator('text')
        def text_not_empty(cls, v: str) -> str:
            if not v.strip():
                raise ValueError("Text cannot be empty or whitespace only")
            return v.strip()
        
        @validator('timestamp')
        def ensure_tz_aware(cls, v: Optional[datetime]) -> Optional[datetime]:
            return ensure_timezone_aware(v) if v else None


class CustomerServiceConversation(BaseSchema):
    """Complete customer service conversation."""
    
    conversation_id: str
    intent: IntentType
    sentiment: SentimentType
    turns: List[ConversationTurn] = Field(
        ..., 
        min_length=SchemaConstants.MIN_TURNS,  # Single source of truth
        max_length=SchemaConstants.MAX_TURNS
    )
    resolution_status: ResolutionType
    language: str = Field(default="en")
    metadata: Metadata = Field(default_factory=dict)
    
    if PYDANTIC_V2:
        @field_validator('turns')
        @classmethod
        def validate_first_turn(cls, v: List[ConversationTurn]) -> List[ConversationTurn]:
            if v and v[0].speaker != "customer":
                raise ValueError("First turn must be from customer")
            return v
        
        @field_validator('language')
        @classmethod
        def validate_language(cls, v: str) -> str:
            if v not in SchemaConstants.SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Language '{v}' not supported. "
                    f"Valid: {sorted(SchemaConstants.SUPPORTED_LANGUAGES)}"
                )
            return v
    else:
        @validator('turns')
        def validate_first_turn(cls, v: List[ConversationTurn]) -> List[ConversationTurn]:
            if v and v[0].speaker != "customer":
                raise ValueError("First turn must be from customer")
            return v
        
        @validator('language')
        def validate_language(cls, v: str) -> str:
            if v not in SchemaConstants.SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Language '{v}' not supported. "
                    f"Valid: {sorted(SchemaConstants.SUPPORTED_LANGUAGES)}"
                )
            return v

# ============================================================================
# DOMAIN 2: TIME-SERIES DATA
# ============================================================================

class TimeSeriesPoint(BaseSchema):
    """Single point in a time series."""
    
    timestamp: datetime
    value: float
    
    if PYDANTIC_V2:
        @field_validator('timestamp')
        @classmethod
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)
        
        @field_validator('value')
        @classmethod
        def value_is_finite(cls, v: float) -> float:
            if not (SchemaConstants.VALUE_LOWER_BOUND < v < SchemaConstants.VALUE_UPPER_BOUND):
                raise ValueError(
                    f"Value must be between {SchemaConstants.VALUE_LOWER_BOUND} "
                    f"and {SchemaConstants.VALUE_UPPER_BOUND}"
                )
            return v
    else:
        @validator('timestamp')
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)
        
        @validator('value')
        def value_is_finite(cls, v: float) -> float:
            if not (SchemaConstants.VALUE_LOWER_BOUND < v < SchemaConstants.VALUE_UPPER_BOUND):
                raise ValueError(
                    f"Value must be between {SchemaConstants.VALUE_LOWER_BOUND} "
                    f"and {SchemaConstants.VALUE_UPPER_BOUND}"
                )
            return v


class TimeSeries(BaseSchema):
    """Complete time series dataset."""
    
    series_id: str
    series_type: SeriesType
    frequency: FrequencyType
    points: List[TimeSeriesPoint] = Field(
        ..., 
        min_length=SchemaConstants.MIN_SERIES_POINTS
    )
    
    # Statistical properties (computed during generation)
    has_trend: bool = False
    has_seasonality: bool = False
    has_noise: bool = True
    anomaly_indices: List[int] = Field(default_factory=list)
    
    metadata: Metadata = Field(default_factory=dict)
    
    if PYDANTIC_V2:
        @field_validator('points')
        @classmethod
        def validate_timestamps_sorted(cls, v: List[TimeSeriesPoint]) -> List[TimeSeriesPoint]:
            timestamps = [p.timestamp for p in v]
            if timestamps != sorted(timestamps):
                raise ValueError("Timestamps must be in ascending order")
            return v
    else:
        @validator('points')
        def validate_timestamps_sorted(cls, v: List[TimeSeriesPoint]) -> List[TimeSeriesPoint]:
            timestamps = [p.timestamp for p in v]
            if timestamps != sorted(timestamps):
                raise ValueError("Timestamps must be in ascending order")
            return v

# ============================================================================
# VALIDATION RESULT SCHEMAS
# ============================================================================

class QualityMetrics(BaseSchema):
    """Quality validation metrics for any dataset."""
    
    completeness_score: float = Field(..., ge=0, le=1)
    consistency_score: float = Field(..., ge=0, le=1)
    realism_score: float = Field(..., ge=0, le=1)
    diversity_score: float = Field(..., ge=0, le=1)
    overall_quality_score: float = Field(..., ge=0, le=100)
    
    issues_found: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    
    metadata: Metadata = Field(default_factory=dict)


class BiasMetrics(BaseSchema):
    """Bias detection metrics."""
    
    demographic_balance: Dict[str, Any] = Field(default_factory=dict)
    sentiment_distribution: Dict[str, float] = Field(default_factory=dict)
    topic_coverage: Dict[str, float] = Field(default_factory=dict)
    
    bias_detected: bool = False
    bias_severity: BiasSeverityType = "none"
    recommendations: List[str] = Field(default_factory=list)
    
    metadata: Metadata = Field(default_factory=dict)

# ============================================================================
# DATASET REGISTRY SCHEMA
# ============================================================================

class DatasetMetadata(BaseSchema):
    """Metadata for registered datasets."""
    
    dataset_id: str
    domain: DomainType
    size: int = Field(..., gt=0)
    generation_date: datetime
    
    # Generation parameters
    model_used: str
    prompt_template_version: str
    generation_config: Dict[str, Any] = Field(default_factory=dict)
    
    # Quality metrics
    quality_metrics: Optional[QualityMetrics] = None
    bias_metrics: Optional[BiasMetrics] = None
    
    # Storage
    file_path: str
    file_format: FileFormatType
    file_size_mb: float = Field(..., ge=0)
    
    # Training results (if applicable)
    model_trained: Optional[str] = None
    model_metrics: Optional[Dict[str, float]] = None
    
    tags: List[str] = Field(default_factory=list)
    notes: Optional[str] = None
    
    if PYDANTIC_V2:
        @field_validator('generation_date')
        @classmethod
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)
    else:
        @validator('generation_date')
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)

# ============================================================================
# SCHEMA VERSION (for tracking prompt/schema compatibility)
# ============================================================================

SCHEMA_VERSION = "1.1.0"

def get_schema_info() -> Dict[str, Any]:
    """Return schema version and compatibility info."""
    return {
        "version": SCHEMA_VERSION,
        "pydantic_version": PYDANTIC_VERSION if 'PYDANTIC_VERSION' in dir() else "unknown",
        "pydantic_v2": PYDANTIC_V2,
        "domains": ["customer_service", "time_series"],
        "domains_planned": ["financial"],
    }
