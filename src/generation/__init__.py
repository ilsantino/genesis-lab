"""
Synthetic data generation module for GENESIS-LAB.

This module provides generators for creating high-quality synthetic datasets
using AWS Bedrock foundation models.

Available generators:
- BaseGenerator: Abstract base class for all generators
- CustomerServiceGenerator: Banking77-style customer service conversations
- TimeSeriesGenerator: Multi-domain time series data (electricity, sensors, etc.)
"""

from .generator import BaseGenerator, CustomerServiceGenerator
from .timeseries_generator import TimeSeriesGenerator
from .schemas import (
    CustomerServiceConversation,
    ConversationTurn,
    TimeSeries,
    TimeSeriesPoint,
    QualityMetrics,
    BiasMetrics,
    DatasetMetadata,
    IntentType,
    SentimentType,
    ResolutionType,
    SCHEMA_VERSION,
)

__all__ = [
    # Generators
    "BaseGenerator",
    "CustomerServiceGenerator",
    "TimeSeriesGenerator",
    # Schemas
    "CustomerServiceConversation",
    "ConversationTurn",
    "TimeSeries",
    "TimeSeriesPoint",
    "QualityMetrics",
    "BiasMetrics",
    "DatasetMetadata",
    # Types
    "IntentType",
    "SentimentType",
    "ResolutionType",
    # Version
    "SCHEMA_VERSION",
]

