"""
Time series data generator using AWS Bedrock.

This module provides:
- TimeSeriesGenerator: Generates realistic time series data for multiple domains

Features:
- Multiple domains: electricity, energy, sensors, financial
- Bilingual support (English/Spanish)
- Configurable patterns: seasonality, trends, anomalies
- Compatible with HuggingFace electricity_hourly format
"""

import json
import logging
import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.aws_client import BedrockClient
from src.utils.config import get_config, DomainType
from src.utils.config.schemas import GenerationParams

from .generator import BaseGenerator
from .templates.timeseries_prompts import (
    SYSTEM_PROMPTS,
    ALL_SERIES_TYPES,
    DOMAIN_SERIES_TYPES,
    SERIES_TYPE_SPECS,
    SERIES_TYPE_TO_DOMAIN,
    DEFAULT_DISTRIBUTION,
    Complexity,
    DataQuality,
    TrendType,
    Language,
    DomainContext,
    build_full_prompt_with_examples,
    validate_timeseries_schema,
    validate_temporal_consistency,
)


__all__ = ["TimeSeriesGenerator"]

logger = logging.getLogger(__name__)


class TimeSeriesGenerator(BaseGenerator):
    """
    Generator for realistic time series data across multiple domains.
    
    Generates time series data compatible with the electricity_hourly dataset
    format and extended to support multiple domains including electricity,
    energy, sensors, and financial data.
    
    Features:
    - 16 series types across 4 domains
    - Configurable seasonality (daily, weekly, monthly, annual)
    - Trend patterns (none, upward, downward, cyclic)
    - Anomaly injection (spike, drop, plateau, drift, outage)
    - Bilingual metadata (English/Spanish)
    - Standardized values (mean~0, std~1) for ML compatibility
    
    Example:
        >>> generator = TimeSeriesGenerator.from_config()
        >>> series = generator.generate_single(
        ...     series_type="residential_consumption",
        ...     length=168,  # 1 week hourly
        ...     complexity="medium"
        ... )
        >>> print(len(series["target"]))  # 168 values
    """
    
    def __init__(
        self,
        client: BedrockClient,
        domain: DomainType = "time_series",
        generation_params: Optional[GenerationParams] = None
    ):
        """
        Initialize TimeSeriesGenerator.
        
        Args:
            client: BedrockClient for LLM calls
            domain: Domain type (should be "time_series")
            generation_params: Optional generation parameter overrides
        """
        super().__init__(client, domain, generation_params)
        
        # Available series types
        self._series_types = ALL_SERIES_TYPES
        self._domain_series_types = DOMAIN_SERIES_TYPES
        
        logger.info(
            f"TimeSeriesGenerator ready with {len(self._series_types)} series types "
            f"across {len(self._domain_series_types)} domains"
        )
    
    @classmethod
    def from_config(cls) -> "TimeSeriesGenerator":
        """Create TimeSeriesGenerator from global configuration."""
        client = BedrockClient.from_config()
        return cls(client=client, domain="time_series")
    
    def _build_system_prompt(self, language: str = "en") -> str:
        """
        Build system prompt for time series generation.
        
        Args:
            language: Language code ("en" or "es")
        
        Returns:
            System prompt string
        """
        return SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["en"])
    
    def _select_random_parameters(
        self,
        series_type: Optional[str] = None,
        domain: Optional[str] = None,
        complexity: Optional[str] = None,
        language: Optional[str] = None,
        data_quality: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Select random parameters based on distribution config.
        
        Args:
            series_type: Optional specific series type (random if None)
            domain: Optional domain filter (random if None)
            complexity: Optional complexity level (random if None)
            language: Optional language (random if None)
            data_quality: Optional data quality (random if None)
        
        Returns:
            Dictionary with selected parameters
        """
        dist = DEFAULT_DISTRIBUTION
        
        # Select domain first if not specified
        if domain:
            selected_domain = domain
        else:
            domains = list(dist["domain"].keys())
            weights = list(dist["domain"].values())
            selected_domain = random.choices(domains, weights=weights)[0]
        
        # Select series type from domain
        if series_type:
            selected_series_type = series_type
        else:
            domain_types = self._domain_series_types.get(selected_domain, ALL_SERIES_TYPES)
            selected_series_type = random.choice(domain_types)
        
        # Select complexity
        if complexity:
            selected_complexity = complexity
        else:
            complexities = list(dist["complexity"].keys())
            weights = list(dist["complexity"].values())
            selected_complexity = random.choices(complexities, weights=weights)[0]
        
        # Select data quality
        if data_quality:
            selected_data_quality = data_quality
        else:
            qualities = list(dist["data_quality"].keys())
            weights = list(dist["data_quality"].values())
            selected_data_quality = random.choices(qualities, weights=weights)[0]
        
        # Select language
        if language:
            selected_language = language
        else:
            languages = list(dist["language"].keys())
            weights = list(dist["language"].values())
            selected_language = random.choices(languages, weights=weights)[0]
        
        # Select seasonality based on complexity
        if selected_complexity == "simple":
            seasonality_types = ["daily"]
        elif selected_complexity == "medium":
            seasonality_types = random.choice([["daily"], ["daily", "weekly"]])
        else:  # complex
            seasonality_types = random.choice([
                ["daily", "weekly"],
                ["daily", "weekly", "annual"],
                ["daily", "annual"]
            ])
        
        # Select trend based on complexity
        if selected_complexity == "simple":
            trend_type = "none"
        else:
            trend_options = ["none", "upward", "downward"]
            if selected_complexity == "complex":
                trend_options.append("cyclic")
            trend_type = random.choice(trend_options)
        
        # Select anomalies based on complexity and data quality
        anomaly_types: List[str] = []
        anomaly_count = 0
        if selected_complexity == "complex" and selected_data_quality != "clean":
            specs = SERIES_TYPE_SPECS.get(selected_series_type, {})
            typical_anomalies = specs.get("typical_anomalies", ["spike", "drop"])
            anomaly_types = random.sample(typical_anomalies, min(2, len(typical_anomalies)))
            anomaly_count = random.randint(1, 3)
        elif selected_complexity == "complex":
            specs = SERIES_TYPE_SPECS.get(selected_series_type, {})
            typical_anomalies = specs.get("typical_anomalies", ["spike"])
            anomaly_types = [random.choice(typical_anomalies)]
            anomaly_count = 1
        
        return {
            "domain": selected_domain,
            "series_type": selected_series_type,
            "complexity": selected_complexity,
            "data_quality": selected_data_quality,
            "language": selected_language,
            "seasonality_types": seasonality_types,
            "trend_type": trend_type,
            "anomaly_types": anomaly_types,
            "anomaly_count": anomaly_count
        }
    
    def generate_single(
        self,
        series_type: Optional[str] = None,
        length: int = 24,
        frequency: str = "1H",
        complexity: Optional[str] = None,
        language: str = "en",
        seasonality_types: Optional[List[str]] = None,
        trend_type: Optional[str] = None,
        anomaly_types: Optional[List[str]] = None,
        anomaly_count: int = 0,
        data_quality: Optional[str] = None,
        domain_context: Optional[str] = None,
        start_date: str = "2024-01-01T00:00:00Z",
        use_standardized: bool = True,
        num_examples: int = 2
    ) -> Dict[str, Any]:
        """
        Generate a single time series.
        
        Args:
            series_type: Type of series (e.g., "residential_consumption")
            length: Number of data points to generate
            frequency: Sampling frequency ("1H", "15min", etc.)
            complexity: "simple", "medium", or "complex"
            language: "en" or "es"
            seasonality_types: List of seasonality patterns to include
            trend_type: Type of trend ("none", "upward", "downward", "cyclic")
            anomaly_types: Types of anomalies to inject
            anomaly_count: Number of anomalies to generate
            data_quality: "clean", "noisy", or "missing_values"
            domain_context: Context type ("residential", "commercial", etc.)
            start_date: Starting timestamp
            use_standardized: Whether to use standardized values
            num_examples: Number of few-shot examples to include
        
        Returns:
            Generated time series dictionary with structure:
            {
                "series_id": str,
                "domain": str,
                "series_type": str,
                "frequency": str,
                "complexity": str,
                "data_quality": str,
                "language": str,
                "length": int,
                "seasonality_types": List[str],
                "trend_type": str,
                "anomaly_types": List[str],
                "anomaly_indices": List[int],
                "domain_context": str,
                "start": str,
                "target": List[float],
                "metadata": {...}
            }
        
        Raises:
            RuntimeError: If generation fails after retries
            ValueError: If invalid parameters provided
        """
        # Select random parameters for unspecified values
        params = self._select_random_parameters(
            series_type=series_type,
            complexity=complexity,
            language=language,
            data_quality=data_quality
        )
        
        selected_series_type = series_type or params["series_type"]
        selected_complexity = complexity or params["complexity"]
        selected_data_quality = data_quality or params["data_quality"]
        selected_seasonality = seasonality_types or params["seasonality_types"]
        selected_trend = trend_type or params["trend_type"]
        selected_anomaly_types = anomaly_types if anomaly_types is not None else params["anomaly_types"]
        selected_anomaly_count = anomaly_count if anomaly_count > 0 else params["anomaly_count"]
        
        # Get domain context from series specs if not provided
        if not domain_context:
            specs = SERIES_TYPE_SPECS.get(selected_series_type, {})
            domain_context = specs.get("context", "mixed")
        
        # Convert to enums for prompt builder
        complexity_enum = Complexity(selected_complexity)
        language_enum = Language(language)
        trend_enum = TrendType(selected_trend) if selected_trend else TrendType.NONE
        quality_enum = DataQuality(selected_data_quality) if selected_data_quality else DataQuality.CLEAN
        context_enum = DomainContext(domain_context) if domain_context else DomainContext.MIXED
        
        # Build prompt with few-shot examples
        prompts = build_full_prompt_with_examples(
            series_type=selected_series_type,
            frequency=frequency,
            length=length,
            complexity=complexity_enum,
            language=language_enum,
            num_examples=num_examples,
            seasonality_types=selected_seasonality,
            trend_type=trend_enum,
            anomaly_types=selected_anomaly_types,
            anomaly_count=selected_anomaly_count,
            data_quality=quality_enum,
            domain_context=context_enum,
            start_date=start_date,
            use_standardized=use_standardized
        )
        
        try:
            # Call LLM
            response = self.client.invoke_model(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=self._generation_params.temperature,
                max_tokens=self._generation_params.max_tokens,
                top_p=self._generation_params.top_p
            )
            
            # Parse response
            timeseries = self._parse_json_response(response)
            
            # Validate schema
            errors = validate_timeseries_schema(timeseries)
            if errors:
                logger.warning(f"Validation errors: {errors}")
                # Try to fix common issues
                timeseries = self._fix_timeseries_schema(
                    timeseries,
                    selected_series_type,
                    frequency,
                    length,
                    selected_complexity,
                    selected_data_quality,
                    language,
                    selected_seasonality,
                    selected_trend,
                    selected_anomaly_types,
                    domain_context,
                    start_date
                )
            
            # Check temporal consistency (warnings only)
            warnings = validate_temporal_consistency(timeseries)
            if warnings:
                logger.debug(f"Temporal consistency warnings: {warnings}")
            
            # Add metadata
            timeseries["metadata"] = timeseries.get("metadata", {})
            timeseries["metadata"].update({
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "model": self._config.aws.default_model,
                "generator_version": "1.0.0",
                "use_standardized": use_standardized
            })
            
            # Ensure series_id is unique
            if not timeseries.get("series_id") or timeseries["series_id"].startswith("ts_XXX"):
                timeseries["series_id"] = f"ts_{uuid.uuid4().hex[:12]}"
            
            self._total_generated += 1
            logger.debug(f"Generated time series: {timeseries['series_id']}")
            
            return timeseries
            
        except Exception as e:
            self._total_failed += 1
            logger.error(f"Generation failed: {e}")
            raise RuntimeError(f"Failed to generate time series: {e}") from e
    
    def _fix_timeseries_schema(
        self,
        timeseries: Dict[str, Any],
        series_type: str,
        frequency: str,
        length: int,
        complexity: str,
        data_quality: str,
        language: str,
        seasonality_types: List[str],
        trend_type: str,
        anomaly_types: List[str],
        domain_context: str,
        start_date: str
    ) -> Dict[str, Any]:
        """
        Attempt to fix common schema issues in generated time series.
        
        Args:
            timeseries: The time series dict to fix
            series_type: Expected series type
            frequency: Expected frequency
            length: Expected length
            complexity: Expected complexity
            data_quality: Expected data quality
            language: Expected language
            seasonality_types: Expected seasonality types
            trend_type: Expected trend type
            anomaly_types: Expected anomaly types
            domain_context: Expected domain context
            start_date: Expected start date
        
        Returns:
            Fixed time series dictionary
        """
        # Ensure required fields exist
        if "series_id" not in timeseries:
            timeseries["series_id"] = f"ts_{uuid.uuid4().hex[:12]}"
        
        if "domain" not in timeseries:
            timeseries["domain"] = SERIES_TYPE_TO_DOMAIN.get(series_type, "electricity")
        
        if "series_type" not in timeseries:
            timeseries["series_type"] = series_type
        
        if "frequency" not in timeseries:
            timeseries["frequency"] = frequency
        
        if "complexity" not in timeseries:
            timeseries["complexity"] = complexity
        
        if "data_quality" not in timeseries:
            timeseries["data_quality"] = data_quality
        
        if "language" not in timeseries:
            timeseries["language"] = language
        
        if "length" not in timeseries:
            if "target" in timeseries:
                timeseries["length"] = len(timeseries["target"])
            else:
                timeseries["length"] = length
        
        if "seasonality_types" not in timeseries:
            timeseries["seasonality_types"] = seasonality_types
        
        if "trend_type" not in timeseries:
            timeseries["trend_type"] = trend_type
        
        if "anomaly_types" not in timeseries:
            timeseries["anomaly_types"] = anomaly_types
        
        if "anomaly_indices" not in timeseries:
            timeseries["anomaly_indices"] = []
        
        if "domain_context" not in timeseries:
            timeseries["domain_context"] = domain_context
        
        if "start" not in timeseries:
            timeseries["start"] = start_date
        
        if "target" not in timeseries:
            timeseries["target"] = []
        
        if "metadata" not in timeseries:
            timeseries["metadata"] = {}
        
        return timeseries
    
    def generate_batch(
        self,
        count: int,
        series_types: Optional[List[str]] = None,
        domain: Optional[str] = None,
        length: int = 24,
        frequency: str = "1H",
        complexity_distribution: Optional[Dict[str, float]] = None,
        language: str = "en",
        continue_on_error: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple time series.
        
        Args:
            count: Number of time series to generate
            series_types: Optional list of series types (cycled if fewer than count)
            domain: Optional domain filter (e.g., "electricity", "sensors")
            length: Number of data points per series
            frequency: Sampling frequency
            complexity_distribution: Optional custom complexity distribution
            language: Target language ("en" or "es")
            continue_on_error: If True, continue generating even if some fail
        
        Returns:
            List of generated time series dictionaries
        
        Raises:
            RuntimeError: If generation fails and continue_on_error is False
        """
        results: List[Dict[str, Any]] = []
        
        # Prepare series type list
        if series_types:
            type_cycle = [series_types[i % len(series_types)] for i in range(count)]
        elif domain:
            domain_types = self._domain_series_types.get(domain, ALL_SERIES_TYPES)
            type_cycle = [random.choice(domain_types) for _ in range(count)]
        else:
            type_cycle = [None] * count
        
        # Custom complexity distribution
        comp_dist = complexity_distribution or DEFAULT_DISTRIBUTION["complexity"]
        
        for i in range(count):
            logger.info(f"Generating time series {i + 1}/{count}...")
            
            # Select complexity based on distribution
            complexities = list(comp_dist.keys())
            weights = list(comp_dist.values())
            selected_complexity = random.choices(complexities, weights=weights)[0]
            
            try:
                series = self.generate_single(
                    series_type=type_cycle[i],
                    length=length,
                    frequency=frequency,
                    complexity=selected_complexity,
                    language=language
                )
                results.append(series)
                
            except Exception as e:
                logger.error(f"Failed to generate time series {i + 1}: {e}")
                if not continue_on_error:
                    raise
        
        logger.info(
            f"Batch generation complete: {len(results)}/{count} successful "
            f"({self._total_failed} total failures)"
        )
        
        return results
    
    def generate_balanced_dataset(
        self,
        total_count: int,
        length: int = 168,
        frequency: str = "1H",
        language: str = "en",
        include_all_series_types: bool = True,
        include_all_domains: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate a balanced dataset across series types, complexities, and domains.
        
        This method ensures good coverage across all dimensions for training data.
        
        Args:
            total_count: Total number of time series to generate
            length: Number of data points per series
            frequency: Sampling frequency
            language: Target language ("en" or "es")
            include_all_series_types: If True, ensures all 16 series types represented
            include_all_domains: If True, ensures all 4 domains are represented
        
        Returns:
            List of generated time series with balanced distribution
        """
        results: List[Dict[str, Any]] = []
        
        if include_all_series_types:
            # Calculate how many per series type (minimum 1 each)
            per_type = max(1, total_count // len(self._series_types))
            remaining = total_count - (per_type * len(self._series_types))
            
            for series_type in self._series_types:
                count_for_type = per_type
                if remaining > 0:
                    count_for_type += 1
                    remaining -= 1
                
                type_results = self.generate_batch(
                    count=count_for_type,
                    series_types=[series_type],
                    length=length,
                    frequency=frequency,
                    language=language,
                    continue_on_error=True
                )
                results.extend(type_results)
                
        elif include_all_domains:
            # Balance across domains
            domains = list(self._domain_series_types.keys())
            per_domain = max(1, total_count // len(domains))
            remaining = total_count - (per_domain * len(domains))
            
            for domain in domains:
                count_for_domain = per_domain
                if remaining > 0:
                    count_for_domain += 1
                    remaining -= 1
                
                domain_results = self.generate_batch(
                    count=count_for_domain,
                    domain=domain,
                    length=length,
                    frequency=frequency,
                    language=language,
                    continue_on_error=True
                )
                results.extend(domain_results)
        else:
            # Just generate with default distribution
            results = self.generate_batch(
                count=total_count,
                length=length,
                frequency=frequency,
                language=language,
                continue_on_error=True
            )
        
        logger.info(f"Generated balanced dataset: {len(results)} time series")
        return results
    
    @property
    def available_series_types(self) -> List[str]:
        """Get list of available series types."""
        return list(self._series_types)
    
    @property
    def series_type_count(self) -> int:
        """Get number of available series types."""
        return len(self._series_types)
    
    @property
    def available_domains(self) -> List[str]:
        """Get list of available domains."""
        return list(self._domain_series_types.keys())
    
    def get_series_types_for_domain(self, domain: str) -> List[str]:
        """
        Get series types available for a specific domain.
        
        Args:
            domain: Domain name (e.g., "electricity", "sensors")
        
        Returns:
            List of series types for the domain
        """
        return self._domain_series_types.get(domain, [])


