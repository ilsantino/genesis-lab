# DOMAIN 2: Time Series Generation (Archived)

**Status:** Archived  
**Last Updated:** January 2026  
**Reason:** Schema/implementation issues, AWS throttling, focus shifted to conversations

---

## Overview

This document archives the time series generation work that was attempted in GENESIS-LAB. The domain was designed to generate synthetic time series data compatible with the `electricity_hourly` HuggingFace dataset format, extended to multiple domains.

---

## What Was Built

### Architecture

```
src/generation/
├── timeseries_generator.py    # TimeSeriesGenerator class
└── templates/
    └── timeseries_prompts.py  # Prompts, schemas, validation

src/validation/
└── timeseries_quality.py      # Statistical validation (ADF, ACF)
```

### Supported Domains (4)

| Domain | Series Types | Weight |
|--------|-------------|--------|
| **electricity** | residential_consumption, commercial_consumption, industrial_load, grid_demand | 50% |
| **energy** | solar_generation, wind_generation, gas_consumption, heating_demand | 20% |
| **sensors** | temperature, pressure, humidity, air_quality | 20% |
| **financial** | stock_price, crypto_price, exchange_rate, trading_volume | 10% |

### Features Implemented

- 16 series types across 4 domains
- Configurable patterns:
  - Seasonality: daily, weekly, monthly, annual
  - Trends: none, upward, downward, cyclic
  - Anomalies: spike, drop, plateau, drift, outage
- Complexity levels: simple, medium, complex
- Data quality modes: clean, noisy, missing_values
- Bilingual metadata (English/Spanish)
- Standardized values (mean≈0, std≈1) for ML compatibility
- Few-shot examples for better LLM generation

### Output Schema

```python
{
    "series_id": "ts_XXX",
    "domain": "electricity" | "energy" | "sensors" | "financial",
    "series_type": str,              # One of 16 types
    "frequency": "15min" | "30min" | "1H" | "1D" | "1W",
    "complexity": "simple" | "medium" | "complex",
    "data_quality": "clean" | "noisy" | "missing_values",
    "language": "en" | "es",
    "length": int,
    "seasonality_types": ["daily", "weekly", ...],
    "trend_type": "none" | "upward" | "downward" | "cyclic",
    "anomaly_types": ["spike", "drop", ...],
    "anomaly_indices": [int, ...],
    "domain_context": "residential" | "commercial" | "industrial" | "mixed",
    "start": "2024-01-01T00:00:00Z",
    "target": [float, ...],          # The actual time series values
    "metadata": {...}
}
```

---

## Why It Failed

### Issue 1: Schema Mismatch (Critical)

**Problem:** The Pydantic schemas in `schemas.py` defined different series types than the prompts generator used.

```python
# schemas.py expected:
SeriesType = Literal[
    "sensor_temperature",
    "sensor_pressure",
    "stock_price",
    "energy_consumption",
    "network_traffic",
    "manufacturing_output"
]

# timeseries_prompts.py generated:
ALL_SERIES_TYPES = [
    "residential_consumption",
    "commercial_consumption",
    "temperature",
    "pressure",
    # ... etc (16 different values)
]
```

**Result:** Every successful LLM response failed Pydantic validation with:
```
14 validation errors for TimeSeries
series_type
  Input should be 'sensor_temperature', 'sensor_pressure'...
```

### Issue 2: Length Non-Compliance

**Problem:** Claude does not reliably generate arrays of exact length.

| Requested | Received |
|-----------|----------|
| 168 points | 180, 190, 192, 211 points |

The LLM consistently generated more points than requested. The fix function updated metadata but did not truncate/pad arrays.

### Issue 3: AWS Bedrock Throttling

**Problem:** High rate of `ThrottlingException` errors from AWS Bedrock.

```
ThrottlingException: Too many requests, please wait before trying again.
```

With 5-second delays between calls, approximately 50% of requests still failed due to throttling. The account quota was insufficient for the generation volume needed.

### Smoke Test Results

```
============================================================
RESULTS
============================================================

Generation Summary:
  Total:   0/20 successful (0%)
  Time:    6647.6s (332.4s avg)

Output Files:
  - data\synthetic\ts_smoke_stationary.json (0 series)
  - data\synthetic\ts_smoke_varied.json (0 series)
```

---

## Lessons Learned

1. **Schema First:** Always verify Pydantic schemas match prompt templates before testing
2. **LLM Limitations:** LLMs are unreliable for generating exact-length numeric arrays
3. **Post-Processing:** Need robust post-processing to fix LLM output (truncate, pad, validate)
4. **Rate Limiting:** AWS Bedrock quotas require careful planning; use longer delays (10-15s) or request quota increases
5. **Incremental Testing:** Test with 1-2 items before scaling to batches

---

## Future Recommendations

If time series generation is revisited:

1. **Fix Schema Alignment**
   - Update `schemas.py` to match the 16 series types from `timeseries_prompts.py`
   - Or simplify prompts to generate only the 6 types in schemas

2. **Add Length Enforcement**
   ```python
   def _fix_length(target: List[float], expected: int) -> List[float]:
       if len(target) > expected:
           return target[:expected]  # Truncate
       elif len(target) < expected:
           # Pad with interpolated values
           return target + [target[-1]] * (expected - len(target))
       return target
   ```

3. **Increase Rate Limiting**
   - Use 10-15 second delays between calls
   - Implement exponential backoff with jitter
   - Consider requesting AWS quota increase

4. **Consider Hybrid Approach**
   - Use LLM to generate parameters and patterns
   - Use algorithmic generation for actual values
   - Validate with statistical tests (ADF, ACF)

5. **Alternative Models**
   - Test with different Bedrock models (Claude Haiku for speed)
   - Consider specialized time series generation models

---

## Archived Code Reference

The following files were part of the time series implementation:

### Core Generator

```python
# timeseries_generator.py (key method)
class TimeSeriesGenerator(BaseGenerator):
    def generate_single(
        self,
        series_type: Optional[str] = None,
        length: int = 24,
        frequency: str = "1H",
        complexity: Optional[str] = None,
        stationary: bool = False
    ) -> Dict[str, Any]:
        # Build prompt with few-shot examples
        prompts = build_full_prompt_with_examples(
            series_type=selected_series_type,
            frequency=frequency,
            length=length,
            complexity=complexity_enum,
            language=language_enum,
            num_examples=num_examples,
            stationary=stationary
        )
        
        # Call LLM
        response = self.client.invoke_model(
            prompt=prompts["user"],
            system_prompt=prompts["system"],
            temperature=self._generation_params.temperature
        )
        
        # Parse and validate
        timeseries = self._parse_json_response(response)
        errors = validate_timeseries_schema(timeseries)
        # ... validation failed here
```

### Quality Validation

```python
# timeseries_quality.py (statistical validation)
class TimeSeriesQualityValidator:
    def validate(self, series: Dict) -> TimeSeriesQualityMetrics:
        target = series.get("target", [])
        
        # Stationarity test (ADF)
        stationarity_score = self._check_stationarity(target)
        
        # Autocorrelation analysis
        autocorr_score = self._check_autocorrelation(target)
        
        # Completeness check
        completeness_score = self._check_completeness(series)
        
        return TimeSeriesQualityMetrics(
            stationarity_score=stationarity_score,
            autocorrelation_score=autocorr_score,
            completeness_score=completeness_score,
            overall_quality_score=overall
        )
```

---

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Updated to conversation-only
- [ROADMAP.md](ROADMAP.md) - Time series removed from milestones
- [DEVLOG.md](DEVLOG.md) - Contains debugging history

---

*This domain may be revisited in a future version after addressing the identified issues.*
