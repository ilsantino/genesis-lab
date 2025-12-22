"""
Time series quality validation module.

Validates synthetic time series data against statistical metrics and reference data.

Metrics:
- Stationarity: ADF test (p < 0.05 = stationary)
- Autocorrelation: Lag-1 ACF comparison with reference
- Completeness: No NaN/Inf values, correct length
- Consistency: Values within expected bounds

Usage:
    uv run python -m src.validation.timeseries_quality
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from statsmodels.tsa.stattools import adfuller, acf

from src.generation.schemas import TimeSeriesQualityMetrics


__all__ = ["TimeSeriesValidator"]


class TimeSeriesValidator:
    """
    Validates quality of synthetic time series data.
    
    Compares synthetic series against reference data (electricity_hourly)
    using statistical tests and distributional comparisons.
    
    Example:
        >>> validator = TimeSeriesValidator()
        >>> series_list = json.load(open("data/synthetic/ts_smoke_test.json"))
        >>> metrics = validator.compute_overall_score(series_list)
        >>> print(f"Quality: {metrics.overall_quality_score:.1f}/100")
    """
    
    # Validation thresholds
    EXPECTED_LENGTH = 24  # Default hourly series length
    VALUE_BOUNDS = (-10.0, 10.0)  # Expected range for standardized values
    ADF_PVALUE_THRESHOLD = 0.05  # p < 0.05 = stationary
    
    def __init__(
        self,
        reference_path: str = "data/reference/timeseries_reference.json",
        expected_length: int = 24
    ):
        """
        Initialize validator with reference data.
        
        Args:
            reference_path: Path to reference time series dataset
            expected_length: Expected length of each series
        """
        self._reference_path = Path(reference_path)
        self._reference_data: Optional[List[Dict]] = None
        self._reference_acf_stats: Optional[Dict[str, float]] = None
        self.EXPECTED_LENGTH = expected_length
        self._issues: List[str] = []
        self._warnings: List[str] = []
        
        self._load_reference_data()
    
    def _load_reference_data(self) -> None:
        """Load and process reference dataset."""
        if not self._reference_path.exists():
            self._warnings.append(f"Reference data not found: {self._reference_path}")
            return
        
        try:
            with open(self._reference_path, "r", encoding="utf-8") as f:
                self._reference_data = json.load(f)
            
            # Compute reference ACF statistics
            if self._reference_data:
                acf_values = []
                for series in self._reference_data[:50]:  # Sample for efficiency
                    values = series.get("values", series.get("target", []))
                    if len(values) >= 10:
                        try:
                            acf_result = acf(values[:min(len(values), 100)], nlags=5, fft=True)
                            acf_values.append(acf_result[1])  # Lag-1 ACF
                        except Exception:
                            pass
                
                if acf_values:
                    self._reference_acf_stats = {
                        "mean": float(np.mean(acf_values)),
                        "std": float(np.std(acf_values)),
                        "min": float(np.min(acf_values)),
                        "max": float(np.max(acf_values))
                    }
                    
        except Exception as e:
            self._warnings.append(f"Error loading reference data: {e}")
    
    def _extract_values(self, series: Dict) -> List[float]:
        """Extract numeric values from a series dict."""
        values = series.get("target", series.get("values", []))
        if not isinstance(values, list):
            return []
        return [v for v in values if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v)]
    
    def validate_basic_stats(self, series_list: List[Dict]) -> Dict[str, float]:
        """
        Check basic properties of each series.
        
        Validates:
        - No NaN or Inf values
        - Correct length
        - Values within bounds
        
        Args:
            series_list: List of time series dictionaries
        
        Returns:
            Dict with completeness, length_correct, bounds_ok scores (0-1)
        """
        if not series_list:
            return {"completeness": 0.0, "length_correct": 0.0, "bounds_ok": 0.0}
        
        n = len(series_list)
        complete_count = 0
        length_correct_count = 0
        bounds_ok_count = 0
        
        for series in series_list:
            values = self._extract_values(series)
            raw_values = series.get("target", series.get("values", []))
            
            # Check completeness (no NaN/Inf, all numeric)
            if len(values) == len(raw_values) and len(values) > 0:
                complete_count += 1
            
            # Check length
            if len(values) == self.EXPECTED_LENGTH:
                length_correct_count += 1
            elif len(values) > 0:
                self._warnings.append(f"Series length {len(values)} != expected {self.EXPECTED_LENGTH}")
            
            # Check bounds
            if values:
                min_val, max_val = min(values), max(values)
                if self.VALUE_BOUNDS[0] <= min_val and max_val <= self.VALUE_BOUNDS[1]:
                    bounds_ok_count += 1
        
        return {
            "completeness": complete_count / n,
            "length_correct": length_correct_count / n,
            "bounds_ok": bounds_ok_count / n
        }
    
    def validate_stationarity(self, series_list: List[Dict]) -> float:
        """
        Run Augmented Dickey-Fuller test on each series.
        
        A p-value < 0.05 indicates the series is stationary (no unit root).
        
        Args:
            series_list: List of time series dictionaries
        
        Returns:
            Fraction of series that are stationary (0-1)
        """
        if not series_list:
            return 0.0
        
        stationary_count = 0
        valid_count = 0
        
        for series in series_list:
            values = self._extract_values(series)
            
            if len(values) < 10:
                continue
            
            valid_count += 1
            
            try:
                # Run ADF test
                result = adfuller(values, autolag="AIC")
                p_value = result[1]
                
                if p_value < self.ADF_PVALUE_THRESHOLD:
                    stationary_count += 1
                    
            except Exception as e:
                self._warnings.append(f"ADF test failed: {e}")
        
        return stationary_count / valid_count if valid_count > 0 else 0.0
    
    def validate_autocorrelation(self, series_list: List[Dict]) -> float:
        """
        Compute lag-1 autocorrelation and compare to reference.
        
        Calculates ACF at lag 1 for each series and compares the distribution
        to the reference dataset's ACF distribution.
        
        Args:
            series_list: List of time series dictionaries
        
        Returns:
            Similarity score (0-1) based on ACF comparison
        """
        if not series_list:
            return 0.0
        
        synthetic_acf = []
        
        for series in series_list:
            values = self._extract_values(series)
            
            if len(values) < 10:
                continue
            
            try:
                acf_result = acf(values, nlags=5, fft=True)
                synthetic_acf.append(acf_result[1])  # Lag-1 ACF
            except Exception:
                pass
        
        if not synthetic_acf:
            return 0.0
        
        # If we have reference stats, compare distributions
        if self._reference_acf_stats:
            ref_mean = self._reference_acf_stats["mean"]
            ref_std = self._reference_acf_stats["std"]
            
            synth_mean = float(np.mean(synthetic_acf))
            synth_std = float(np.std(synthetic_acf))
            
            # Score based on how close means are (within 2 std)
            mean_diff = abs(synth_mean - ref_mean)
            mean_score = max(0, 1 - mean_diff / (2 * max(ref_std, 0.1)))
            
            # Score based on std similarity
            std_ratio = min(synth_std, ref_std) / max(synth_std, ref_std, 0.01)
            
            return (mean_score + std_ratio) / 2
        else:
            # Without reference, just check ACF is reasonable (between -1 and 1)
            valid_acf = [a for a in synthetic_acf if -1 <= a <= 1]
            return len(valid_acf) / len(synthetic_acf) if synthetic_acf else 0.0
    
    def compute_overall_score(self, series_list: List[Dict]) -> TimeSeriesQualityMetrics:
        """
        Run all validations and compute overall quality score.
        
        Weighting:
        - Basic stats (completeness + consistency): 30%
        - Stationarity: 35%
        - Autocorrelation: 35%
        
        Args:
            series_list: List of time series dictionaries
        
        Returns:
            TimeSeriesQualityMetrics with all scores
        """
        self._issues = []
        self._warnings = []
        
        # Basic stats
        basic_stats = self.validate_basic_stats(series_list)
        completeness = basic_stats["completeness"]
        consistency = (basic_stats["length_correct"] + basic_stats["bounds_ok"]) / 2
        
        # Stationarity
        stationarity = self.validate_stationarity(series_list)
        
        # Autocorrelation
        autocorrelation = self.validate_autocorrelation(series_list)
        
        # Overall score (weighted average)
        basic_weight = 0.30
        stationarity_weight = 0.35
        acf_weight = 0.35
        
        overall_raw = (
            basic_weight * (completeness + consistency) / 2 +
            stationarity_weight * stationarity +
            acf_weight * autocorrelation
        )
        overall_score = overall_raw * 100
        
        # Collect issues
        if completeness < 0.9:
            self._issues.append(f"Low completeness: {completeness:.1%}")
        if consistency < 0.9:
            self._issues.append(f"Low consistency: {consistency:.1%}")
        if stationarity < 0.5:
            self._issues.append(f"Many non-stationary series: {stationarity:.1%} stationary")
        if overall_score < 70:
            self._issues.append(f"Overall quality below threshold: {overall_score:.1f}/100")
        
        return TimeSeriesQualityMetrics(
            stationarity_score=stationarity,
            autocorrelation_score=autocorrelation,
            completeness_score=completeness,
            consistency_score=consistency,
            overall_quality_score=overall_score,
            issues_found=self._issues.copy(),
            warnings=self._warnings.copy(),
            metadata={
                "num_series": len(series_list),
                "expected_length": self.EXPECTED_LENGTH,
                "reference_loaded": self._reference_data is not None,
                "basic_completeness": basic_stats["completeness"],
                "basic_length_correct": basic_stats["length_correct"],
                "basic_bounds_ok": basic_stats["bounds_ok"]
            }
        )
    
    def print_report(self, metrics: TimeSeriesQualityMetrics) -> None:
        """Print a formatted quality report."""
        print("\n" + "=" * 60)
        print("TIME SERIES QUALITY REPORT")
        print("=" * 60)
        
        num_series = metrics.metadata.get("num_series", 0)
        print(f"\nDataset: {num_series} series")
        print(f"Expected length: {metrics.metadata.get('expected_length', 'N/A')}")
        print(f"Reference loaded: {metrics.metadata.get('reference_loaded', False)}")
        
        print("\n" + "-" * 40)
        print("SCORES")
        print("-" * 40)
        
        def bar(score, width=20):
            filled = int(score * width)
            return "#" * filled + "-" * (width - filled)
        
        print(f"  Completeness:    {metrics.completeness_score:.2f} [{bar(metrics.completeness_score)}]")
        print(f"  Consistency:     {metrics.consistency_score:.2f} [{bar(metrics.consistency_score)}]")
        print(f"  Stationarity:    {metrics.stationarity_score:.2f} [{bar(metrics.stationarity_score)}]")
        print(f"  Autocorrelation: {metrics.autocorrelation_score:.2f} [{bar(metrics.autocorrelation_score)}]")
        
        print("\n" + "-" * 40)
        overall_bar = bar(metrics.overall_quality_score / 100, 30)
        print(f"  OVERALL: {metrics.overall_quality_score:.1f}/100 [{overall_bar}]")
        print("-" * 40)
        
        if metrics.issues_found:
            print("\n  ISSUES:")
            for issue in metrics.issues_found:
                print(f"    - {issue}")
        
        if metrics.warnings:
            print("\n  WARNINGS:")
            for warning in metrics.warnings[:5]:  # Limit to 5
                print(f"    - {warning}")
        
        print("\n" + "=" * 60)


def main():
    """Run time series validation on synthetic data."""
    import sys
    
    # Try multiple paths
    synthetic_paths = [
        Path("data/synthetic/ts_smoke_test.json"),
        Path("data/synthetic/ts_smoke_v2.json"),
        Path("data/synthetic/timeseries_smoke_test.json"),
    ]
    
    synthetic_path = None
    for path in synthetic_paths:
        if path.exists():
            synthetic_path = path
            break
    
    if not synthetic_path:
        print("[ERROR] No synthetic time series data found.")
        print("  Run scripts/smoke_test_ts.py first.")
        return 1
    
    print(f"Loading: {synthetic_path}")
    
    with open(synthetic_path, "r", encoding="utf-8") as f:
        series_list = json.load(f)
    
    print(f"Loaded {len(series_list)} series")
    
    # Validate
    validator = TimeSeriesValidator(expected_length=24)
    metrics = validator.compute_overall_score(series_list)
    
    # Print report
    validator.print_report(metrics)
    
    # Return exit code based on quality
    if metrics.overall_quality_score >= 70:
        print("\n[OK] Time series quality is acceptable")
        return 0
    else:
        print("\n[FAIL] Time series quality below threshold")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

