"""
Visualization utilities for GENESIS LAB.

Provides helper functions for data analysis and visualization.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def load_conversations(file_path: str) -> List[Dict[str, Any]]:
    """
    Load conversations from a JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        List of conversation dictionaries
    """
    path = Path(file_path)
    if not path.exists():
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else []


def analyze_conversations(conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze a list of conversations and compute statistics.
    
    Args:
        conversations: List of conversation dictionaries
    
    Returns:
        Dictionary with analysis results
    """
    if not conversations:
        return {
            "count": 0,
            "intents": {},
            "sentiments": {},
            "complexities": {},
            "languages": {},
            "avg_turns": 0,
            "resolution_statuses": {},
        }
    
    # Extract data
    intents = [c.get("intent", "unknown") for c in conversations]
    sentiments = [c.get("sentiment", "unknown") for c in conversations]
    complexities = [c.get("complexity", "unknown") for c in conversations]
    languages = [c.get("language", "unknown") for c in conversations]
    turn_counts = [c.get("turn_count", 0) for c in conversations]
    resolution_statuses = [c.get("resolution_status", "unknown") for c in conversations]
    
    return {
        "count": len(conversations),
        "intents": dict(Counter(intents)),
        "sentiments": dict(Counter(sentiments)),
        "complexities": dict(Counter(complexities)),
        "languages": dict(Counter(languages)),
        "avg_turns": sum(turn_counts) / len(turn_counts) if turn_counts else 0,
        "resolution_statuses": dict(Counter(resolution_statuses)),
    }


def get_sentiment_percentages(sentiments: Dict[str, int]) -> Dict[str, float]:
    """
    Convert sentiment counts to percentages.
    
    Args:
        sentiments: Dict with sentiment counts
    
    Returns:
        Dict with sentiment percentages
    """
    total = sum(sentiments.values())
    if total == 0:
        return {}
    
    return {k: (v / total) * 100 for k, v in sentiments.items()}


def get_language_percentages(languages: Dict[str, int]) -> Dict[str, float]:
    """
    Convert language counts to percentages.
    
    Args:
        languages: Dict with language counts
    
    Returns:
        Dict with language percentages
    """
    total = sum(languages.values())
    if total == 0:
        return {}
    
    return {k: (v / total) * 100 for k, v in languages.items()}


def get_top_intents(intents: Dict[str, int], top_n: int = 10) -> Dict[str, int]:
    """
    Get top N intents by count.
    
    Args:
        intents: Dict with intent counts
        top_n: Number of top intents to return
    
    Returns:
        Dict with top N intents
    """
    sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_intents[:top_n])


def conversations_to_dataframe(conversations: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Convert conversations to a pandas DataFrame.
    
    Args:
        conversations: List of conversation dictionaries
    
    Returns:
        DataFrame with conversation metadata
    """
    if not conversations:
        return pd.DataFrame()
    
    records = []
    for conv in conversations:
        records.append({
            "conversation_id": conv.get("conversation_id", ""),
            "intent": conv.get("intent", ""),
            "category": conv.get("category", ""),
            "sentiment": conv.get("sentiment", ""),
            "complexity": conv.get("complexity", ""),
            "language": conv.get("language", ""),
            "turn_count": conv.get("turn_count", 0),
            "resolution_status": conv.get("resolution_status", ""),
            "emotion_arc": conv.get("customer_emotion_arc", ""),
        })
    
    return pd.DataFrame(records)


def get_quality_summary(quality_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Generate a summary of quality metrics.
    
    Args:
        quality_metrics: Dict with quality scores
    
    Returns:
        Summary with scores and status
    """
    thresholds = {
        "completeness": 0.95,
        "consistency": 0.90,
        "realism": 0.85,
        "diversity": 0.80,
    }
    
    summary = {}
    for metric, value in quality_metrics.items():
        threshold = thresholds.get(metric, 0.8)
        summary[metric] = {
            "value": value,
            "percentage": value * 100,
            "status": "good" if value >= threshold else "warning" if value >= threshold * 0.8 else "error",
            "threshold": threshold,
        }
    
    # Calculate overall
    if quality_metrics:
        overall = sum(quality_metrics.values()) / len(quality_metrics)
        summary["overall"] = {
            "value": overall,
            "percentage": overall * 100,
            "status": "good" if overall >= 0.85 else "warning" if overall >= 0.70 else "error",
        }
    
    return summary


def get_bias_summary(bias_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of bias metrics.
    
    Args:
        bias_metrics: Dict with bias detection results
    
    Returns:
        Summary with findings and recommendations
    """
    summary = {
        "severity": bias_metrics.get("severity", "unknown"),
        "findings": [],
        "recommendations": [],
    }
    
    # Check sentiment balance
    sentiment_dist = bias_metrics.get("sentiment_distribution", {})
    if sentiment_dist:
        max_sentiment = max(sentiment_dist.values()) if sentiment_dist else 0
        if max_sentiment > 0.5:
            summary["findings"].append("Sentiment distribution is imbalanced")
            summary["recommendations"].append("Generate more diverse sentiment samples")
    
    # Check intent coverage
    intent_coverage = bias_metrics.get("intent_coverage", 0)
    if intent_coverage < 0.5:
        summary["findings"].append(f"Low intent coverage: {intent_coverage:.1%}")
        summary["recommendations"].append("Target underrepresented intents in generation")
    
    # Check language balance
    language_dist = bias_metrics.get("language_distribution", {})
    if language_dist:
        lang_values = list(language_dist.values())
        if len(lang_values) > 1 and max(lang_values) > 0.8:
            summary["findings"].append("Language distribution is imbalanced")
            summary["recommendations"].append("Generate more samples in underrepresented languages")
    
    return summary


def format_number(value: float, decimals: int = 1) -> str:
    """Format a number for display."""
    if value >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    else:
        return f"{value:.{decimals}f}"


def get_sample_conversations(
    conversations: List[Dict[str, Any]],
    n: int = 5,
    intent: Optional[str] = None,
    sentiment: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Get sample conversations with optional filtering.
    
    Args:
        conversations: List of all conversations
        n: Number of samples to return
        intent: Filter by intent
        sentiment: Filter by sentiment
    
    Returns:
        List of sample conversations
    """
    filtered = conversations
    
    if intent:
        filtered = [c for c in filtered if c.get("intent") == intent]
    
    if sentiment:
        filtered = [c for c in filtered if c.get("sentiment") == sentiment]
    
    return filtered[:n]


def compare_datasets(
    dataset1: List[Dict[str, Any]],
    dataset2: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compare two datasets and compute similarity metrics.
    
    Args:
        dataset1: First dataset
        dataset2: Second dataset
    
    Returns:
        Comparison results
    """
    analysis1 = analyze_conversations(dataset1)
    analysis2 = analyze_conversations(dataset2)
    
    # Compare intent distributions
    intents1 = set(analysis1["intents"].keys())
    intents2 = set(analysis2["intents"].keys())
    
    intent_overlap = len(intents1 & intents2) / max(len(intents1 | intents2), 1)
    
    # Compare sentiment distributions
    sent1 = analysis1["sentiments"]
    sent2 = analysis2["sentiments"]
    
    sentiment_diff = {}
    all_sentiments = set(sent1.keys()) | set(sent2.keys())
    total1 = sum(sent1.values()) or 1
    total2 = sum(sent2.values()) or 1
    
    for sent in all_sentiments:
        pct1 = (sent1.get(sent, 0) / total1) * 100
        pct2 = (sent2.get(sent, 0) / total2) * 100
        sentiment_diff[sent] = {"dataset1": pct1, "dataset2": pct2, "diff": abs(pct1 - pct2)}
    
    return {
        "dataset1_count": analysis1["count"],
        "dataset2_count": analysis2["count"],
        "intent_overlap": intent_overlap,
        "sentiment_comparison": sentiment_diff,
        "avg_turns_diff": abs(analysis1["avg_turns"] - analysis2["avg_turns"]),
        "analysis1": analysis1,
        "analysis2": analysis2,
    }
