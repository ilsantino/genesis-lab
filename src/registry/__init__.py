"""
Dataset registry module for tracking synthetic datasets.

Provides:
- DatasetRegistry: SQLite-based registry for datasets, metrics, and training runs
"""

from .database import DatasetRegistry

__all__ = ["DatasetRegistry"]


