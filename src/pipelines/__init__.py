"""
Pipeline orchestrators for end-to-end workflows.

Provides automated pipelines for:
- Customer Service: Generation, validation, training
"""

from .customer_service_pipeline import CustomerServicePipeline

__all__ = ["CustomerServicePipeline"]
