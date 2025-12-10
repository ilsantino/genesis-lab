"""
Future domain schemas - NOT IMPLEMENTED IN MVP.

These schemas define the architecture for planned features.
Do not import in production code until implementation is complete.

Status: ARCHITECTURE ONLY
Target Version: 2.0.0
"""

from __future__ import annotations

from typing import Optional, List
from datetime import datetime

# Import base from main schemas
try:
    from .schemas import (
        BaseSchema, 
        Metadata, 
        PYDANTIC_V2,
        ensure_timezone_aware
    )
except ImportError:
    from schemas import (
        BaseSchema, 
        Metadata, 
        PYDANTIC_V2,
        ensure_timezone_aware
    )

if PYDANTIC_V2:
    from pydantic import Field, field_validator
    from typing import Literal
else:
    from pydantic import Field, validator
    from typing_extensions import Literal

# ============================================================================
# DOMAIN 3: TABULAR FINANCIAL (PLANNED)
# ============================================================================

TransactionType = Literal[
    "purchase",
    "withdrawal",
    "deposit",
    "transfer",
    "payment"
]

MerchantCategory = Literal[
    "retail",
    "groceries",
    "restaurants",
    "travel",
    "utilities",
    "entertainment",
    "healthcare",
    "education",
    "other"
]


class FinancialTransaction(BaseSchema):
    """
    Financial transaction schema.
    
    Status: ARCHITECTURE ONLY - NOT IMPLEMENTED
    
    This schema defines the structure for synthetic financial transactions
    that will be generated in version 2.0.0.
    
    Planned features:
    - Fraud pattern injection
    - Realistic merchant distribution
    - Account behavior modeling
    - Cross-account relationship graphs
    """
    
    transaction_id: str
    account_id: str
    transaction_type: TransactionType
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    timestamp: datetime
    merchant_category: Optional[MerchantCategory] = None
    merchant_name: Optional[str] = None
    
    # Fraud detection labels
    is_fraudulent: bool = False
    fraud_type: Optional[str] = None  # e.g., "card_theft", "account_takeover"
    risk_score: float = Field(default=0.0, ge=0, le=1)
    
    # Geographic data
    transaction_country: str = "US"
    transaction_city: Optional[str] = None
    
    metadata: Metadata = Field(default_factory=dict)
    
    if PYDANTIC_V2:
        @field_validator('timestamp')
        @classmethod
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)
        
        @field_validator('risk_score')
        @classmethod
        def validate_fraud_consistency(cls, v: float, info) -> float:
            # In v2, we'd use info.data to access other fields
            return v
    else:
        @validator('timestamp')
        def ensure_tz_aware(cls, v: datetime) -> datetime:
            return ensure_timezone_aware(v)


class FinancialAccount(BaseSchema):
    """
    Financial account schema (companion to FinancialTransaction).
    
    Status: ARCHITECTURE ONLY - NOT IMPLEMENTED
    """
    
    account_id: str
    account_type: Literal["checking", "savings", "credit", "investment"]
    balance: float = Field(default=0.0)
    credit_limit: Optional[float] = None
    
    # Account holder demographics (for bias testing)
    holder_age_bracket: Optional[Literal["18-25", "26-35", "36-50", "51-65", "65+"]] = None
    holder_income_bracket: Optional[Literal["low", "medium", "high"]] = None
    
    account_opened_date: datetime
    is_active: bool = True
    
    metadata: Metadata = Field(default_factory=dict)


# ============================================================================
# DOMAIN 4: MEDICAL RECORDS (PLANNED - v3.0.0)
# ============================================================================

# Placeholder for future medical/healthcare domain
# Will require HIPAA-compliant synthetic data generation patterns

# ============================================================================
# VERSION INFO
# ============================================================================

FUTURE_SCHEMA_VERSION = "0.1.0"  # Pre-release
TARGET_RELEASE = "2.0.0"


