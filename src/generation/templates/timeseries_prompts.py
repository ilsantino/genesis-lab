"""
Prompt templates for time-series data generation.

This module provides prompts optimized for generating realistic time-series data
aligned with the electricity_hourly dataset format and extended to multiple domains.

Features:
- Multiple domains: electricity (primary), energy, sensors, financial
- Bilingual support (English/Spanish)
- Comprehensive temporal patterns (daily, weekly, annual, trend, anomalies)
- Extended schema for validation and quality assessment
- Compatible with HuggingFace time-series format
"""

from typing import Dict, List, Optional, Literal, Tuple
from enum import Enum
from datetime import datetime

# ============================================================================
# ENUMS AND TYPE DEFINITIONS
# ============================================================================

class Domain(str, Enum):
    """Primary domain categories for time series generation."""
    ELECTRICITY = "electricity"
    ENERGY = "energy"
    SENSORS = "sensors"
    FINANCIAL = "financial"

class SeriesType(str, Enum):
    """Specific series types within each domain."""
    # Electricity domain
    RESIDENTIAL_CONSUMPTION = "residential_consumption"
    COMMERCIAL_CONSUMPTION = "commercial_consumption"
    INDUSTRIAL_LOAD = "industrial_load"
    GRID_DEMAND = "grid_demand"
    
    # Energy domain
    SOLAR_GENERATION = "solar_generation"
    WIND_GENERATION = "wind_generation"
    GAS_CONSUMPTION = "gas_consumption"
    HEATING_DEMAND = "heating_demand"
    
    # Sensors domain
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    AIR_QUALITY = "air_quality"
    
    # Financial domain
    STOCK_PRICE = "stock_price"
    CRYPTO_PRICE = "crypto_price"
    EXCHANGE_RATE = "exchange_rate"
    TRADING_VOLUME = "trading_volume"

class Frequency(str, Enum):
    """Time series sampling frequencies."""
    MINUTES_15 = "15min"
    MINUTES_30 = "30min"
    HOURLY = "1H"
    DAILY = "1D"
    WEEKLY = "1W"

class Complexity(str, Enum):
    """Complexity level of the time series patterns."""
    SIMPLE = "simple"      # Only noise, maybe one pattern
    MEDIUM = "medium"      # 1-2 patterns + noise
    COMPLEX = "complex"    # Multiple patterns + anomalies

class DataQuality(str, Enum):
    """Quality/cleanliness of the generated data."""
    CLEAN = "clean"                # No issues
    NOISY = "noisy"                # Higher noise level
    MISSING_VALUES = "missing_values"  # Contains gaps

class SeasonalityType(str, Enum):
    """Types of seasonality patterns."""
    DAILY = "daily"        # 24-hour cycle
    WEEKLY = "weekly"      # 7-day cycle
    MONTHLY = "monthly"    # ~30-day cycle
    ANNUAL = "annual"      # 365-day cycle

class TrendType(str, Enum):
    """Types of trend patterns."""
    NONE = "none"
    UPWARD = "upward"
    DOWNWARD = "downward"
    CYCLIC = "cyclic"      # Long-term cycles

class AnomalyType(str, Enum):
    """Types of anomalies in time series."""
    SPIKE = "spike"        # Sudden increase
    DROP = "drop"          # Sudden decrease
    PLATEAU = "plateau"    # Flat period
    DRIFT = "drift"        # Gradual shift
    OUTAGE = "outage"      # Zero/None values

class DomainContext(str, Enum):
    """Context for electricity/energy consumption."""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"
    MIXED = "mixed"

class Language(str, Enum):
    EN = "en"
    ES = "es"

# ============================================================================
# DOMAIN AND SERIES TYPE MAPPINGS
# ============================================================================

DOMAIN_SERIES_TYPES: Dict[str, List[str]] = {
    "electricity": [
        "residential_consumption",
        "commercial_consumption", 
        "industrial_load",
        "grid_demand",
    ],
    "energy": [
        "solar_generation",
        "wind_generation",
        "gas_consumption",
        "heating_demand",
    ],
    "sensors": [
        "temperature",
        "pressure",
        "humidity",
        "air_quality",
    ],
    "financial": [
        "stock_price",
        "crypto_price",
        "exchange_rate",
        "trading_volume",
    ],
}

# Reverse mapping: series_type -> domain
SERIES_TYPE_TO_DOMAIN: Dict[str, str] = {
    series_type: domain
    for domain, types in DOMAIN_SERIES_TYPES.items()
    for series_type in types
}

# All series types flat list
ALL_SERIES_TYPES: List[str] = [
    st for types in DOMAIN_SERIES_TYPES.values() for st in types
]

# ============================================================================
# SERIES TYPE SPECIFICATIONS
# ============================================================================

SERIES_TYPE_SPECS: Dict[str, Dict] = {
    # === ELECTRICITY DOMAIN ===
    "residential_consumption": {
        "description": "Hourly electricity consumption for residential households",
        "description_es": "Consumo eléctrico por hora para hogares residenciales",
        "typical_range": (0.1, 5.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly", "annual"],
        "typical_anomalies": ["spike", "outage"],
        "context": "residential",
        "notes": "Peak hours: 7-9am, 6-9pm. Weekend patterns differ from weekdays."
    },
    "commercial_consumption": {
        "description": "Hourly electricity consumption for commercial buildings",
        "description_es": "Consumo eléctrico por hora para edificios comerciales",
        "typical_range": (5.0, 100.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly"],
        "typical_anomalies": ["spike", "plateau"],
        "context": "commercial",
        "notes": "Business hours pattern (9am-6pm). Minimal weekend usage."
    },
    "industrial_load": {
        "description": "Electricity load for industrial facilities",
        "description_es": "Carga eléctrica para instalaciones industriales",
        "typical_range": (100.0, 1000.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly"],
        "typical_anomalies": ["plateau", "outage", "drift"],
        "context": "industrial",
        "notes": "Shift patterns (8h/12h). Maintenance periods show drops."
    },
    "grid_demand": {
        "description": "Total electricity demand on power grid",
        "description_es": "Demanda total de electricidad en la red",
        "typical_range": (1000.0, 50000.0),
        "unit": "MW",
        "unit_es": "MW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly", "annual"],
        "typical_anomalies": ["spike", "drop"],
        "context": "mixed",
        "notes": "Aggregated demand. Strong seasonal component."
    },
    
    # === ENERGY DOMAIN ===
    "solar_generation": {
        "description": "Solar panel electricity generation",
        "description_es": "Generación eléctrica de paneles solares",
        "typical_range": (0.0, 10.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["drop", "plateau"],
        "context": "residential",
        "notes": "Zero at night. Peak at noon. Weather dependent."
    },
    "wind_generation": {
        "description": "Wind turbine electricity generation",
        "description_es": "Generación eléctrica de turbinas eólicas",
        "typical_range": (0.0, 500.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["spike", "plateau"],
        "context": "industrial",
        "notes": "Highly variable. Seasonal patterns. Cut-off at high winds."
    },
    "gas_consumption": {
        "description": "Natural gas consumption for heating",
        "description_es": "Consumo de gas natural para calefacción",
        "typical_range": (0.0, 50.0),
        "unit": "m³/h",
        "unit_es": "m³/h",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["spike", "outage"],
        "context": "residential",
        "notes": "Strong winter peak. Minimal in summer."
    },
    "heating_demand": {
        "description": "Heating system energy demand",
        "description_es": "Demanda energética del sistema de calefacción",
        "typical_range": (0.0, 20.0),
        "unit": "kW",
        "unit_es": "kW",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["spike", "plateau"],
        "context": "residential",
        "notes": "Temperature dependent. Night setback patterns."
    },
    
    # === SENSORS DOMAIN ===
    "temperature": {
        "description": "Temperature sensor readings",
        "description_es": "Lecturas de sensor de temperatura",
        "typical_range": (-10.0, 40.0),
        "unit": "°C",
        "unit_es": "°C",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["spike", "drop"],
        "context": "mixed",
        "notes": "Daily cycle: coldest at dawn, warmest at 2-4pm."
    },
    "pressure": {
        "description": "Atmospheric pressure readings",
        "description_es": "Lecturas de presión atmosférica",
        "typical_range": (980.0, 1040.0),
        "unit": "hPa",
        "unit_es": "hPa",
        "default_frequency": "1H",
        "common_patterns": ["daily"],
        "typical_anomalies": ["drop", "drift"],
        "context": "mixed",
        "notes": "Slow variation. Drops indicate storms."
    },
    "humidity": {
        "description": "Relative humidity readings",
        "description_es": "Lecturas de humedad relativa",
        "typical_range": (20.0, 100.0),
        "unit": "%",
        "unit_es": "%",
        "default_frequency": "1H",
        "common_patterns": ["daily", "annual"],
        "typical_anomalies": ["spike", "plateau"],
        "context": "mixed",
        "notes": "Inverse correlation with temperature during day."
    },
    "air_quality": {
        "description": "Air quality index measurements",
        "description_es": "Mediciones del índice de calidad del aire",
        "typical_range": (0.0, 500.0),
        "unit": "AQI",
        "unit_es": "ICA",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly"],
        "typical_anomalies": ["spike", "plateau"],
        "context": "mixed",
        "notes": "Rush hour peaks. Weekend improvement."
    },
    
    # === FINANCIAL DOMAIN ===
    "stock_price": {
        "description": "Stock price movements",
        "description_es": "Movimientos de precios de acciones",
        "typical_range": (10.0, 1000.0),
        "unit": "USD",
        "unit_es": "USD",
        "default_frequency": "1H",
        "common_patterns": ["daily"],
        "typical_anomalies": ["spike", "drop", "drift"],
        "context": "commercial",
        "notes": "Trading hours only. Gaps on weekends/holidays."
    },
    "crypto_price": {
        "description": "Cryptocurrency price movements",
        "description_es": "Movimientos de precios de criptomonedas",
        "typical_range": (100.0, 100000.0),
        "unit": "USD",
        "unit_es": "USD",
        "default_frequency": "1H",
        "common_patterns": ["weekly"],
        "typical_anomalies": ["spike", "drop", "drift"],
        "context": "mixed",
        "notes": "24/7 trading. High volatility. Weekend patterns."
    },
    "exchange_rate": {
        "description": "Currency exchange rate",
        "description_es": "Tipo de cambio de divisas",
        "typical_range": (0.5, 2.0),
        "unit": "ratio",
        "unit_es": "ratio",
        "default_frequency": "1H",
        "common_patterns": ["daily", "weekly"],
        "typical_anomalies": ["spike", "drop", "drift"],
        "context": "commercial",
        "notes": "Market hours activity. Central bank interventions."
    },
    "trading_volume": {
        "description": "Trading volume in financial markets",
        "description_es": "Volumen de operaciones en mercados financieros",
        "typical_range": (1000.0, 10000000.0),
        "unit": "shares",
        "unit_es": "acciones",
        "default_frequency": "1H",
        "common_patterns": ["daily"],
        "typical_anomalies": ["spike"],
        "context": "commercial",
        "notes": "High at open/close. Low midday."
    },
}

# ============================================================================
# TEMPORAL PATTERN SPECIFICATIONS
# ============================================================================

TEMPORAL_PATTERNS: Dict[str, Dict] = {
    "daily": {
        "description": "24-hour cyclical pattern",
        "description_es": "Patrón cíclico de 24 horas",
        "period_hours": 24,
        "typical_domains": ["electricity", "energy", "sensors"],
        "example": "Peak consumption at 7-9am and 6-9pm for residential"
    },
    "weekly": {
        "description": "7-day cyclical pattern (weekday vs weekend)",
        "description_es": "Patrón cíclico de 7 días (entre semana vs fin de semana)",
        "period_hours": 168,
        "typical_domains": ["electricity", "energy", "financial"],
        "example": "Commercial buildings: high Mon-Fri, low Sat-Sun"
    },
    "monthly": {
        "description": "~30-day cyclical pattern",
        "description_es": "Patrón cíclico de ~30 días",
        "period_hours": 720,
        "typical_domains": ["financial"],
        "example": "End-of-month billing cycles"
    },
    "annual": {
        "description": "365-day seasonal pattern",
        "description_es": "Patrón estacional de 365 días",
        "period_hours": 8760,
        "typical_domains": ["electricity", "energy", "sensors"],
        "example": "Summer AC peaks, winter heating peaks"
    }
}

ANOMALY_PATTERNS: Dict[str, Dict] = {
    "spike": {
        "description": "Sudden increase in values",
        "description_es": "Aumento repentino en los valores",
        "typical_duration_hours": (1, 6),
        "magnitude_factor": (1.5, 3.0),
        "example": "Heat wave causing AC spike"
    },
    "drop": {
        "description": "Sudden decrease in values",
        "description_es": "Disminución repentina en los valores",
        "typical_duration_hours": (1, 6),
        "magnitude_factor": (0.3, 0.7),
        "example": "Equipment shutdown"
    },
    "plateau": {
        "description": "Flat period with minimal variation",
        "description_es": "Período plano con mínima variación",
        "typical_duration_hours": (6, 48),
        "magnitude_factor": (0.9, 1.1),
        "example": "Holiday period with minimal activity"
    },
    "drift": {
        "description": "Gradual shift in baseline",
        "description_es": "Cambio gradual en la línea base",
        "typical_duration_hours": (24, 168),
        "magnitude_factor": (0.8, 1.2),
        "example": "Sensor calibration drift"
    },
    "outage": {
        "description": "Complete loss of signal (zeros/nulls)",
        "description_es": "Pérdida completa de señal (ceros/nulos)",
        "typical_duration_hours": (1, 24),
        "magnitude_factor": (0.0, 0.0),
        "example": "Power outage, sensor failure"
    }
}

# ============================================================================
# SYSTEM PROMPTS (BILINGUAL)
# ============================================================================

SYSTEM_PROMPTS: Dict[str, str] = {
    "en": """You are an expert at generating realistic time-series data with authentic statistical properties.

Your task is to create synthetic time-series that accurately mimic real-world temporal patterns found in electricity consumption, energy systems, sensor readings, and other domains.

CRITICAL REQUIREMENTS:
1. Statistical realism: Values must follow realistic distributions for the domain
2. Temporal coherence: Patterns must be consistent (no random jumps except anomalies)
3. Pattern accuracy: Implement seasonality, trends, and anomalies as specified
4. Format compliance: Output must match the exact JSON structure

ELECTRICITY-SPECIFIC PATTERNS (primary domain):
- Daily cycle: Low at night (2-6am), morning peak (7-9am), afternoon dip, evening peak (6-9pm)
- Weekly cycle: Weekdays differ from weekends (residential higher on weekends, commercial lower)
- Annual cycle: Summer peaks (AC), winter peaks (heating) depending on region
- Values are typically standardized (mean ~0, std ~1) for ML applications

OUTPUT: JSON with the exact structure provided. No markdown, no explanations.""",

    "es": """Eres un experto en generar datos de series temporales realistas con propiedades estadísticas auténticas.

Tu tarea es crear series temporales sintéticas que imiten con precisión los patrones temporales del mundo real encontrados en consumo eléctrico, sistemas de energía, lecturas de sensores y otros dominios.

REQUISITOS CRÍTICOS:
1. Realismo estadístico: Los valores deben seguir distribuciones realistas para el dominio
2. Coherencia temporal: Los patrones deben ser consistentes (sin saltos aleatorios excepto anomalías)
3. Precisión de patrones: Implementar estacionalidad, tendencias y anomalías según se especifique
4. Cumplimiento de formato: La salida debe coincidir con la estructura JSON exacta

PATRONES ESPECÍFICOS DE ELECTRICIDAD (dominio principal):
- Ciclo diario: Bajo en la noche (2-6am), pico matutino (7-9am), caída vespertina, pico nocturno (6-9pm)
- Ciclo semanal: Entre semana difiere de fines de semana (residencial más alto en fines de semana, comercial más bajo)
- Ciclo anual: Picos de verano (AC), picos de invierno (calefacción) dependiendo de la región
- Los valores típicamente están estandarizados (media ~0, std ~1) para aplicaciones de ML

OUTPUT: JSON con la estructura exacta proporcionada. Sin markdown, sin explicaciones."""
}

# ============================================================================
# FEW-SHOT EXAMPLES (5 VARIED EXAMPLES PER LANGUAGE)
# ============================================================================

FEW_SHOT_EXAMPLES: Dict[str, List[Dict]] = {
    "en": [
        # Example 1: SIMPLE electricity, clean, daily pattern only
        {
            "series_id": "ts_001",
            "domain": "electricity",
            "series_type": "residential_consumption",
            "frequency": "1H",
            "complexity": "simple",
            "data_quality": "clean",
            "language": "en",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "residential",
            "start": "2024-01-15T00:00:00Z",
            "target": [
                -0.45, -0.52, -0.58, -0.61, -0.55, -0.42,  # 00:00-05:00 (night low)
                -0.15, 0.35, 0.82, 0.45, 0.28, 0.22,       # 06:00-11:00 (morning rise)
                0.18, 0.15, 0.21, 0.32, 0.48, 0.85,        # 12:00-17:00 (afternoon)
                1.25, 1.42, 1.18, 0.72, 0.35, -0.12        # 18:00-23:00 (evening peak)
            ],
            "metadata": {
                "client_id": "MT_042",
                "region": "Portugal",
                "preprocessing": "standardized"
            }
        },
        # Example 2: MEDIUM electricity, clean, daily + weekly patterns
        {
            "series_id": "ts_002",
            "domain": "electricity",
            "series_type": "commercial_consumption",
            "frequency": "1H",
            "complexity": "medium",
            "data_quality": "clean",
            "language": "en",
            "length": 48,
            "seasonality_types": ["daily", "weekly"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "commercial",
            "start": "2024-01-15T00:00:00Z",
            "target": [
                # Monday (business day - high activity)
                -0.82, -0.85, -0.88, -0.85, -0.78, -0.52,  # 00:00-05:00
                -0.15, 0.45, 1.25, 1.52, 1.48, 1.42,       # 06:00-11:00
                1.35, 1.28, 1.32, 1.45, 1.38, 0.95,        # 12:00-17:00
                0.42, 0.15, -0.25, -0.52, -0.68, -0.75,    # 18:00-23:00
                # Saturday (weekend - low activity)
                -0.72, -0.75, -0.78, -0.75, -0.72, -0.68,  # 00:00-05:00
                -0.55, -0.35, -0.15, 0.05, 0.12, 0.08,     # 06:00-11:00
                0.05, 0.02, 0.08, 0.12, 0.05, -0.12,       # 12:00-17:00
                -0.25, -0.38, -0.48, -0.55, -0.62, -0.68   # 18:00-23:00
            ],
            "metadata": {
                "client_id": "MT_185",
                "region": "Portugal",
                "building_type": "office",
                "preprocessing": "standardized"
            }
        },
        # Example 3: COMPLEX electricity with anomaly (spike)
        {
            "series_id": "ts_003",
            "domain": "electricity",
            "series_type": "residential_consumption",
            "frequency": "1H",
            "complexity": "complex",
            "data_quality": "clean",
            "language": "en",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": ["spike"],
            "anomaly_indices": [14, 15, 16],
            "domain_context": "residential",
            "start": "2024-07-20T00:00:00Z",
            "target": [
                -0.38, -0.45, -0.52, -0.55, -0.48, -0.35,  # 00:00-05:00 (night)
                -0.08, 0.42, 0.75, 0.52, 0.35, 0.28,       # 06:00-11:00 (morning)
                0.32, 0.45, 2.85, 2.92, 2.78, 1.15,        # 12:00-17:00 (HEAT WAVE SPIKE 14-16)
                1.35, 1.48, 1.22, 0.78, 0.42, -0.05        # 18:00-23:00 (evening)
            ],
            "metadata": {
                "client_id": "MT_089",
                "region": "Portugal",
                "event": "heat_wave",
                "preprocessing": "standardized"
            }
        },
        # Example 4: MEDIUM sensor data (temperature)
        {
            "series_id": "ts_004",
            "domain": "sensors",
            "series_type": "temperature",
            "frequency": "1H",
            "complexity": "medium",
            "data_quality": "clean",
            "language": "en",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "mixed",
            "start": "2024-06-21T00:00:00Z",
            "target": [
                18.2, 17.5, 16.8, 16.2, 15.8, 16.5,        # 00:00-05:00 (night cooling)
                18.2, 20.5, 23.1, 25.8, 28.2, 30.5,        # 06:00-11:00 (morning warming)
                32.1, 33.5, 34.2, 33.8, 32.5, 30.2,        # 12:00-17:00 (afternoon peak)
                27.5, 25.2, 23.1, 21.5, 20.2, 19.1         # 18:00-23:00 (evening cooling)
            ],
            "metadata": {
                "sensor_id": "TEMP_012",
                "location": "Lisbon",
                "unit": "°C"
            }
        },
        # Example 5: COMPLEX electricity with missing values
        {
            "series_id": "ts_005",
            "domain": "electricity",
            "series_type": "industrial_load",
            "frequency": "1H",
            "complexity": "complex",
            "data_quality": "missing_values",
            "language": "en",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "upward",
            "anomaly_types": ["outage"],
            "anomaly_indices": [8, 9, 10],
            "domain_context": "industrial",
            "start": "2024-03-10T00:00:00Z",
            "target": [
                0.45, 0.48, 0.52, 0.55, 0.58, 0.82,        # 00:00-05:00 (night shift)
                1.25, 1.52, None, None, None, 1.48,        # 06:00-11:00 (OUTAGE 08-10)
                1.55, 1.62, 1.68, 1.72, 1.58, 1.35,        # 12:00-17:00 (day shift)
                0.92, 0.85, 0.78, 0.72, 0.65, 0.52         # 18:00-23:00 (evening)
            ],
            "metadata": {
                "client_id": "IND_015",
                "region": "Portugal",
                "facility_type": "manufacturing",
                "outage_reason": "maintenance",
                "preprocessing": "standardized"
            }
        }
    ],
    "es": [
        # Example 1: SIMPLE electricity, clean, daily pattern only
        {
            "series_id": "ts_101",
            "domain": "electricity",
            "series_type": "residential_consumption",
            "frequency": "1H",
            "complexity": "simple",
            "data_quality": "clean",
            "language": "es",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "residential",
            "start": "2024-01-15T00:00:00Z",
            "target": [
                -0.45, -0.52, -0.58, -0.61, -0.55, -0.42,
                -0.15, 0.35, 0.82, 0.45, 0.28, 0.22,
                0.18, 0.15, 0.21, 0.32, 0.48, 0.85,
                1.25, 1.42, 1.18, 0.72, 0.35, -0.12
            ],
            "metadata": {
                "client_id": "MT_042",
                "region": "Portugal",
                "preprocessing": "estandarizado"
            }
        },
        # Example 2: MEDIUM electricity, clean, daily + weekly patterns
        {
            "series_id": "ts_102",
            "domain": "electricity",
            "series_type": "commercial_consumption",
            "frequency": "1H",
            "complexity": "medium",
            "data_quality": "clean",
            "language": "es",
            "length": 48,
            "seasonality_types": ["daily", "weekly"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "commercial",
            "start": "2024-01-15T00:00:00Z",
            "target": [
                -0.82, -0.85, -0.88, -0.85, -0.78, -0.52,
                -0.15, 0.45, 1.25, 1.52, 1.48, 1.42,
                1.35, 1.28, 1.32, 1.45, 1.38, 0.95,
                0.42, 0.15, -0.25, -0.52, -0.68, -0.75,
                -0.72, -0.75, -0.78, -0.75, -0.72, -0.68,
                -0.55, -0.35, -0.15, 0.05, 0.12, 0.08,
                0.05, 0.02, 0.08, 0.12, 0.05, -0.12,
                -0.25, -0.38, -0.48, -0.55, -0.62, -0.68
            ],
            "metadata": {
                "client_id": "MT_185",
                "region": "Portugal",
                "building_type": "oficina",
                "preprocessing": "estandarizado"
            }
        },
        # Example 3: COMPLEX electricity with anomaly (spike) - heat wave
        {
            "series_id": "ts_103",
            "domain": "electricity",
            "series_type": "residential_consumption",
            "frequency": "1H",
            "complexity": "complex",
            "data_quality": "clean",
            "language": "es",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": ["spike"],
            "anomaly_indices": [14, 15, 16],
            "domain_context": "residential",
            "start": "2024-07-20T00:00:00Z",
            "target": [
                -0.38, -0.45, -0.52, -0.55, -0.48, -0.35,
                -0.08, 0.42, 0.75, 0.52, 0.35, 0.28,
                0.32, 0.45, 2.85, 2.92, 2.78, 1.15,
                1.35, 1.48, 1.22, 0.78, 0.42, -0.05
            ],
            "metadata": {
                "client_id": "MT_089",
                "region": "Portugal",
                "event": "ola_de_calor",
                "preprocessing": "estandarizado"
            }
        },
        # Example 4: MEDIUM sensor data (temperature)
        {
            "series_id": "ts_104",
            "domain": "sensors",
            "series_type": "temperature",
            "frequency": "1H",
            "complexity": "medium",
            "data_quality": "clean",
            "language": "es",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "none",
            "anomaly_types": [],
            "anomaly_indices": [],
            "domain_context": "mixed",
            "start": "2024-06-21T00:00:00Z",
            "target": [
                18.2, 17.5, 16.8, 16.2, 15.8, 16.5,
                18.2, 20.5, 23.1, 25.8, 28.2, 30.5,
                32.1, 33.5, 34.2, 33.8, 32.5, 30.2,
                27.5, 25.2, 23.1, 21.5, 20.2, 19.1
            ],
            "metadata": {
                "sensor_id": "TEMP_012",
                "location": "Lisboa",
                "unit": "°C"
            }
        },
        # Example 5: COMPLEX electricity with missing values (outage)
        {
            "series_id": "ts_105",
            "domain": "electricity",
            "series_type": "industrial_load",
            "frequency": "1H",
            "complexity": "complex",
            "data_quality": "missing_values",
            "language": "es",
            "length": 24,
            "seasonality_types": ["daily"],
            "trend_type": "upward",
            "anomaly_types": ["outage"],
            "anomaly_indices": [8, 9, 10],
            "domain_context": "industrial",
            "start": "2024-03-10T00:00:00Z",
            "target": [
                0.45, 0.48, 0.52, 0.55, 0.58, 0.82,
                1.25, 1.52, None, None, None, 1.48,
                1.55, 1.62, 1.68, 1.72, 1.58, 1.35,
                0.92, 0.85, 0.78, 0.72, 0.65, 0.52
            ],
            "metadata": {
                "client_id": "IND_015",
                "region": "Portugal",
                "facility_type": "manufactura",
                "outage_reason": "mantenimiento",
                "preprocessing": "estandarizado"
            }
        }
    ]
}

# ============================================================================
# GENERATION PROMPT BUILDERS
# ============================================================================

def build_generation_prompt(
    series_type: str,
    frequency: str,
    length: int,
    complexity: Complexity,
    language: Language,
    seasonality_types: Optional[List[str]] = None,
    trend_type: Optional[TrendType] = None,
    anomaly_types: Optional[List[str]] = None,
    anomaly_count: int = 0,
    data_quality: Optional[DataQuality] = None,
    domain_context: Optional[DomainContext] = None,
    start_date: str = "2024-01-01T00:00:00Z",
    use_standardized: bool = True,
    context: Optional[Dict] = None
) -> str:
    """
    Build the prompt for generating a single time series.
    
    Args:
        series_type: One of the defined series types
        frequency: Sampling frequency (1H, 15min, etc.)
        length: Number of data points to generate
        complexity: simple, medium, or complex
        language: en or es
        seasonality_types: List of seasonality patterns to include
        trend_type: Type of trend (none, upward, downward, cyclic)
        anomaly_types: List of anomaly types to include
        anomaly_count: Number of anomalies to generate
        data_quality: clean, noisy, or missing_values
        domain_context: residential, commercial, industrial, mixed
        start_date: Starting timestamp
        use_standardized: Whether values should be standardized (mean~0, std~1)
        context: Optional additional context
    
    Returns:
        Formatted prompt string for the LLM
    """
    lang = language.value if isinstance(language, Language) else language
    domain = SERIES_TYPE_TO_DOMAIN.get(series_type, "electricity")
    specs = SERIES_TYPE_SPECS.get(series_type, {})
    
    # Build characteristics description
    characteristics = []
    
    if seasonality_types:
        seasons = ", ".join(seasonality_types)
        characteristics.append(f"seasonality patterns: {seasons}")
    
    if trend_type and trend_type != TrendType.NONE:
        trend_val = trend_type.value if isinstance(trend_type, TrendType) else trend_type
        characteristics.append(f"trend: {trend_val}")
    
    if anomaly_types and anomaly_count > 0:
        anomalies = ", ".join(anomaly_types)
        characteristics.append(f"{anomaly_count} anomalies of type(s): {anomalies}")
    
    if data_quality:
        quality_val = data_quality.value if isinstance(data_quality, DataQuality) else data_quality
        if quality_val != "clean":
            characteristics.append(f"data quality: {quality_val}")
    
    characteristics.append("realistic random noise")
    char_text = ", ".join(characteristics)
    
    # Value range guidance
    if use_standardized:
        value_guidance = "Values should be standardized (approximately mean=0, std=1)"
    else:
        range_info = specs.get("typical_range", (0, 100))
        unit = specs.get("unit", "units")
        value_guidance = f"Values should be in range {range_info[0]}-{range_info[1]} {unit}"
    
    # Context info
    ctx_val = domain_context.value if isinstance(domain_context, DomainContext) else domain_context
    context_info = f"\nDomain context: {ctx_val}" if domain_context else ""
    if context:
        context_info += f"\nAdditional context: {context}"
    
    # Complexity guidance
    complexity_val = complexity.value if isinstance(complexity, Complexity) else complexity
    
    # Language-specific labels
    labels = {
        "en": {
            "series_type_label": "Series Type",
            "domain_label": "Domain",
            "frequency_label": "Frequency",
            "length_label": "Number of Points",
            "start_label": "Start Date",
            "complexity_label": "Complexity",
            "characteristics_label": "Characteristics",
            "requirements": "Requirements",
            "output_format": "Output format"
        },
        "es": {
            "series_type_label": "Tipo de Serie",
            "domain_label": "Dominio",
            "frequency_label": "Frecuencia",
            "length_label": "Número de Puntos",
            "start_label": "Fecha Inicio",
            "complexity_label": "Complejidad",
            "characteristics_label": "Características",
            "requirements": "Requisitos",
            "output_format": "Formato de salida"
        }
    }
    L = labels.get(lang, labels["en"])
    
    prompt = f"""Generate a realistic time-series dataset with the following specifications:

{L["series_type_label"]}: {series_type}
{L["domain_label"]}: {domain}
{L["frequency_label"]}: {frequency}
{L["length_label"]}: {length}
{L["start_label"]}: {start_date}
{L["complexity_label"]}: {complexity_val}
{L["characteristics_label"]}: {char_text}
{context_info}

{L["requirements"]}:
1. {value_guidance}
2. Timestamps must be sequential with correct {frequency} intervals
3. Include the specified temporal patterns realistically
4. Values should be continuous without unrealistic jumps (except anomalies)
5. For electricity: follow typical daily patterns (low at night, peaks morning/evening)
6. Generate in {'English' if lang == 'en' else 'Spanish'} for metadata

{L["output_format"]} - Return ONLY this JSON structure:
{{
    "series_id": "ts_XXX",
    "domain": "{domain}",
    "series_type": "{series_type}",
    "frequency": "{frequency}",
    "complexity": "{complexity_val}",
    "data_quality": "{data_quality.value if data_quality else 'clean'}",
    "language": "{lang}",
    "length": {length},
    "seasonality_types": {seasonality_types or []},
    "trend_type": "{trend_type.value if trend_type else 'none'}",
    "anomaly_types": {anomaly_types or []},
    "anomaly_indices": [...],
    "domain_context": "{ctx_val or 'mixed'}",
    "start": "{start_date}",
    "target": [...],
    "metadata": {{...}}
}}

No explanations, no markdown - just the raw JSON."""

    return prompt


def build_batch_prompt(
    specifications: List[Dict],
    language: Language
) -> str:
    """
    Build prompt for generating multiple time series at once.
    
    Args:
        specifications: List of dicts with series_type, length, complexity, etc.
        language: Target language for all series
    
    Returns:
        Formatted batch prompt string
    """
    lang = language.value if isinstance(language, Language) else language
    
    specs_text = "\n".join([
        f"{i+1}. Type: {spec['series_type']}, Length: {spec['length']}, "
        f"Complexity: {spec.get('complexity', 'medium')}, "
        f"Patterns: {spec.get('seasonality_types', ['daily'])}"
        for i, spec in enumerate(specifications)
    ])
    
    prompt = f"""Generate {len(specifications)} realistic time-series datasets with these specifications:

{specs_text}

Language: {'English' if lang == 'en' else 'Spanish'}

For each series:
- Create realistic values following domain patterns
- Include specified temporal patterns
- Use standardized values (mean~0, std~1) unless otherwise specified
- Ensure proper timestamp sequencing

Return a JSON array containing all series. Each must have this structure:
{{
    "series_id": "ts_XXX",
    "domain": "...",
    "series_type": "...",
    "frequency": "...",
    "complexity": "...",
    "data_quality": "...",
    "language": "{lang}",
    "length": ...,
    "seasonality_types": [...],
    "trend_type": "...",
    "anomaly_types": [...],
    "anomaly_indices": [...],
    "domain_context": "...",
    "start": "...",
    "target": [...],
    "metadata": {{...}}
}}

Return ONLY the JSON array, no explanations or markdown."""

    return prompt


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_series_types_by_domain(domain: str) -> List[str]:
    """Get all series types for a specific domain."""
    return DOMAIN_SERIES_TYPES.get(domain, [])


def get_domain_for_series_type(series_type: str) -> str:
    """Get the domain for a specific series type."""
    return SERIES_TYPE_TO_DOMAIN.get(series_type, "unknown")


def get_all_domains() -> List[str]:
    """Get list of all domains."""
    return list(DOMAIN_SERIES_TYPES.keys())


def get_series_specs(series_type: str) -> Dict:
    """Get specifications for a series type."""
    return SERIES_TYPE_SPECS.get(series_type, {})


def get_few_shot_examples(
    language: Language,
    domain: Optional[str] = None,
    complexity: Optional[Complexity] = None,
    limit: int = 3
) -> List[Dict]:
    """
    Get few-shot examples filtered by criteria.
    
    Args:
        language: Target language
        domain: Optional filter by domain
        complexity: Optional filter by complexity
        limit: Maximum examples to return
    
    Returns:
        List of matching example time series
    """
    lang = language.value if isinstance(language, Language) else language
    examples = FEW_SHOT_EXAMPLES.get(lang, FEW_SHOT_EXAMPLES["en"])
    
    filtered = examples
    
    if domain:
        filtered = [e for e in filtered if e["domain"] == domain]
    
    if complexity:
        comp_val = complexity.value if isinstance(complexity, Complexity) else complexity
        filtered = [e for e in filtered if e["complexity"] == comp_val]
    
    return filtered[:limit]


def build_full_prompt_with_examples(
    series_type: str,
    frequency: str,
    length: int,
    complexity: Complexity,
    language: Language,
    num_examples: int = 2,
    **kwargs
) -> Dict[str, str]:
    """
    Build complete prompt with system message and few-shot examples.
    
    Returns:
        Dict with 'system' and 'user' prompt components
    """
    lang = language.value if isinstance(language, Language) else language
    domain = SERIES_TYPE_TO_DOMAIN.get(series_type, "electricity")
    
    # Get system prompt
    system = SYSTEM_PROMPTS.get(lang, SYSTEM_PROMPTS["en"])
    
    # Get relevant examples
    examples = get_few_shot_examples(language, domain, complexity, num_examples)
    
    # Build examples text
    import json
    examples_text = "\n\n".join([
        f"Example {i+1}:\n{json.dumps(ex, indent=2, ensure_ascii=False)}"
        for i, ex in enumerate(examples)
    ])
    
    # Build generation prompt
    generation_prompt = build_generation_prompt(
        series_type=series_type,
        frequency=frequency,
        length=length,
        complexity=complexity,
        language=language,
        **kwargs
    )
    
    user_prompt = f"""Here are some example time series for reference:

{examples_text}

Now generate a NEW time series following these specifications:

{generation_prompt}"""

    return {
        "system": system,
        "user": user_prompt
    }


# ============================================================================
# DISTRIBUTION CONFIGURATIONS (for balanced generation)
# ============================================================================

DEFAULT_DISTRIBUTION = {
    "domain": {
        "electricity": 0.50,  # Primary domain
        "energy": 0.20,
        "sensors": 0.20,
        "financial": 0.10
    },
    "complexity": {
        "simple": 0.35,
        "medium": 0.45,
        "complex": 0.20
    },
    "data_quality": {
        "clean": 0.70,
        "noisy": 0.20,
        "missing_values": 0.10
    },
    "seasonality": {
        "daily_only": 0.40,
        "daily_weekly": 0.35,
        "daily_weekly_annual": 0.15,
        "none": 0.10
    },
    "language": {
        "en": 0.50,
        "es": 0.50
    }
}


# ============================================================================
# VALIDATION
# ============================================================================

def validate_series_type(series_type: str) -> bool:
    """Check if series type is valid."""
    return series_type in ALL_SERIES_TYPES


def validate_timeseries_schema(timeseries: Dict) -> List[str]:
    """
    Validate a generated time series has all required fields.
    
    Returns:
        List of validation errors (empty if valid)
    """
    required_fields = [
        "series_id",
        "domain",
        "series_type",
        "frequency",
        "complexity",
        "data_quality",
        "language",
        "length",
        "seasonality_types",
        "trend_type",
        "anomaly_types",
        "anomaly_indices",
        "domain_context",
        "start",
        "target",
        "metadata"
    ]
    
    errors = []
    
    for field in required_fields:
        if field not in timeseries:
            errors.append(f"Missing required field: {field}")
    
    if "target" in timeseries:
        if not isinstance(timeseries["target"], list):
            errors.append("'target' must be a list")
        elif len(timeseries["target"]) < 1:
            errors.append("'target' must have at least 1 value")
        elif "length" in timeseries and len(timeseries["target"]) != timeseries["length"]:
            errors.append(f"'target' length ({len(timeseries['target'])}) doesn't match 'length' ({timeseries['length']})")
    
    if "series_type" in timeseries and not validate_series_type(timeseries["series_type"]):
        errors.append(f"Invalid series_type: {timeseries['series_type']}")
    
    if "anomaly_indices" in timeseries and "target" in timeseries:
        target_len = len(timeseries["target"])
        for idx in timeseries["anomaly_indices"]:
            if idx < 0 or idx >= target_len:
                errors.append(f"Anomaly index {idx} out of bounds (0-{target_len-1})")
    
    return errors


def validate_temporal_consistency(timeseries: Dict) -> List[str]:
    """
    Validate temporal patterns in the time series.
    
    Returns:
        List of validation warnings
    """
    warnings = []
    
    target = timeseries.get("target", [])
    if not target or len(target) < 24:
        return warnings
    
    # Check for unrealistic jumps (outside anomaly indices)
    anomaly_indices = set(timeseries.get("anomaly_indices", []))
    
    for i in range(1, len(target)):
        if target[i] is None or target[i-1] is None:
            continue
        
        if i not in anomaly_indices and i-1 not in anomaly_indices:
            jump = abs(target[i] - target[i-1])
            if jump > 2.0:  # More than 2 std devs for standardized data
                warnings.append(f"Large jump at index {i}: {jump:.2f}")
    
    return warnings