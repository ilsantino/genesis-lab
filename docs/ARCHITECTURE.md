# GENESIS-LAB — Architecture Overview

Este documento describe la arquitectura técnica de GENESIS-LAB, su organización interna, responsabilidades por módulo, principios de diseño y componentes principales. Su propósito es servir como referencia para el desarrollo, mantenimiento y escalamiento del proyecto.

---

## 1. Objetivo de la arquitectura

GENESIS-LAB está diseñado como un sistema modular para:

- **Generación de datos sintéticos** utilizando modelos de AWS Bedrock
- **Validación de calidad y sesgos**
- **Registro y manejo de metadatos** de datasets generados
- **Entrenamiento ligero de modelos** cuando sea necesario
- **Interacción mediante interfaz** basada en Streamlit
- **Futura integración** con agentes de IA y pipelines automatizados

La arquitectura prioriza **claridad**, **mantenibilidad**, **extensibilidad** y **separación estricta de responsabilidades**.

---

## 2. Estructura general del proyecto

```
GENESIS-LAB/
├── .github/
├── .venv/
├── data/
│   ├── raw/
│   ├── synthetic/
│   └── reference/
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEVLOG.md
│   ├── PROJECTSTATUS.md
│   ├── ROADMAP.md
│   └── TDR.md
├── logs/
├── models/
├── notebooks/
├── src/
│   ├── generation/
│   │   ├── templates/
│   │   │   └── customer_service_prompts.py
│   │   ├── generator.py
│   │   └── schemas.py
│   ├── validation/
│   │   ├── quality.py
│   │   └── bias.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── models.py
│   ├── registry/
│   │   └── database.py
│   └── utils/
│       ├── config.py
│       ├── aws_client.py
│       └── logger.py
├── tests/
├── ui/
│   ├── app.py
│   ├── components/
│   │   ├── cards.py
│   │   ├── charts.py
│   │   └── styles.py
│   └── pages/
│       ├── generate.py
│       ├── validate.py
│       ├── training.py
│       ├── registry.py
│       └── compare.py
├── .cursorrules
├── .env
├── .env.template
├── pyproject.toml
├── README.md
└── uv.lock
```

---

## 3. Descripción detallada por módulo

### 3.1 `/data`

| Subdirectorio | Propósito |
|---------------|-----------|
| `data/raw/` | Datos originales o datasets base utilizados como referencia o comparación |
| `data/synthetic/` | Salida generada por el módulo de generación sintética. Incluye versiones, metadatos y logs |
| `data/reference/` | Datasets externos descargados o utilizados como ground truth (Banking77) |

Este directorio no contiene lógica; solo almacenamiento estructurado.

### 3.2 `/models`

Contiene modelos entrenados, checkpoints o artefactos generados por procesos internos de entrenamiento.

Puede incluir wrappers o modelos livianos generados con `trainer.py` (por ejemplo embeddings o clasificadores pequeños).

### 3.3 `/notebooks`

Notebooks exploratorios de análisis, experimentación y documentación técnica.

No forman parte del código de producción, pero complementan la investigación y pruebas.

### 3.4 `/src`

Carpeta principal de la lógica del proyecto.

#### a) `src/generation/`

Funcionalidad principal de generación sintética.

| Archivo | Responsabilidad |
|---------|-----------------|
| `generator.py` | Interacción con Bedrock, construcción de prompts, control de parámetros, retorno estandarizado |
| `schemas.py` | Schemas Pydantic para validación de datos generados |
| `templates/customer_service_prompts.py` | Prompts para conversaciones Banking77 (77 intents, bilingüe) |

#### b) `src/validation/`

Evaluación de calidad, consistencia y sesgos.

| Archivo | Responsabilidad |
|---------|-----------------|
| `quality.py` | Métricas objetivas: completitud, coherencia, diversidad, formato correcto |
| `bias.py` | Detección de sesgos lingüísticos o temáticos |

Estos módulos producen reportes estructurados que alimentan el registro.

#### c) `src/training/`

Módulos completos para entrenamiento de clasificadores de intents.

| Archivo | Responsabilidad |
|---------|-----------------|
| `intent_classifier.py` | TF-IDF + LogisticRegression/RandomForest/XGBoost para clasificación de intents |
| `trainer.py` | Orquestación de experimentos: `Trainer`, `ExperimentTracker`, `HyperparameterSearch` |
| `models.py` | Configuraciones: `ModelConfig`, `DataConfig`, `TrainingConfig`, `ExperimentConfig`, `PRESETS` |

**Funcionalidades:**
- Entrenamiento con múltiples algoritmos (LogReg, RandomForest, XGBoost)
- Cross-validation con k-folds configurable
- Grid search y random search para hiperparámetros
- Experiment tracking con métricas y artefactos
- Presets predefinidos: `fast`, `balanced`, `best`

#### d) `src/registry/`

Registro centralizado de datasets generados.

| Archivo | Responsabilidad |
|---------|-----------------|
| `database.py` | Registro de cada dataset con metadatos: fecha, parámetros, modelo, calidad, ruta |

Implementado en SQLite para el MVP.

#### e) `src/utils/`

Utilidades generales del sistema.

| Archivo | Responsabilidad |
|---------|-----------------|
| `config.py` | Variables de entorno, carga del `.env`, configuración global, rutas, constantes |
| `aws_client.py` | Cliente para AWS Bedrock: inicialización, invocación, manejo de errores |
| `logger.py` | Logger centralizado para errores, métricas, eventos y diagnósticos |

---

## 4. UI (Streamlit)

La UI está implementada como un sistema de componentes reutilizables con tema oscuro y diseño responsivo.

### Estructura

```
ui/
├── app.py                 # Punto de entrada + navegación
├── __init__.py
├── components/
│   ├── cards.py           # 12 componentes reutilizables
│   ├── charts.py          # 9 wrappers de Plotly
│   └── styles.py          # CSS glassmorphism + responsive
└── pages/
    ├── generate.py        # Generación de datos
    ├── validate.py        # Validación con métricas
    ├── training.py        # Entrenamiento de modelos
    ├── registry.py        # Registro de datasets
    └── compare.py         # Comparación de datasets
```

### `/ui/app.py`

Punto de entrada principal para la interfaz. Controla:
- Navegación con active page highlighting
- Sidebar con estadísticas en tiempo real
- Routing a páginas
- Inicialización de estado global

### `/ui/components/`

Sistema de componentes reutilizables:

| Componente | Archivo | Uso |
|------------|---------|-----|
| `page_header()` | cards.py | Header estandarizado para todas las páginas |
| `stat_card()` | cards.py | Métricas con valor grande |
| `metric_card()` | cards.py | Métricas con indicador de status |
| `domain_card()` | cards.py | Cards para dominios en home |
| `info_banner()` | cards.py | Banners de info/warning/error |
| `loading_spinner()` | cards.py | Spinner animado |
| `skeleton_card()` | cards.py | Placeholder shimmer |
| `error_state()` | cards.py | Estado de error con retry |
| `empty_state()` | cards.py | Placeholder para estados vacíos |
| `intent_distribution_chart()` | charts.py | Barras horizontales |
| `sentiment_pie_chart()` | charts.py | Donut chart |
| `quality_gauge()` | charts.py | Gauge para scores |
| `metrics_radar_chart()` | charts.py | Radar para métricas |

### `/ui/components/styles.py`

Sistema de estilos centralizado:
- **Tema oscuro** con glassmorphism (backdrop blur + transparencia)
- **CSS Variables** para paleta de colores
- **Gradientes** primarios (#667eea → #764ba2)
- **Animaciones** (fadeIn, pulse, gradient-shift)
- **Media queries** responsivos (1024px, 768px, 480px)

### `/ui/pages/`

Cada funcionalidad vive como una página independiente:

| Página | Funcionalidad |
|--------|---------------|
| `generate.py` | Configuración y generación de datos sintéticos |
| `validate.py` | Análisis de calidad, sesgos y distribuciones |
| `training.py` | Entrenamiento de clasificadores con presets, CV, y tracking |
| `registry.py` | Browse, search y export de datasets |
| `compare.py` | Comparación side-by-side de datasets |

### Flujo de UI

```
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (Router)                          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐           │
│  │  Home   │  │Generate │  │Validate │  │Registry │  │Compare│ │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └───┬───┘ │
└───────┼────────────┼────────────┼────────────┼───────────┼─────┘
        │            │            │            │           │
        ▼            ▼            ▼            ▼           ▼
   ┌──────────────────────────────────────────────────────────┐
   │                 components/ (Shared)                      │
   │  cards.py │ charts.py │ styles.py                        │
   └──────────────────────────────────────────────────────────┘
```

---

## 5. Pruebas

### `/tests/`

Contiene pruebas unitarias y de integración.

Cada módulo crítico debe tener pruebas asociadas:
- `test_config.py`
- `test_generator.py`
- `test_validation.py`
- `test_registry.py`
- `test_aws_client.py`

En fases posteriores se incluirán pruebas automáticas de CI/CD.

---

## 6. Componentes externos

| Componente | Propósito |
|------------|-----------|
| **AWS Bedrock** | Proveedor LLM principal (Claude 3.5 Sonnet) |
| **boto3** | SDK para comunicación con Bedrock, S3 y servicios auxiliares |
| **Streamlit** | Framework para la interfaz interactiva del MVP |
| **uv + pyproject.toml** | Manejo moderno de entornos y dependencias |
| **HuggingFace Datasets** | Datasets de referencia (Banking77) |

---

## 7. Principios arquitectónicos del proyecto

| Principio | Descripción |
|-----------|-------------|
| **Separación estricta de responsabilidades** | Cada módulo tiene una sola función claramente definida |
| **Extensibilidad** | Nuevos modelos o funciones deben integrarse sin alterar módulos existentes |
| **Ausencia de secretos en código** | Todo debe manejarse desde `.env` y `config.py` |
| **Modularidad y composición** | Los módulos deben poder conectarse entre sí sin dependencia circular |
| **Compatibilidad con IA asistida** | Código limpio, estructurado y predecible para facilitar generación automatizada |
| **Evolución incremental** | La arquitectura permite crecer hacia pipelines automatizados, agentes, UI avanzada, APIs externas |

---

## 8. Flujo general del sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                      1. UI (Streamlit)                          │
│         Usuario configura parámetros de generación              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   2. Generation Module                          │
│    generator.py + templates/ → Construye prompts                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   3. AWS Bedrock (Claude)                       │
│              Genera datos sintéticos                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   4. Validation Module                          │
│         quality.py + bias.py → Evalúa calidad                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   5. Registry Module                            │
│       database.py → Registra dataset + metadatos                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   6. UI (Streamlit)                             │
│              Muestra resultados al usuario                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Datasets de Referencia

### 9.1 Customer Service: Banking77

| Atributo | Valor |
|----------|-------|
| **Fuente** | `PolyAI/banking77` (HuggingFace) |
| **Dominio** | Neobank/Fintech (Revolut, Monzo style) |
| **Intents** | 77 categorías agrupadas en 11 categorías |
| **Idiomas soportados** | Inglés, Español |
| **Template** | `src/generation/templates/customer_service_prompts.py` |

**Categorías de intents:**
- card_management (18), card_payments (7), cash_atm (7)
- transfers (10), top_up (12), exchange_currency (5)
- account_security (4), verification_identity (4)
- account_management (5), payment_methods (2), refunds (3)

### 9.2 Time Series (ARCHIVADO)

> **Nota:** El dominio Time Series fue archivado debido a problemas técnicos.
> Ver [DOMAIN2_TIMESERIES.md](DOMAIN2_TIMESERIES.md) para documentación completa.

---

## 10. Contratos Entre Módulos (Data Contracts)

Esta sección define las interfaces y expectativas entre módulos del sistema. Cada contrato especifica el formato de entrada y salida que cada módulo espera y garantiza.

> **Referencias de implementación:**
> - Schemas de conversaciones: `src/generation/templates/customer_service_prompts.py`
> - Schemas Pydantic: `src/generation/schemas.py`

---

### 10.1 Generation → Validation

#### Input esperado por Validation

```python
{
    "domain": "customer_service",
    "data": [<schema_objects>],  # Lista de objetos validados
    "metadata": {
        "model_used": str,               # e.g., "claude-3-5-sonnet-20241022"
        "generation_date": datetime,     # ISO 8601 format
        "generation_params": {
            "temperature": float,
            "max_tokens": int,
            "top_p": float
        },
        "total_generated": int,
        "generation_time_seconds": float,
        "language": "en" | "es",
        "prompt_template_version": str
    }
}
```

#### Schema: Customer Service Conversation

```python
{
    "conversation_id": str,              # "conv_XXX"
    "intent": str,                       # Uno de 77 Banking77 intents
    "category": str,                     # Categoría del intent (11 categorías)
    "sentiment": "positive" | "neutral" | "negative",
    "complexity": "simple" | "medium" | "complex",
    "language": "en" | "es",
    "turn_count": int,
    "customer_emotion_arc": str,         # e.g., "frustrated_to_satisfied"
    "resolution_time_category": "quick" | "standard" | "extended",
    "resolution_status": "resolved" | "escalated" | "unresolved",
    "turns": [
        {"speaker": "customer" | "agent", "text": str}
    ]
}
```

#### Schema: Time Series (ARCHIVADO)

> Ver [DOMAIN2_TIMESERIES.md](DOMAIN2_TIMESERIES.md) para el schema archivado.

#### Output de Generation

**Caso exitoso:**
```python
{
    "success": True,
    "data": [<schema_objects>],
    "metadata": {...},
    "error": None
}
```

**Caso de error:**
```python
{
    "success": False,
    "data": [],
    "metadata": None,
    "error": "Descripción del error"
}
```

---

### 10.2 Validation → Registry

#### Input esperado por Registry

```python
{
    "dataset_id": str,                   # UUID único
    "domain": str,                       # "customer_service"
    "data": [],                          # Raw data objects
    "generation_metadata": dict,         # From generator
    "quality_metrics": QualityMetrics,   # From quality.py
    "bias_metrics": BiasMetrics,         # From bias.py
    "file_path": str,                    # Where data is saved
    "file_format": "json" | "jsonl" | "parquet",
    "file_size_mb": float
}
```

#### Output de Validation

```python
{
    "success": bool,
    "quality_passed": bool,              # True if quality_score >= threshold
    "bias_passed": bool,                 # True if no severe bias detected
    "quality_metrics": QualityMetrics,
    "bias_metrics": BiasMetrics,
    "issues": List[str],                 # Critical issues found
    "warnings": List[str],               # Non-critical warnings
    "recommendations": List[str],        # Suggestions for improvement
    "error": Optional[str]
}
```

---

### 10.3 Registry → Training

#### Input esperado por Training

```python
{
    "dataset_id": str,
    "domain": str,
    "data_path": str,                    # Path to load data from
    "task_type": "classification" | "regression" | "forecasting",
    "target_column": str,                # What to predict
    "feature_columns": List[str],        # What to use as features
    "training_config": dict              # From config.py
}
```

#### Output de Training

```python
{
    "success": bool,
    "model_name": str,                   # e.g., "xgboost_classifier"
    "model_path": str,                   # Where model is saved
    "metrics": {
        "accuracy": float,
        "f1_score": float,
        "precision": float,
        "recall": float,
        # ... other metrics depending on task
    },
    "training_time_seconds": float,
    "hyperparameters_used": dict,
    "error": Optional[str]
}
```

---

### 10.4 All Modules → UI (Streamlit)

#### Status Updates (for progress bars)

```python
{
    "stage": "generation" | "validation" | "training" | "registry",
    "progress": float,                   # 0.0 to 1.0
    "current_step": str,                 # Human-readable description
    "total_steps": int,
    "current_step_number": int,
    "eta_seconds": Optional[float]
}
```

#### Error Reporting

```python
{
    "error_type": "ValidationError" | "GenerationError" | "TrainingError",
    "error_message": str,                # User-friendly message
    "module": str,                       # Which module raised the error
    "timestamp": datetime,
    "traceback": Optional[str],          # Full traceback for debugging
    "suggestion": Optional[str]          # How to fix the error
}
```

---

### 10.5 AWS Bedrock Client → All Modules

#### Bedrock Invocation Input

```python
{
    "model_id": str,                     # From config.BEDROCK_MODEL_IDS
    "prompt": str,                       # User prompt
    "system_prompt": Optional[str],      # System instructions
    "temperature": float,                # 0.0 to 1.0
    "max_tokens": int,                   # Max response length
    "top_p": float                       # 0.0 to 1.0
}
```

#### Bedrock Invocation Output

```python
{
    "success": bool,
    "response_text": Optional[str],      # LLM response
    "error": Optional[str],
    "tokens_used": {
        "input": int,
        "output": int,
        "total": int
    },
    "latency_ms": float,
    "model_id": str
}
```

---

### 10.6 Error Handling Contract

Todos los módulos deben seguir este patrón:

#### Estructura de Retorno Estándar

```python
{
    "success": bool,
    "data": Any,                         # Result data if success=True
    "error": Optional[str],              # Error message if success=False
    "metadata": Optional[dict]           # Additional context
}
```

#### Reglas de Error Handling

| Regla | Descripción |
|-------|-------------|
| **Logging obligatorio** | Todos los errores deben loggearse usando el logger centralizado |
| **Errores recuperables** | Manejar internamente con retry logic (máx 3 intentos) |
| **Errores críticos** | Propagar hacia arriba con contexto claro |
| **Validación temprana** | Validar inputs antes de procesamiento costoso |
| **Mensajes útiles** | Incluir sugerencias de solución cuando sea posible |

---

### 10.7 Validation Metrics Contract

#### Quality Metrics (Todos los dominios)

```python
{
    "completeness_score": float,         # 0.0-1.0: % of required fields present
    "consistency_score": float,          # 0.0-1.0: Internal consistency
    "realism_score": float,              # 0.0-1.0: Comparison to reference data
    "diversity_score": float,            # 0.0-1.0: Variety in generated data
    "overall_quality_score": float       # 0-100: Weighted combination
}
```

#### Customer Service Specific Metrics

```python
{
    "turn_coherence_score": float,       # 0.0-1.0: Conversation flow quality
    "intent_distribution_score": float,  # 0.0-1.0: Coverage of 77 intents
    "intent_category_balance": float,    # 0.0-1.0: Balance across 11 categories
    "sentiment_balance_score": float,    # 0.0-1.0: Distribution of sentiments
    "complexity_distribution": float,    # 0.0-1.0: Mix of simple/medium/complex
    "language_quality_score": float,     # 0.0-1.0: Grammar and naturalness
    "resolution_rate": float,            # 0.0-1.0: % resolved conversations
    "emotion_arc_variety": float         # 0.0-1.0: Variety in emotion arcs
}
```

#### Time Series Specific Metrics (ARCHIVADO)

> Ver [DOMAIN2_TIMESERIES.md](DOMAIN2_TIMESERIES.md) para métricas archivadas.

---

### 10.8 Data Format Standards

#### File Naming Convention

```
{domain}_{dataset_id}_{timestamp}.{format}

Ejemplos:
- customer_service_a3f2e1d4_20240101_120000.jsonl
```

#### Format Selection

| Formato | Uso | Características |
|---------|-----|-----------------|
| **JSON** | Datasets pequeños (<1000 registros) | Legible, fácil debug |
| **JSONL** | Datasets grandes (≥1000 registros) | Un objeto por línea, streaming |
| **Parquet** | Datos tabulares grandes | Compresión snappy, eficiente |

#### Standards

- **Encoding**: UTF-8
- **Timestamps**: ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ)
- **Null values**: `null` en JSON, `None` en Python

---

### 10.9 Versioning Contract

#### Dataset Versioning

```python
{
    "version": str,                      # Semantic versioning: "1.0.0"
    "created_at": datetime,
    "parent_version": Optional[str],     # If derived from another dataset
    "changes": List[str],                # What changed from parent
    "backward_compatible": bool,
    "prompt_template_version": str,      # Version of prompts used
    "reference_dataset_version": str     # Version of Banking77/electricity used
}
```

#### Model Versioning

```python
{
    "model_version": str,                # "1.0.0"
    "dataset_version": str,              # Which dataset was used
    "trained_at": datetime,
    "framework": str,                    # "scikit-learn" | "xgboost"
    "framework_version": str
}
```

---

### 10.10 Testing Contract

Cada módulo debe tener:

| Tipo | Descripción |
|------|-------------|
| **Unit tests** | Funciones individuales |
| **Integration tests** | Contratos entre módulos |
| **End-to-end tests** | Pipeline completo |

#### Test Data Location

```
tests/
├── fixtures/
│   ├── customer_service_sample.json
│   └── reference_data/
│       └── banking77_sample.json
│       └── electricity_sample.json
└── test_*.py
```

#### Contract Validation Tests

```python
# Ejemplo de test de contrato
def test_generation_to_validation_contract():
    """Verify Generation output matches Validation input contract."""
    generation_output = generator.generate(...)
    
    # Debe tener estructura correcta
    assert "success" in generation_output
    assert "data" in generation_output
    assert "metadata" in generation_output
    
    # Data debe cumplir schema
    for item in generation_output["data"]:
        errors = validate_conversation_schema(item)
        assert len(errors) == 0, f"Schema errors: {errors}"
```

---

## 11. Estado actual de la arquitectura

| Componente | Estado |
|------------|--------|
| Estructura general | ✅ Creada |
| Módulos definidos | ✅ Definidos |
| Dependencias instaladas | ✅ Completado |
| AWS conectado | ✅ Configurado |
| Templates de prompts | ✅ Customer Service (77 intents) |
| Contratos documentados | ✅ Completado |
| Datasets de referencia | ✅ Banking77 |
| Schemas Pydantic | ✅ Completado |
| Módulo de generación | ✅ Completado |
| Módulo de validación | ✅ Completado (quality + bias) |
| Módulo de training | 🔄 Parcial (intent_classifier.py) |
| Registry | ✅ Completado (SQLite) |
| UI Sistema | ✅ Completado |
| UI Componentes | ✅ 12 componentes reutilizables |
| UI Charts | ✅ 9 charts Plotly |
| UI Páginas | ✅ 5 páginas funcionales |
| UI Responsivo | ✅ Media queries |
| Tests | 🔄 Parcial (falta registry, batch) |

**Leyenda:** ✅ Completado | 🔄 En progreso | ⏳ Pendiente

---

## 12. Changelog

| Fecha | Cambio |
|-------|--------|
| 2024-01-XX | Estructura inicial del proyecto |
| 2024-01-XX | Templates de prompts: customer_service_prompts.py (77 intents Banking77, bilingüe) |
| 2024-01-XX | Documentación de contratos entre módulos |
| 2026-01-16 | Refactor: archivado time series, enfoque en conversaciones |
| 2026-01-20 | UI completa: sistema de componentes, tema oscuro, diseño responsivo |

---

*Última actualización: 2026-01-20*