# GENESIS-LAB — Development Log

> Registro de desarrollo del proyecto de generación de datos sintéticos con AWS Bedrock.

---

## Índice

- [Día 0 — Setup Inicial](#día-0--setup-inicial)
- [Día 1 — Schemas, Templates y Reference Datasets](#día-1--schemas-templates-y-reference-datasets)
- [Día 2 — Motor de Generación + AWS Bedrock](#día-2--motor-de-generación--aws-bedrock)
- [Día 3 — Validation Pipeline + Training Baseline](#día-3--validation-pipeline--training-baseline)
- [Próximos Pasos — Día 4](#próximos-pasos--día-4)

---

## Día 0 — Setup Inicial

**Fecha:** 2024-12-15

### Resumen

Configuración inicial del proyecto: estructura de carpetas, dependencias, AWS credentials, y GitHub.

### Configuración del Entorno

- Estructura base: `src/`, `ui/`, `tests/`, `data/`, `models/`, `notebooks/`
- Gestor de dependencias: `uv` con `pyproject.toml`
- Archivo `.cursorrules` para estilo y arquitectura

**Dependencias instaladas:**
```
boto3, streamlit, python-dotenv, pandas, numpy, pydantic
```

### AWS Setup

| Componente | Configuración |
|------------|---------------|
| IAM User | `genesis-lab-dev-ilsantino` |
| Permisos | AmazonBedrockFullAccess, AmazonS3FullAccess |
| Región | `us-east-1` |
| CLI | AWS CLI v2 configurado |

**Validación exitosa:**
```bash
aws bedrock list-foundation-models --region us-east-1
```

### GitHub Setup

- Repositorio remoto creado
- Git local inicializado
- Sincronización correcta con `origin/main`

### Decisiones Arquitectónicas

1. **Arquitectura modular** en `src/` con separación por responsabilidad
2. **Secretos via `.env`** — nunca hardcodear
3. **Bedrock** como proveedor principal de LLM
4. **Streamlit** como UI inicial
5. **Claude 3.5 Sonnet** como modelo principal

---

## Día 1 — Schemas, Templates y Reference Datasets

**Fecha:** 2024-12-16

### Resumen

Implementación de schemas Pydantic, descarga de datasets de referencia desde HuggingFace, y creación de templates de prompts bilingües.

### Archivos Creados

| Archivo | Descripción |
|---------|-------------|
| `src/generation/schemas.py` | Modelos Pydantic para todos los dominios |
| `src/generation/templates/customer_service_prompts.py` | Templates bilingües (77 intents) |
| `src/generation/templates/timeseries_prompts.py` | Templates para series temporales (16 tipos) |
| `src/utils/download_references.py` | Script para descargar datasets de HuggingFace |
| `data/reference/customer_service_reference.json` | 500 ejemplos de Banking77 |
| `data/reference/timeseries_reference.json` | 100 series de electricity_hourly |

### Schemas Implementados

**Customer Service:**
- `CustomerServiceConversation`: Conversaciones multi-turn con 77 intents
- `ConversationTurn`: Turnos individuales (speaker + text)
- Validación automática: primera interacción siempre del cliente

**Time Series:**
- `TimeSeries`: Series temporales con 16 tipos en 4 dominios
- `TimeSeriesPoint`: Puntos individuales (timestamp + value)
- Validación automática: timestamps ordenados cronológicamente

**Métricas:**
- `QualityMetrics`: Scores de completeness, consistency, realism, diversity
- `BiasMetrics`: Distribución demográfica, sentimiento, cobertura de tópicos
- `DatasetMetadata`: Registro de datasets generados

### Reference Datasets

| Dataset | Fuente | Cantidad | Uso |
|---------|--------|----------|-----|
| Customer Service | `banking77` | 500 ejemplos | Validación de intents |
| Time Series | `electricity_hourly` | 100 series × 500 puntos | Validación estadística |

> **Nota:** Se cambió de `ETDataset/ett` (2 series correlacionadas) a `electricity_hourly` (370 series independientes) para validación estadística robusta.

### Templates de Prompts

**Customer Service (77 intents en 11 categorías):**
- System prompts bilingües (EN/ES)
- 10 few-shot examples (5 EN + 5 ES)
- Tono neobank/fintech digital
- Funciones: `validate_intent()`, `validate_conversation_schema()`, `build_batch_prompt()`

**Time Series (16 tipos en 4 dominios):**
- Dominios: electricity (50%), energy (20%), sensors (20%), financial (10%)
- 10 few-shot examples bilingües
- Formato compatible con HuggingFace
- Funciones: `validate_series_type()`, `validate_timeseries_schema()`, `validate_temporal_consistency()`

### Decisiones Estratégicas

1. **77 intents de Banking77** en lugar de 10 genéricos → alineación con reference dataset
2. **Bilingüismo completo** (EN/ES) → mercados objetivo México/España
3. **Cambio de dataset** ETT → electricity_hourly → validación estadística robusta
4. **Tono neobank** en lugar de corporativo → coherencia con banking77
5. **Schemas expandidos** (11 campos conversaciones, 17 campos time series)

---

## Día 2 — Motor de Generación + AWS Bedrock

**Fecha:** 2024-12-20

### Resumen

Implementación completa del motor de generación con AWS Bedrock. Generadores funcionales para customer service y time series, validados con smoke tests y 16 unit tests.

### Archivos Creados

| Archivo | Descripción | Líneas |
|---------|-------------|--------|
| `src/generation/generator.py` | BaseGenerator + CustomerServiceGenerator | ~500 |
| `src/generation/timeseries_generator.py` | TimeSeriesGenerator | ~570 |
| `src/generation/__init__.py` | Exports de módulo | ~45 |
| `scripts/smoke_test.py` | Test de humo con throttling protection | ~220 |
| `scripts/test_batch_generation.py` | Prueba batch conversaciones | ~40 |
| `scripts/test_timeseries_generation.py` | Prueba series temporales | ~106 |
| `tests/test_generators.py` | Unit tests con mocks | ~508 |

### Archivos Modificados

| Archivo | Cambio |
|---------|--------|
| `src/utils/config/loader.py` | Fix modelo Claude → prefijo `us.` |
| `pyproject.toml` | Fix typo línea 33 |

### CustomerServiceGenerator

Generador de conversaciones estilo Banking77 para neobanks/fintech.

**Características:**
- 77 intents organizados en 11 categorías
- Bilingüe (EN/ES)
- Few-shot prompting (2 ejemplos por defecto)
- Configuración: sentimiento, complejidad, emotion_arc
- Validación y corrección automática de schema

**Estructura de salida:**
```json
{
  "conversation_id": "conv_abc123",
  "intent": "card_arrival",
  "category": "cards",
  "sentiment": "neutral",
  "complexity": "simple",
  "language": "en",
  "turn_count": 4,
  "resolution_status": "resolved",
  "turns": [
    {"speaker": "customer", "text": "..."},
    {"speaker": "agent", "text": "..."}
  ]
}
```

### TimeSeriesGenerator

Generador de series temporales multi-dominio compatible con HuggingFace.

**Dominios y tipos (16 total):**

| Dominio | Peso | Tipos |
|---------|------|-------|
| electricity | 50% | residential_consumption, commercial_consumption, industrial_load, grid_demand |
| energy | 20% | solar_generation, wind_generation, gas_consumption, heating_demand |
| sensors | 20% | temperature, pressure, humidity, air_quality |
| financial | 10% | stock_price, crypto_price, exchange_rate, trading_volume |

**Estructura de salida:**
```json
{
  "series_id": "ts_abc123",
  "domain": "electricity",
  "series_type": "residential_consumption",
  "frequency": "1H",
  "length": 24,
  "target": [0.2, 0.1, -0.1, ...],
  "seasonality_types": ["daily"],
  "trend_type": "none"
}
```

### Resultados: Smoke Test

**Configuración:**
- Batch size: 2 items
- Delay entre batches: 3 segundos
- Total objetivo: 10 conversaciones + 10 series

**Resultados:**

| Dominio | Generados | Validados | Throttled |
|---------|-----------|-----------|-----------|
| Customer Service | 5/10 | 5/5 ✓ | 5 |
| Time Series | 5/10 | 5/5 ✓ | 5 |
| **Total** | **10/20** | **10/10** | **10** |

**Tiempo total:** 9.3 minutos

**Archivos generados:**
- `data/synthetic/customer_service_smoke_test.json`
- `data/synthetic/timeseries_smoke_test.json`

> El 50% de pérdida se debe a throttling de AWS Bedrock, no a errores de código.

### Resultados: Unit Tests

**Ejecución:** `uv run pytest tests/test_generators.py -v`

**Resultado:** 16/16 passed (5.32s)

| Clase | Tests | Estado |
|-------|-------|--------|
| TestCustomerServiceGenerator | 6 | ✅ |
| TestTimeSeriesGenerator | 6 | ✅ |
| TestJSONParsing | 2 | ✅ |
| TestErrorHandling | 2 | ✅ |

### Problemas Resueltos

#### 1. Cross-Region Inference (Claude 3.5 Sonnet)

```
ValidationException: Invocation of model ID anthropic.claude-3-5-sonnet-20241022-v2:0 
with on-demand throughput isn't supported.
```

**Solución:** Agregar prefijo `us.` al model ID:
```python
"claude_35_sonnet": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
```

#### 2. ThrottlingException

```
ThrottlingException: Too many requests, please wait before trying again.
```

**Solución:**
- Retry logic con backoff exponencial (2s, 4s, 8s)
- Delays entre batches (3s)
- Flag `continue_on_error=True`

#### 3. from_config() missing argument

```
TypeError: BaseGenerator.from_config() missing 1 required positional argument: 'domain'
```

**Solución:** Override de `from_config()` en cada subclase con domain hardcodeado.

#### 4. pyproject.toml corrupted

```
TOML parse error at line 33: string values must be quoted
```

**Solución:** Corregir typo `s]` → `]`.

#### 5. UnicodeEncodeError (Windows)

```
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705'
```

**Solución:** Reemplazar emojis por texto ASCII `[OK]`.

### Git Commits

| Commit | Descripción |
|--------|-------------|
| `a1eb8ef` | docs: Update DEVLOG with Day 2 progress |
| `e537a62` | Day 2: Add unit tests (16 tests, mocked AWS) |
| `d591a78` | Day 2: Bedrock client + generators + smoke test |

### Checklist Día 2

| Entregable | Estado |
|------------|--------|
| Cliente AWS Bedrock con rate limiting/retry | ✅ |
| Clase base BaseGenerator | ✅ |
| CustomerServiceGenerator funcional | ✅ |
| TimeSeriesGenerator funcional | ✅ |
| Smoke test script | ✅ |
| Unit tests (16 tests) | ✅ |
| Fix cross-region inference | ✅ |
| Caching de prompts | ⬜ Pendiente |
| Generación 100+100 items | ⬜ Parcial |

---

## Día 3 — Validation Pipeline + Training Baseline

**Fecha:** 2024-12-21

### Resumen

Pipeline completo de validación y primer modelo de clasificación. Generación de 100 conversaciones bilingües con validación en tiempo real, detección de sesgos, y baseline de clasificación de intents.

### Accomplishments

#### Módulos Implementados

| Módulo | Descripción | Líneas |
|--------|-------------|--------|
| `src/validation/quality.py` | QualityValidator con métricas JSD | ~500 |
| `src/validation/bias.py` | BiasDetector con análisis de distribuciones | ~500 |
| `src/registry/database.py` | DatasetRegistry con SQLite | ~400 |
| `src/training/intent_classifier.py` | TF-IDF + LogisticRegression | ~560 |

#### Dataset Generado

| Métrica | Valor |
|---------|-------|
| **Total conversaciones** | 100 |
| **Idiomas** | 50 EN + 50 ES |
| **Intents cubiertos** | 77/77 (100%) |
| **Success rate** | 100% |
| **Tiempo de generación** | 51.5 minutos |
| **Costo estimado** | ~$1.00 |

**Archivo:** `data/synthetic/customer_service_100.json`

### Quality Metrics

| Métrica | Score | Descripción |
|---------|-------|-------------|
| **Completeness** | 1.00 | Todos los campos requeridos presentes |
| **Consistency** | 1.00 | Turnos alternados, primer speaker = customer |
| **Realism** | 0.43 | Distribución uniforme vs Banking77 skewed (esperado) |
| **Diversity** | 0.83 | Vocabulario variado, mensajes únicos |
| **OVERALL** | **81.3/100** | Weighted average |

> **Nota sobre Realism:** El score de 0.43 es esperado porque generamos distribución uniforme (1-2 ejemplos por intent) mientras que Banking77 tiene distribución sesgada. Esto es óptimo para training de clasificadores.

### Bias Analysis

| Check | Resultado | Target | Status |
|-------|-----------|--------|--------|
| Sentiment | 30% pos / 50% neu / 20% neg | 30/50/20 | ✅ Perfect match |
| Language | 50% EN / 50% ES | 50/50 | ✅ Balanced |
| Complexity | 30% simple / 50% med / 20% complex | 30/50/20 | ✅ Perfect match |
| Intent Coverage | 77/77 | 100% | ✅ Complete |

**Bias Detected:** No (severity: none)

### Training Results: Intent Classifier

| Métrica | Valor |
|---------|-------|
| **Test Accuracy** | 15.0% |
| **F1 Score (macro)** | 10.3% |
| **Training samples** | 80 |
| **Test samples** | 20 |
| **Unique intents** | 77 |
| **TF-IDF features** | 1,297 |

**Model:** `models/trained/intent_classifier.pkl`

> **Nota sobre Accuracy:** 15% es esperado con 77 clases y solo 100 muestras (~1.3 por clase). Random baseline sería 1.3% (1/77), así que nuestro modelo es 11× mejor que random. Para 85%+ accuracy se necesitan 5,000+ muestras.

### Issues Encountered & Solutions

#### 1. AWS Throttling

**Problema:** 50% failure rate en smoke tests iniciales.

**Diagnóstico:** Script `diagnose_throttling.py` probó delays de 3s, 6s, 10s.

**Solución:**
- Delay óptimo: 5 segundos entre llamadas
- Batch size: 1 (secuencial)
- `.env` actualizado: `BEDROCK_DELAY_SECONDS=5.0`

**Resultado:** 100% success rate en generación de 100 items.

#### 2. Stratified Split Error

```
ValueError: The least populated classes in y have only 1 member
```

**Solución:** Deshabilitar stratification cuando min_count < 2 por clase.

#### 3. sklearn API Change

```
TypeError: LogisticRegression.__init__() got an unexpected keyword argument 'multi_class'
```

**Solución:** Remover parámetro `multi_class` (deprecated en sklearn 1.8).

#### 4. PowerShell Quote Escaping

**Problema:** Comandos inline con comillas fallaban en PowerShell.

**Solución:** Crear scripts Python dedicados en lugar de one-liners.

### Files Created

| Archivo | Descripción |
|---------|-------------|
| `src/validation/quality.py` | QualityValidator con Jensen-Shannon divergence |
| `src/validation/bias.py` | BiasDetector para sentiment, intent, language |
| `src/validation/__init__.py` | Exports del módulo |
| `src/registry/database.py` | DatasetRegistry con SQLite |
| `src/registry/__init__.py` | Exports del módulo |
| `src/training/intent_classifier.py` | TF-IDF + LogisticRegression baseline |
| `src/training/__init__.py` | Exports del módulo |
| `scripts/generate_100.py` | Generación con checkpointing |
| `scripts/diagnose_throttling.py` | Diagnóstico de rate limiting |
| `scripts/health_check.py` | Verificación de sistema |
| `scripts/validate_100.py` | Validación de dataset |
| `scripts/register_datasets.py` | Registro en SQLite |
| `scripts/register_training.py` | Registro de training runs |
| `data/synthetic/customer_service_100.json` | 100 conversaciones bilingües |
| `data/registry.db` | Base de datos SQLite |
| `models/trained/intent_classifier.pkl` | Modelo entrenado |

### Files Modified

| Archivo | Cambio |
|---------|--------|
| `pyproject.toml` | Añadido scikit-learn |
| `uv.lock` | Actualizado con scipy, joblib, threadpoolctl |
| `scripts/smoke_test.py` | Mejorado con bilingüismo y delays |

### Git Commit

```
[main 3126a10] Day 3: Validation pipeline + training baseline
 21 files changed, 8869 insertions(+), 107 deletions(-)
```

### Checklist Día 3

| Entregable | Estado |
|------------|--------|
| QualityValidator funcional | ✅ |
| BiasDetector funcional | ✅ |
| DatasetRegistry con SQLite | ✅ |
| Generación 100 conversaciones | ✅ |
| IntentClassifier baseline | ✅ |
| Registro en database | ✅ |
| Unit tests passing | ✅ |
| Documentación actualizada | ✅ |

---

## Próximos Pasos — Día 4

### Prioridad Alta

1. **Escalar a 1K Conversaciones**
   - Ejecutar `generate_100.py` modificado para 1000 items
   - Estimar tiempo: ~8-9 horas con delay 5s
   - Considerar overnight run
   - Expected accuracy improvement: ~60-70%

2. **Time Series Pipeline Completo**
   - Generar 100 series temporales
   - Implementar TimeSeriesValidator
   - Entrenar forecasting baseline

### Prioridad Media

3. **Mejorar Intent Classifier**
   - Probar XGBoost como alternativa
   - Añadir embeddings (sentence-transformers)
   - Cross-validation para mejor estimación

4. **UI Streamlit Básico**
   - Dashboard de visualización de métricas
   - Gráficas de distribución de intents
   - Trigger manual de generación

### Prioridad Baja

5. **Optimizaciones**
   - Prompt caching para reducir costos
   - Batch processing paralelo (si AWS permite)
   - Export a HuggingFace Hub

---

## Referencia Rápida

### Comandos Útiles

```bash
# Smoke test
uv run python -m scripts.smoke_test

# Unit tests
uv run pytest tests/test_generators.py -v

# Todos los tests (sin integration)
uv run pytest -m "not integration"

# Verificar modelo configurado
uv run python -c "from src.utils.config import get_config; print(get_config().aws.bedrock_model_ids)"
```

### Modelos Bedrock Disponibles

| Modelo | ID |
|--------|-----|
| Claude 3.5 Sonnet (default) | `us.anthropic.claude-3-5-sonnet-20241022-v2:0` |
| Claude 3 Sonnet | `anthropic.claude-3-sonnet-20240229-v1:0` |
| Claude 3 Haiku | `anthropic.claude-3-haiku-20240307-v1:0` |
| Amazon Nova Pro | `us.amazon.nova-pro-v1:0` |

### Límites de Throttling

| Configuración | Éxito Esperado |
|---------------|----------------|
| Sin delay | ~30% |
| Batch 2 + delay 3s | ~50% |
| Batch 1 + delay 5s | ~80% |
| Batch 1 + delay 10s | ~95% |

### Variables de Entorno

```bash
AWS_ACCESS_KEY_ID=xxx
AWS_SECRET_ACCESS_KEY=xxx
AWS_REGION=us-east-1
```
