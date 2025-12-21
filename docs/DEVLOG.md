# GENESIS-LAB — Development Log

> Registro de desarrollo del proyecto de generación de datos sintéticos con AWS Bedrock.

---

## Índice

- [Día 0 — Setup Inicial](#día-0--setup-inicial)
- [Día 1 — Schemas, Templates y Reference Datasets](#día-1--schemas-templates-y-reference-datasets)
- [Día 2 — Motor de Generación + AWS Bedrock](#día-2--motor-de-generación--aws-bedrock)
- [Próximos Pasos — Día 3](#próximos-pasos--día-3)

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

## Próximos Pasos — Día 3

### Prioridad Alta

1. **Validation Module** (`src/validation/quality.py`)
   - Comparar sintéticos vs reference datasets
   - Métricas: completeness, consistency, realism, diversity
   - Usar schemas QualityMetrics y BiasMetrics

2. **Bias Detection** (`src/validation/bias.py`)
   - Sesgos en distribución de sentimientos
   - Cobertura de intents/series types
   - Alertas si bias > threshold

### Prioridad Media

3. **Dataset Completo**
   - Generar 100 conversaciones + 100 series
   - Ejecutar overnight para evitar throttling
   - Guardar en formato JSON Lines

4. **Prompt Caching**
   - Cachear prompts frecuentes
   - Reducir tokens y costos

### Prioridad Baja

5. **UI Streamlit**
   - Dashboard de visualización
   - Triggers de generación manual
   - Gráficas de métricas

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
