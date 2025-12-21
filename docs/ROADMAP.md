# GENESIS-LAB — Roadmap

Este documento describe las fases de desarrollo planificadas para GENESIS-LAB.

**Última actualización:** 2024-12-21

---

## Visión General

```
MVP (Días 0-3)     →     v1.0 (Días 4-7)     →     v2.0 (Futuro)
   ✅ Completado          🔄 En progreso           ⬜ Planificado
```

---

## Fase 1: MVP (Días 0-3) ✅

### Objetivos
- Estructura base del proyecto
- Generadores funcionales con AWS Bedrock
- Pipeline de validación básico
- Primer modelo de clasificación

### Checklist

| Feature | Estado | Día |
|---------|--------|-----|
| Estructura de proyecto | ✅ | 0 |
| Configuración AWS/Bedrock | ✅ | 0 |
| GitHub setup | ✅ | 0 |
| Schemas Pydantic | ✅ | 1 |
| Reference datasets (Banking77, electricity) | ✅ | 1 |
| Prompt templates bilingües | ✅ | 1 |
| BedrockClient con retry/rate limiting | ✅ | 2 |
| CustomerServiceGenerator | ✅ | 2 |
| TimeSeriesGenerator | ✅ | 2 |
| Smoke tests | ✅ | 2 |
| Unit tests (16 tests) | ✅ | 2 |
| QualityValidator | ✅ | 3 |
| BiasDetector | ✅ | 3 |
| DatasetRegistry (SQLite) | ✅ | 3 |
| IntentClassifier baseline | ✅ | 3 |
| 100 conversaciones generadas | ✅ | 3 |

### Métricas Alcanzadas
- 100 conversaciones bilingües (50 EN + 50 ES)
- 77/77 intents cubiertos (100%)
- Quality score: 81.3/100
- Classifier accuracy: 15% (baseline)

---

## Fase 2: v1.0 (Días 4-7) 🔄

### Objetivos
- Escalar generación a 1K+ items
- Mejorar accuracy del clasificador
- UI básica con Streamlit
- Pipeline de time series completo

### Checklist

| Feature | Estado | Prioridad |
|---------|--------|-----------|
| Generar 1K conversaciones | ⬜ | Alta |
| TimeSeriesValidator | ⬜ | Alta |
| Generar 100+ time series | ⬜ | Alta |
| UI Streamlit: Dashboard | ⬜ | Media |
| UI Streamlit: Generación manual | ⬜ | Media |
| XGBoost classifier | ⬜ | Media |
| Sentence embeddings | ⬜ | Media |
| Forecasting baseline | ⬜ | Media |
| Prompt caching | ⬜ | Baja |
| Export HuggingFace Hub | ⬜ | Baja |

### Metas
- 1,000+ conversaciones
- Classifier accuracy: 60-70%
- UI funcional para demos
- Time series pipeline completo

---

## Fase 3: v2.0 (Futuro) ⬜

### Objetivos
- Agentes autónomos para iteración de datasets
- RLHF pipeline
- Métricas de fairness avanzadas
- Dashboard comparativo

### Checklist

| Feature | Estado | Notas |
|---------|--------|-------|
| Agente autónomo de iteración | ⬜ | Requiere Claude API |
| RLHF training pipeline | ⬜ | Requiere datasets grandes |
| Fairness metrics avanzadas | ⬜ | Demographic parity, etc. |
| S3 export con versionamiento | ⬜ | Para producción |
| Dashboard comparativo | ⬜ | Comparar versiones de datasets |
| Financial transactions domain | ⬜ | Documentado, no implementado |
| Multi-model support | ⬜ | Nova, Llama, etc. |

---

## Timeline Estimado

```
Diciembre 2024
├── Día 0-1: Setup + Schemas ✅
├── Día 2: Generators + Tests ✅
├── Día 3: Validation + Training ✅
├── Día 4-5: Scale to 1K + UI
├── Día 6-7: Time Series + Polish

Enero 2025
├── v1.0 Release
├── Agent integration
└── RLHF experiments
```

---

## Referencias

- [DEVLOG.md](DEVLOG.md) - Progreso detallado día a día
- [PROJECTSTATUS.md](PROJECTSTATUS.md) - Estado actual del proyecto
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura técnica

