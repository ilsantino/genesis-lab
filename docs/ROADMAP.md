# GENESIS-LAB — Roadmap

Este documento describe las fases de desarrollo planificadas para GENESIS-LAB.

**Última actualización:** 2026-01-16

---

## Visión General

```
MVP (Días 0-3)     →     v1.0 (Días 4-7)     →     v2.0 (Futuro)
   ✅ Completado          🔄 En progreso           ⬜ Planificado
```

**Enfoque actual:** Customer Service Conversations (Banking77)

> **Nota:** El dominio Time Series fue archivado. Ver [DOMAIN2_TIMESERIES.md](DOMAIN2_TIMESERIES.md) para detalles.

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
| Reference dataset (Banking77) | ✅ | 1 |
| Prompt templates bilingües | ✅ | 1 |
| BedrockClient con retry/rate limiting | ✅ | 2 |
| CustomerServiceGenerator | ✅ | 2 |
| Smoke tests | ✅ | 2 |
| Unit tests | ✅ | 2 |
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
- Escalar generación a 1K+ conversaciones
- Mejorar accuracy del clasificador a 60-70%
- UI básica con Streamlit
- Pipeline de entrenamiento completo

### Checklist

| Feature | Estado | Prioridad |
|---------|--------|-----------|
| Generar 1K conversaciones | ⬜ | Alta |
| Mejorar calidad de generación | ⬜ | Alta |
| XGBoost classifier | ⬜ | Alta |
| Sentence embeddings | ⬜ | Alta |
| UI Streamlit: Dashboard | ⬜ | Media |
| UI Streamlit: Generación manual | ⬜ | Media |
| Prompt caching | ⬜ | Baja |
| Export HuggingFace Hub | ⬜ | Baja |

### Metas
- 1,000+ conversaciones generadas
- Classifier accuracy: 60-70%
- UI funcional para demos
- Pipeline de training reproducible

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
| Financial transactions domain | ⬜ | Nuevo dominio planificado |
| Multi-model support | ⬜ | Nova, Llama, etc. |

---

## Timeline Estimado

```
Enero 2026
├── Fase 1 completada ✅
├── Refactor: focus on conversations ✅
├── Scale to 1K conversations
├── Train classifier to 60-70%
└── v1.0 Release

Febrero 2026
├── Agent integration
├── RLHF experiments
└── v2.0 planning
```

---

## Dominios

| Dominio | Estado | Notas |
|---------|--------|-------|
| Customer Service (Banking77) | ✅ Activo | Enfoque principal |
| Time Series | ⚠️ Archivado | Ver DOMAIN2_TIMESERIES.md |
| Financial Transactions | ⬜ Futuro | v2.0+ |

---

## Referencias

- [DEVLOG.md](DEVLOG.md) - Progreso detallado día a día
- [PROJECTSTATUS.md](PROJECTSTATUS.md) - Estado actual del proyecto
- [ARCHITECTURE.md](ARCHITECTURE.md) - Arquitectura técnica
- [DOMAIN2_TIMESERIES.md](DOMAIN2_TIMESERIES.md) - Time Series (archivado)
