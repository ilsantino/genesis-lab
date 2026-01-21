 GENESIS-LAB — Technical Decision Record (TDR)

Este documento registra las decisiones técnicas críticas del proyecto GENESIS-LAB, su razón de ser y su impacto.

---
TDR-001 — Uso de uv como gestor de entornos
Razón: modernidad, velocidad, reproducibilidad y simplicidad.  
Impacto: elimina `requirements.txt` y centraliza dependencias en `pyproject.toml`.

---
TDR-002 — Arquitectura modular en src/
Razón: escalabilidad, claridad, mantenibilidad y compatibilidad con IA asistida.  
Impacto: el proyecto se organiza en: generation, validation, training, registry, utils.

---
TDR-003 — AWS Bedrock como proveedor LLM principal
Razón: estabilidad, calidad, soporte enterprise, costos predecibles.  
Impacto: dependencia en boto3 y módulo `aws_client.py`.

---
TDR-004 — Streamlit como UI inicial
Razón: rapidez para prototipar, interfaz accesible y fácil de integrar.  
Impacto: estructura en `ui/app.py` y `ui/pages/`.

---
TDR-005 — Manejo de secretos vía .env
Razón: seguridad y separación de configuración.  
Impacto: obligatoriedad de implementar `config.py` y no almacenar llaves en el código.

---
TDR-006 — Región AWS us-east-1
Razón: compatibilidad completa con Bedrock y modelos (incluido Claude cuando se habilite).  
Impacto: todas las llamadas a AWS dependen de esta región.

---
TDR-007 — Cross-Region Inference para Claude 3.5 Sonnet
Fecha: 2024-12-20  
Razón: Claude 3.5 Sonnet requiere cross-region inference profile en AWS Bedrock.  
Solución: prefijo `us.` en model ID (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`).  
Impacto: todos los model IDs de Claude 3.5+ deben incluir prefijo regional.

---
TDR-008 — Delay de 5 segundos para mitigación de throttling
Fecha: 2024-12-21  
Razón: AWS Bedrock throttlea requests frecuentes (>10 RPM estimado).  
Diagnóstico: script `diagnose_throttling.py` probó delays de 3s, 6s, 10s.  
Resultado: 5s delay + batch size 1 = 100% success rate.  
Impacto: `.env` incluye `BEDROCK_DELAY_SECONDS=5.0`.

---
TDR-009 — SQLite para Dataset Registry
Fecha: 2024-12-21  
Alternativas consideradas: PostgreSQL, DynamoDB, JSON files.  
Razón: simplicidad, zero-config, portable, suficiente para MVP.  
Impacto: archivo `data/registry.db` con tablas: datasets, quality_metrics, training_runs.  
Migración futura: fácil migrar a PostgreSQL si se requiere concurrencia.

---
TDR-010 — TF-IDF + LogisticRegression como baseline ML
Fecha: 2024-12-21  
Alternativas consideradas: XGBoost, sentence-transformers, fine-tuned BERT.  
Razón: simplicidad, interpretabilidad, velocidad de entrenamiento, sin GPU.  
Resultado: 15% accuracy con 100 samples (77 clases). Esperado.  
Impacto: modelo guardado en `models/trained/intent_classifier.pkl`.  
Siguiente paso: escalar datos o probar embeddings para mejorar accuracy.

---
TDR-011 — Arquitectura de componentes UI reutilizables
Fecha: 2026-01-20  
Alternativas consideradas: Inline HTML en cada página, uso directo de st.* widgets.  
Razón: reutilización, consistencia visual, mantenibilidad, separación de responsabilidades.  
Implementación: directorio `ui/components/` con tres módulos (cards.py, charts.py, styles.py).  
Impacto: 12 componentes reutilizables, todas las páginas usan los mismos componentes.  
Beneficio: cambios de estilo/comportamiento se hacen en un solo lugar.

---
TDR-012 — Tema oscuro con glassmorphism
Fecha: 2026-01-20  
Alternativas consideradas: Tema claro de Streamlit, tema oscuro sin efectos, Material Design.  
Razón: modernidad, legibilidad en monitores de desarrollo, coherencia con herramientas DevOps.  
Implementación: CSS personalizado en `styles.py` con:
- Backdrop blur (glassmorphism)
- CSS variables para paleta de colores
- Gradientes primarios (#667eea → #764ba2)
- Animaciones sutiles (fadeIn, pulse)
Impacto: inyección de CSS vía `inject_custom_css()` en punto de entrada.  
Trade-off: más código CSS que tema predeterminado, pero mejor experiencia visual.

---
TDR-013 — Diseño responsivo con media queries
Fecha: 2026-01-20  
Alternativas consideradas: Solo desktop, breakpoints de Bootstrap, CSS Grid.  
Razón: accesibilidad móvil, demos en tablets, mejor UX general.  
Implementación: media queries en `styles.py` para breakpoints:
- 1024px (tablets landscape)
- 768px (tablets portrait)
- 480px (mobile)
Impacto: tipografía, padding y elementos reducen tamaño proporcionalmente.  
Limitación: Streamlit no es mobile-first, algunas limitaciones de layout persisten.