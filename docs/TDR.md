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