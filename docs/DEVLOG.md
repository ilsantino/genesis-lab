# GENESIS-LAB — Development Log (DEVLOG)

Este documento registra todos los avances diarios del proyecto GENESIS-LAB.  
Corresponde al Día 1 de trabajo consolidado.

---

## 📅 Día 1 — Setup inicial, arquitectura y configuración AWS

### 🔧 Configuración del entorno
- Creación de estructura base del proyecto: `src/`, `ui/`, `tests/`, `data/`, `models/`, `notebooks/`.
- Configuración de `pyproject.toml` como gestor principal de dependencias (usando uv).
- Creación de `.cursorrules` para estandarizar estilo, arquitectura y comportamiento de la IA asistente.
- Instalación de dependencias iniciales:
  - boto3  
  - streamlit  
  - python-dotenv  
  - pandas  
  - numpy
- Creación del entorno virtual usando `uv`.

### ☁️ AWS Setup completo
- Creación de IAM User: `genesis-lab-dev-ilsantino`.
- Creación de IAM Group con permisos mínimos:
  - AmazonBedrockFullAccess  
  - AmazonS3FullAccess
- Generación de Access Key y Secret Key.
- Instalación correcta de AWS CLI v2 (solucionando problemas de PATH).
- Configuración de credenciales vía `aws configure`.
- Ajuste de región a `us-east-1`.
- Validación satisfactoria de Bedrock con:
aws bedrock list-foundation-models --region us-east-1

markdown
Copy code

### 🔗 GitHub Setup
- Creación del repositorio remoto.
- Inicialización de Git local.
- Resolución del error “fetch first” al hacer push.
- Sincronización correcta del repo.

### 🖥 UI Base
- Creación de `ui/app.py` y estructura inicial de páginas en `ui/pages/`.
- Confirmación de que Streamlit no requiere extensión en Cursor.

### 🧱 Decisiones Arquitectónicas
- Uso de arquitectura modular en `src/`.
- Separación estricta por responsabilidad.
- Nada de hardcodear secretos (uso obligatorio de `.env + config.py`).
- Bedrock como proveedor principal.
- Streamlit como UI inicial.
- Llama/Nova como modelos temporales hasta aprobación de Claude.

---

## Estado actual del proyecto
- Entorno completamente funcional.
- AWS funcionando y Bedrock accesible.
- Proyecto inicializado con arquitectura limpia.
- Repo conectado a GitHub.
- UI básica creada.

---

## Próximas actividades
- Implementar `src/utils/config.py`.
- Implementar `src/utils/aws_client.py`.
- Crear logger central.
- Realizar primera invocación Bedrock desde Python.
- Iniciar módulo `generator.py`.