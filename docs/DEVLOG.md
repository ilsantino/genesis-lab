# GENESIS-LAB — Development Log (DEVLOG)

Este documento registra todos los avances diarios del proyecto GENESIS-LAB.  
Corresponde al Día 1 de trabajo consolidado.

---

## 📅 Día 0 — Setup inicial, arquitectura y configuración AWS

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
- Validación satisfactoria de Bedrock con: aws bedrock list-foundation-models --region us-east-1

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
# GENESIS-LAB - Project Status

## Día 1 Completado - Schemas, Reference Datasets, y Templates

Fecha: 2024-12-16

### Resumen Ejecutivo

Hemos completado exitosamente la fase de fundación del proyecto, estableciendo todos los schemas de datos, descargando datasets de referencia de HuggingFace, y creando templates de prompts bilingües expandidos. Las mejoras estratégicas realizadas durante la implementación han elevado significativamente la calidad técnica del proyecto más allá del plan original.

---

### Schemas de Datos Implementados

Creamos el archivo src slash generation slash schemas punto py con modelos Pydantic que definen la estructura de datos para todos los dominios del proyecto. Para el dominio de customer service, implementamos la clase CustomerServiceConversation que valida conversaciones multi-turn con diez intents diferentes, tres niveles de sentiment, y validación automática de que la primera interacción siempre sea del cliente. Para el dominio de series temporales, implementamos la clase TimeSeries que valida series temporales con seis tipos diferentes de series, cuatro frecuencias posibles, y validación automática de que los timestamps estén ordenados cronológicamente.

Adicionalmente definimos schemas para métricas de calidad con QualityMetrics que incluye scores de completeness, consistency, realism y diversity, y para métricas de bias con BiasMetrics que incluye distribución demográfica, distribución de sentimiento, y cobertura de tópicos. Finalmente creamos DatasetMetadata para el registro de datasets con información de generación, métricas de calidad, métricas de bias, y resultados de entrenamiento.

El tercer dominio financiero está documentado arquitecturalmente con FinancialTransaction pero no está implementado en el MVP, siguiendo la decisión estratégica de profundizar en dos dominios en lugar de implementar superficialmente tres.

---

### Reference Datasets Descargados

Descargamos exitosamente dos datasets de referencia desde HuggingFace que usaremos para validación de calidad y detección de bias. Para customer service descargamos quinientos ejemplos del dataset banking77 que contiene consultas de servicios bancarios digitales clasificadas en setenta y siete intents diferentes. Este dataset se guardó en data slash reference slash customer_service_reference punto json.

Para series temporales inicialmente planeamos usar ETDataset slash ett, pero identificamos durante la implementación que este dataset solo contenía dos series correlacionadas del mismo transformador eléctrico, lo cual era insuficiente para validación estadística robusta. Tomamos la decisión estratégica de cambiar a LeoTungAnh slash electricity_hourly, que proporciona trescientas setenta series independientes de consumo eléctrico de hogares portugueses reales. Descargamos cien series de las trescientas setenta disponibles, con quinientos puntos temporales cada una, y las guardamos en data slash reference slash timeseries_reference punto json.

Este cambio de dataset fue crítico porque con solo dos series correlacionadas no podíamos calcular distribuciones estadísticas significativas ni validar diversidad entre múltiples series sintéticas generadas. Con cien series independientes, el sistema de validación podrá comparar robustamente si los datos sintéticos muestran la misma variabilidad y patrones que datos reales de múltiples entidades independientes.

---

### Templates de Prompts - Customer Service

Creamos src slash generation slash templates slash customer_service_prompts punto py con templates bilingües expandidos que superan significativamente el diseño original. En lugar de los diez intents genéricos planificados, integramos los setenta y siete intents de banking77 organizados en once categorías funcionales, lo cual permite que nuestros datos sintéticos sean directamente comparables con el dataset de referencia durante la validación.

Implementamos bilingüismo completo con system prompts en inglés y español, y diez few-shot examples de alta calidad, cinco en cada idioma. El tono fue ajustado de corporativo tradicional a estilo neobank o fintech digital, reflejando que banking77 proviene de contexto de banca digital moderna similar a Revolut o Nubank, no de banca tradicional. Esta coherencia tonal es crítica para que los datos sintéticos sean realistas cuando se comparen con el reference dataset.

Expandimos el schema de conversaciones de cinco campos a once campos, agregando category para agrupar los setenta y siete intents, complexity con tres niveles, customer_emotion_arc para tracking de evolución emocional durante la conversación, y resolution_time_category para clasificar la eficiencia de resolución en instant, quick, standard o extended.

Implementamos validación built-in con las funciones validate_intent y validate_conversation_schema, siguiendo arquitectura de producción donde validamos en el punto de generación para prevenir errores downstream. También agregamos build_batch_prompt para generar múltiples conversaciones en una sola llamada al LLM, optimizando costos de AWS Bedrock.

---

### Templates de Prompts - Time Series

Creamos src slash generation slash templates slash timeseries_prompts punto py con una estructura similar pero adaptada a datos numéricos temporales. Expandimos de seis tipos genéricos a cuatro dominios estructurados con dieciséis series types: electricity con cincuenta por ciento del peso incluyendo residential_consumption y grid_demand, energy con veinte por ciento incluyendo solar_generation y wind_generation, sensors con veinte por ciento incluyendo temperature y pressure, y financial con diez por ciento incluyendo stock_price y crypto_price.

Implementamos bilingüismo con system prompts y diez few-shot examples, cinco en inglés y cinco en español. Expandimos el schema de series temporales de siete campos a diecisiete campos, agregando seasonality_types para especificar múltiples tipos de estacionalidad simultáneos, trend_type para clasificar tendencias, anomaly_types para especificar tipos de anomalías presentes, y domain_context para información específica del dominio.

Cambiamos el formato target de lista de objetos con timestamp y value a lista simple de valores numéricos, haciéndolo compatible con el formato estándar de HuggingFace que usa nuestro reference dataset. Esto permite comparación directa durante validación sin necesidad de transformaciones intermedias.

Implementamos tres funciones de validación: validate_series_type para verificar que el tipo de serie sea válido, validate_timeseries_schema para verificar la estructura del output, y validate_temporal_consistency para verificar que los timestamps estén correctamente espaciados y los patrones temporales sean coherentes.

---

### Configuración Centralizada

Actualizamos src slash utils slash config punto py con configuración detallada por dominio. Para customer service especificamos el reference dataset como banking77, los diez intents originales más una nota de que los templates usan setenta y siete intents, y parámetros de generación con temperature cero punto siete, max_tokens mil, y top_p cero punto nueve.

Para time series especificamos el reference dataset como electricity, los seis series types con frecuencias de one minute, five minutes, one hour y one day, y parámetros de generación con temperature más baja de cero punto cinco para mayor consistencia numérica, max_tokens dos mil, y top_p cero punto ochenta y cinco.

Configuramos thresholds de validación con mínimos de noventa y cinco por ciento para completeness, noventa por ciento para consistency, ochenta y cinco por ciento para realism, ochenta por ciento para diversity, y ochenta y cinco puntos cero para overall quality score. Para bias detection establecimos máximo de cero punto tres para sentiment imbalance y mínimo de cero punto siete para topic coverage.

Definimos la configuración de training con test size de veinte por ciento, validation size de diez por ciento, random state cuarenta y dos, y modelos habilitados incluyendo logistic regression y xgboost, dejando random forest deshabilitado para el MVP.

---

### Decisiones Estratégicas Clave

Durante la implementación tomamos varias decisiones estratégicas que mejoraron significativamente el proyecto. La primera fue integrar completamente los setenta y siete intents de banking77 en lugar de usar diez intents genéricos, lo cual alinea la generación sintética con el dataset de referencia usado en evaluación, facilitando comparaciones directas y validación rigurosa.

La segunda decisión fue implementar bilingüismo completo en inglés y español, reconociendo que México y España son mercados objetivo de iaGO y que proyectos bilingües son raros y valiosos en portfolios académicos. Esto agrega complejidad técnica pero también demuestra capacidad de trabajar con múltiples idiomas.

La tercera decisión fue cambiar de ETT a electricity_hourly para series temporales, basada en el análisis crítico de que dos series correlacionadas eran insuficientes para validación estadística robusta. Este cambio asegura que podemos validar adecuadamente la diversidad y realismo de datos sintéticos generados.

La cuarta decisión fue ajustar el tono de corporativo a neobank o fintech digital, reconociendo que banking77 proviene de contexto de banca digital moderna y que mantener coherencia tonal es crítico para realismo de datos sintéticos.

La quinta decisión fue expandir significativamente los schemas de cinco a once campos en conversations y de siete a diecisiete campos en time series, habilitando análisis mucho más ricos durante validación, bias detection y training.

---

### Estructura de Archivos Actual

La estructura del proyecto quedó organizada de la siguiente manera. En la raíz tenemos data con subdirectorios raw para datos crudos, synthetic para datos generados, y reference para datasets de referencia que contiene customer_service_reference punto json y timeseries_reference punto json. También tenemos docs con este archivo PROJECT_STATUS punto md y próximamente DOMAIN3_FINANCIAL punto md, models para modelos entrenados, y logs para archivos de log.

En src tenemos generation con schemas punto py y templates que contiene customer_service_prompts punto py y timeseries_prompts punto py. También tenemos utils con config punto py y download_references punto py, validation que está vacío por ahora, training que está vacío por ahora, y ui que está vacío por ahora.

---

### Métricas del Día 1

Comparando con el plan original, superamos significativamente las expectativas. En customer service expandimos de diez a setenta y siete intents, de dos a diez few-shot examples, de cinco a once campos en el schema, y agregamos dos funciones de validación. En time series expandimos de seis a dieciséis series types, de uno a diez few-shot examples, de siete a diecisiete campos en el schema, y agregamos tres funciones de validación.

Implementamos bilingüismo completo no planeado originalmente, tomamos una decisión crítica de cambio de dataset basada en análisis técnico, y establecimos arquitectura de validación built-in desde el diseño.

---

### Próximos Pasos - Día 2

Para mañana implementaremos el motor de generación que usa AWS Bedrock. Crearemos el cliente de AWS con manejo de rate limiting y retry logic, implementaremos el generador de conversations usando los templates bilingües, implementaremos el generador de time series, y agregaremos caching de prompts para optimizar costos.

El objetivo del día dos es poder generar cien conversaciones de customer service y cien series temporales sintéticas exitosamente, validar que cumplen con los schemas de Pydantic, y guardar los datos generados en formato JSON lines en data slash synthetic.

---

### Notas Técnicas

Las versiones de dependencias instaladas son pydantic para validación de schemas, python-dotenv para cargar variables de entorno desde punto env, datasets y huggingface_hub para acceso a datasets de HuggingFace.

El archivo punto env debe contener AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, y AWS_REGION configurados con tus credenciales de AWS Bedrock. El archivo punto cursorrules define la arquitectura modular del proyecto y debe mantenerse actualizado si hacemos cambios arquitecturales.

Usamos uv como gestor de dependencias en lugar de pip, por lo que todos los comandos de instalación usan uv add en lugar de pip install. El archivo pyproject punto toml es gestionado automáticamente por uv.
