GENESIS-LAB — Architecture Overview

Este documento describe la arquitectura técnica de GENESIS-LAB, su organización interna, responsabilidades por módulo, principios de diseño y componentes principales. Su propósito es servir como referencia para el desarrollo, mantenimiento y escalamiento del proyecto.

1. Objetivo de la arquitectura
GENESIS-LAB está diseñado como un sistema modular para:
generación de datos sintéticos utilizando modelos de AWS Bedrock,
validación de calidad y sesgos,
registro y manejo de metadatos de datasets generados,
entrenamiento ligero de modelos cuando sea necesario,
interacción mediante una interfaz basada en Streamlit,
futura integración con agentes de IA y pipelines automatizados.

La arquitectura prioriza claridad, mantenibilidad, extensibilidad y separación estricta de responsabilidades.

2. Estructura general del proyecto
GENESIS-LAB/
  gitgun/
  .venv/
  data/
    raw/
    synthetic/
    reference/
  docs/
    ARCHITECTURE.md
    DEVLOG.md
    PROJECTSTATUS.md
    ROADMAP.md
    TDR.md
  models/
  notebooks/
  src/
    generation/
    validation/
    training/
    registry/
    utils/
  tests/
  ui/
    app.py
    pages/
  .cursorrules
  .env
  .env.template
  pyproject.toml
  README.md
  uv.lock

A continuación se detalla la función y responsabilidad de cada carpeta y módulo.

3. Descripción detallada por módulo
3.1 /data:
- data/raw/: Datos originales o datasets base utilizados como referencia o comparación.
- data/synthetic/: Salida generada por el módulo de generación sintética. Incluye versiones, metadatos y logs.
- data/reference/: Datasets externos descargados o utilizados como ground truth.
Este directorio no contiene lógica; solo almacenamiento estructurado.

3.2 /models
Contiene modelos entrenados, checkpoints o artefactos generados por procesos internos de entrenamiento.
Puede incluir wrappers o modelos livianos generados con trainer.py (por ejemplo embeddings o clasificadores pequeños).

3.3 /notebooks
Notebooks exploratorios de análisis, experimentación y documentación técnica.
No forman parte del código de producción, pero complementan la investigación y pruebas.

3.4 /src
Carpeta principal de la lógica del proyecto.

a) src/generation/
Funcionalidad principal de generación sintética.
- generator.py: Define clases y funciones que interactúan con Bedrock para generar datos en distintos formatos (texto, tablas, prompts estructurados). Responsable de: construcción de prompts, interacción con el cliente AWS, control de parámetros del modelo, retorno de resultados en formato estandarizado.
- templates/: Plantillas reutilizables para generación, definidas como JSON, YAML o Python dictionaries.

b) src/validation/
Evaluación de calidad, consistencia y sesgos.
- quality.py: Métricas objetivas como completitud, coherencia, diversidad, formato correcto y cumplimiento de reglas.
- bias.py: Detección de sesgos lingüísticos o temáticos utilizando heurísticas o modelos secundarios.

Estos módulos producen reportes estructurados que alimentan el registro.

c) src/training/
Módulos para entrenamiento ligero o ajuste interno.
- trainer.py: Permite entrenar modelos complementarios (clasificadores, pequeñas redes, filtros, embeddings).
- models.py: Define estructuras internas para representar modelos entrenados, cargar pesos o exportarlos.

Este módulo es opcional para el MVP inicial, pero la arquitectura ya lo considera para escalabilidad futura.

d) src/registry/
Registro centralizado de datasets generados.
- database.py: Registra cada dataset generado con sus metadatos: fecha, parámetros del prompt, modelo utilizado, calidad obtenida, validaciones, ruta del archivo generado.

Puede implementarse en SQLite, Parquet o JSON estructurado según necesidades.

e) src/utils/
Utilidades generales del sistema.
- config.py: Maneja variables de entorno, carga del .env, configuración global, rutas, y constantes del proyecto.
- aws_client.py: Cliente generalizado para interactuar con AWS Bedrock mediante boto3: inicialización, manejo de excepciones, wrapper para invocación de modelos, manejo de modelos alternativos (Llama, Nova, Claude cuando esté disponible).
- logger.py: Logger centralizado para registrar errores, métricas, eventos y diagnósticos del sistema.

Estos utilitarios son fundamentales y utilizados por todos los demás módulos.

4. UI (Streamlit)
/ui/app.py
Punto de entrada principal para la interfaz.
Controla: 
- navegación entre páginas
- inicialización de estado global
- carga de configuraciones.

/ui/pages/
Cada funcionalidad vive como una página independiente: 
- generate.py → página para generar datos sintéticos, 
- validate.py → muestra resultados de validación
- registry.py → consulta del historial de datasets

Streamlit se usa únicamente como interfaz de experimentación para el MVP.

5. Pruebas
/tests/
Contiene pruebas unitarias y de integración.
Cada módulo crítico debe tener pruebas asociadas: test_config.py, test_generator.py, test_validation.py, test_registry.py, test_aws_client.py
En fases posteriores se incluirán pruebas automáticas de CI/CD.

6. Componentes externos
AWS Bedrock: Proveedor LLM principal (Claude cuando esté habilitado; temporalmente Llama/Nova).
boto3: SDK para comunicación con Bedrock, S3 y servicios auxiliares.
Streamlit: Framework para la interfaz interactiva del MVP.
uv + pyproject.toml: Manejo moderno de entornos y dependencias.

7. Principios arquitectónicos del proyecto
Separación estricta de responsabilidades:Cada módulo tiene una sola función claramente definida.
Extensibilidad: Nuevos modelos o funciones deben integrarse sin alterar módulos existentes.
Ausencia de secretos en código: Todo debe manejarse desde .env y config.py.
Modularidad y composición: Los módulos deben poder conectarse entre sí sin dependencia circular.
Compatibilidad con IA asistida (Cursor, Claude): Código limpio, estructurado y predecible para facilitar generación automatizada.
Evolución incremental: 
La arquitectura permite crecer hacia: pipelines automatizados, agentes, UI avanzada, APIs externas.

8. Flujo general del sistema
El usuario interactúa con la UI (Streamlit).
La UI envía parámetros al módulo generation.
generator.py construye el prompt y llama a aws_client.py.
Bedrock devuelve la generación.
El resultado pasa por validaciones (validation/).
Se registra el dataset en registry/database.py.
La UI muestra los resultados.

9. Estado actual de la arquitectura
Estructura general creada.
Módulos definidos pero aún no implementados.
Dependencias instaladas.
AWS conectado.
UI inicial creada.
Preparado para comenzar implementación de backend.