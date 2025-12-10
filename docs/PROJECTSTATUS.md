GENESIS-LAB — Project Status Document

Este documento resume el estado actual del proyecto GENESIS-LAB, incluyendo el trabajo realizado, decisiones tomadas, ajustes efectuados y lo que falta por desarrollar en las siguientes fases. Está diseñado para mantenerse actualizado conforme avance el proyecto.

1. Estado general del proyecto
GENESIS-LAB cuenta ya con la estructura base completa del repositorio, un entorno de desarrollo funcional y configuraciones iniciales en AWS y GitHub. El proyecto está preparado para comenzar la implementación del backend modular y las primeras interacciones con modelos de Bedrock. La arquitectura fue definida desde el inicio con un enfoque en modularidad, mantenibilidad y escalabilidad.

2. Avances realizados:
Entorno y dependencias
Creación de la estructura del proyecto con carpetas src/, ui/, data/, tests/, models/, notebooks/.
Configuración de pyproject.toml y uso de uv como gestor de dependencias.
Instalación de librerías base: boto3, streamlit, numpy, pandas, python-dotenv.
Archivo .cursorrules creado para guiar al asistente de IA en estándares de estilo y arquitectura.
Archivo .env creado para variables de entorno (sin versionar).

AWS:
Creación de IAM User exclusivo para el proyecto.
Creación de un IAM Group con políticas mínimas necesarias (AmazonBedrockFullAccess, AmazonS3FullAccess).
Generación y configuración de credenciales.
Solución de problemas con AWS CLI (instalación, PATH, configuración).
Configuración correcta de región en us-east-1.
Validación exitosa del acceso a Bedrock.

GitHub:
Creación de repositorio remoto y sincronización con el entorno local.
Corrección de conflictos de push del repositorio inicial.

UI y arquitectura interna:
Creación de la base de la aplicación con Streamlit (ui/app.py).
Creación de estructura inicial para páginas dentro de ui/pages/.
Definición de la arquitectura modular del backend en src/.

Decisiones técnicas consolidadas:
Uso de uv en lugar de venv o pip.
Separación estricta en módulos (generation, validation, training, registry, utils).
Uso obligatorio de .env para configuración sensible.
AWS Bedrock como proveedor de LLMs.
Streamlit como interfaz inicial del sistema.

3. Cambios realizados durante el proceso
Ajuste de región AWS a us-east-1 por compatibilidad con Bedrock.
Corrección de la configuración de AWS CLI y reinstalación por problemas en la detección del ejecutable.
Ajuste del repositorio para eliminar conflictos iniciales con el remoto.
Simplificación del ambiente de trabajo mediante uv para evitar instalaciones duplicadas y mantener coherencia.
Eliminación de la necesidad de un requirements.txt, migrando dependencias a pyproject.toml.
Redefinición del manejo de secretos para evitar cualquier riesgo de filtración.

4. Trabajo pendiente para la siguiente fase
Implementaciones técnicas inmediatas:
Crear src/utils/config.py para centralizar configuración del proyecto.
Implementar src/utils/aws_client.py como wrapper para llamadas a Bedrock.
Crear un módulo de logging (logger.py) para trazabilidad del sistema.
Implementar y probar la primera invocación a un modelo de Bedrock desde Python.
Definir las plantillas iniciales de generación para generator.py.

Diseño y funcionalidad del backend:
Implementar generator.py para generación sintética.
Implementar módulo de validación de datos (quality.py, bias.py).
Crear sistema de registro de datasets en registry/database.py.

Desarrollo de la interfaz:
Crear página funcional para generación dentro de Streamlit.
Crear interfaz para revisar métricas de calidad.
Crear vista para explorar datasets generados.

Consolidación del pipeline:
Integración progresiva de: generación → validación → registro.
Preparar estructura para futura integración de agentes (cuando se apruebe Claude).

5. Riesgos y consideraciones actuales
Falta de acceso aprobado a Claude 3.5 puede retrasar pruebas avanzadas; se utilizarán modelos alternos mientras tanto.
El proyecto depende de mantener coherencia en la arquitectura modular para evitar deuda técnica desde temprano.
El manejo de credenciales debe seguir estrictamente las prácticas definias.

6. Próximos pasos recomendados
Finalizar la configuración interna del módulo utils (configuración, AWS client, logging).
Ejecutar la primera prueba real de comunicación con Bedrock desde código.
Construir la primera versión mínima del generador de datos.
Establecer el sistema de registro local de datasets generados.
Iniciar la construcción de las páginas de Streamlit con funcionalidades básicas.