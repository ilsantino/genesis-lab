Modular Synthetic Data Generation Factory for LLM Training Pipelines

GENESIS-LAB es un framework modular diseñado para crear, evaluar y administrar datasets sintéticos de alta calidad utilizando AWS Bedrock, pipelines con LLMs, y una arquitectura escalable basada en principios modernos de ingeniería de datos y agentes.

El proyecto funciona como una fábrica de datasets sintéticos, permitiendo:
- Generación controlada y reproducible
- Validación estadística y semántica
- Entrenamiento incremental
- Registro y versionamiento de datasets
- Exposición vía UI para uso interno y demo

⚙️ Arquitectura del Proyecto
El proyecto está organizado en una estructura modular:

genesis-lab/
│
├── src/
│   ├── generation/      # Generación sintética (prompts, pipelines, Bedrock)
│   ├── training/        # Rutinas para entrenamiento incremental
│   ├── validation/      # Métricas, calidad, distribución, sesgos
│   ├── registry/        # Registro y versionado de datasets
│   ├── utils/           # Config, cliente AWS, helpers
│   └── __init__.py
│
├── ui/
│   ├── pages/           # Pages de Streamlit
│   └── app.py           # UI principal
│
├── models/              # Modelos entrenados o checkpoints internos (vacío por .gitignore)
├── data/                # Datasets locales (excluidos del repo)
├── docs/                # Documentación técnica
│
├── .cursorrules         # Reglas del IDE asistido por IA
├── .env.template        # Variables requeridas (sin secretos)
├── pyproject.toml       # Dependencias del proyecto
└── README.md

🚀 Objetivos Principales
- Crear un sistema automatizado para la fabricación de datasets sintéticos.
- Diseñar pipelines reproducibles con AWS Bedrock y modelos LLM.
- Implementar métricas de calidad, distribución y sesgos para validación.
- Construir una UI interactiva para orquestar y visualizar procesos.
- Mantener una arquitectura expansible para nuevos módulos y agentes.

🔧 Instalación
1. Clonar el repositorio
git clone https://github.com/ilsantino/genesis-lab.git
cd genesis-lab

2. Crear entorno con uv
uv venv
uv sync

3. Crear archivo .env
Usa como referencia: cp .env.template .env

Rellena:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1

Nunca subas tu .env al repo.

▶️ Cómo ejecutar la UI: uv run streamlit run ui/app.py
La UI permite:
- Probar generadores de datos
- Visualizar métricas
- Exportar datasets
- Ejecutar validaciones

🧱 Módulos Principales
1. generation/
Pipelines para creación sintética:
- Templates de prompts
- Flujos con LLMs (Bedrock u otros modelos)
- Controladores de distribución y volumen

2. validation/
Métricas clave incluidas:
- Distribución estadística
- Sesgo semántico
- Calidad lingüística
- Divergencia vs dataset real

3. training/
Rutinas para:
- Entrenamiento incremental
- Evaluación
- Exportación de checkpoints (localmente, no en repo)

4. registry/
Gestión del dataset fabricado:
- Versioning
- Metadata
- Exportación a S3 o local

🧪 Tests
Ubicados en:tests/

Incluye pruebas unitarias para:
Funciones clave de generación
Validaciones estadísticas
Integración de pipelines

📄 Documentación
Toda la documentación del sistema está en:

docs/
├── ARCHITECTURE.md
├── DEVLOG.md
├── PROJECTSTATUS.md
├── ROADMAP.md
└── TDR.md

🧩 Roadmap (High Level)
 Integración Bedrock completa
 Agente autónomo para iterar datasets
 Sistema de scoring de calidad
 Entrenamiento RLHF
 Export direct to S3 + version control
 Métricas de fairness más avanzadas
 Dashboard comparativo de datasets

👤 Autor

Santiago Álvarez (Santino)
Founder & CEO, iaGO
AI-first innovation, automation & synthetic intelligence systems.