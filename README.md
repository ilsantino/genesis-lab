Modular Synthetic Data Generation Factory for LLM Training Pipelines

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://genesis-lab-dfxa5n7gkykdk9bc8uhfpn.streamlit.app)

**Live Demo:** https://genesis-lab-dfxa5n7gkykdk9bc8uhfpn.streamlit.app

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

## 🚀 Quick Start

```bash
# Generar 10 conversaciones de prueba
uv run python scripts/generate_data.py --total 10 --delay 3

# Generar 100 conversaciones bilingües (~1.5h)
uv run python scripts/generate_data.py --total 100

# Generación overnight (500 items con auto-pause)
uv run python scripts/generate_data.py --total 500 --max-failures 10

# Resumir generación interrumpida
uv run python scripts/generate_data.py --total 500 --resume

# Ver plan sin generar (dry run)
uv run python scripts/generate_data.py --total 100 --dry-run

# Validar calidad del dataset
uv run python scripts/validate_100.py

# Entrenar clasificador de intents
uv run python -m src.training.intent_classifier

# Health check del sistema
uv run python scripts/health_check.py
```

## UI Streamlit

Ejecutar: `uv run streamlit run ui/app.py`

### Páginas disponibles:
- **Home** - Dashboard con dominios y métricas
- **Generate** - Configuración y generación de datos sintéticos
- **Validate** - Análisis de calidad, sesgos y distribuciones
- **Training** - Entrenamiento de clasificadores con presets y CV
- **Registry** - Browse, search y export de datasets
- **Compare** - Comparación side-by-side de datasets
- **Help** - Documentación completa del sistema

### Características:
- Tema oscuro con glassmorphism
- 12 componentes UI reutilizables
- 9 charts interactivos (Plotly)
- Diseño responsivo
- Estados de carga y error

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

## Métricas Actuales (Día 4)

| Métrica | Valor |
|---------|-------|
| Conversaciones generadas | 100 |
| Intents cubiertos | 77/77 (100%) |
| Quality score | 81.3/100 |
| Idiomas | 50% EN / 50% ES |
| Costo por conversación | ~$0.01 |

## Roadmap (High Level)

- [x] Integración Bedrock completa
- [x] Sistema de scoring de calidad
- [x] Validación de sesgos
- [x] Dataset Registry (SQLite)
- [x] Intent Classifier baseline
- [x] **UI Streamlit completa** (Dashboard, Generate, Validate, Registry, Compare)
- [x] **Dashboard comparativo de datasets**
- [x] **Sistema de componentes UI reutilizables**
- [x] **Diseño responsivo con tema oscuro**
- [ ] Escalar a 1K+ conversaciones (esperando AWS quota)
- [ ] Agente autónomo para iterar datasets
- [ ] Entrenamiento RLHF
- [ ] Export direct to S3 + version control
- [ ] Métricas de fairness más avanzadas

👤 Autor

Santiago Álvarez (Santino)
Founder & CEO, iaGO
AI-first innovation, automation & synthetic intelligence systems.