"""
GENESIS LAB - Synthetic Data Factory

Main Streamlit application entry point with navigation and home page.

Run with: uv run streamlit run ui/app.py
"""

import streamlit as st
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import components
from ui.components.styles import inject_custom_css, COLORS, get_divider
from ui.components.cards import domain_card, stat_card, feature_list, info_banner

# Page configuration
st.set_page_config(
    page_title="GENESIS LAB",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()

# Initialize session state
if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"

if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


def render_sidebar():
    """Render the navigation sidebar."""
    current_page = st.session_state.get("current_page", "Home")
    
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0 2rem 0;">
            <span style="font-size: 3rem;">🧬</span>
            <h1 style="font-size: 1.5rem; margin: 0.5rem 0 0 0; color: white;">GENESIS LAB</h1>
            <p style="font-size: 0.8rem; color: #718096; margin: 0;">Synthetic Data Factory</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        st.markdown('<p style="color: #718096; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Navigation</p>', unsafe_allow_html=True)
        
        pages = [
            ("🏠 Home", "Home"),
            ("⚡ Generate", "Generate"),
            ("📊 Validate", "Validate"),
            ("🎓 Training", "Training"),
            ("📁 Registry", "Registry"),
            ("🔄 Compare", "Compare"),
            ("📚 Help", "Help"),
        ]
        
        for label, page in pages:
            is_active = current_page == page
            
            # Render active page with different styling
            if is_active:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3));
                    border-left: 3px solid #667eea;
                    border-radius: 0 8px 8px 0;
                    padding: 0.75rem 1rem;
                    margin-bottom: 0.25rem;
                    color: white;
                    font-weight: 600;
                ">{label}</div>
                """, unsafe_allow_html=True)
            else:
                if st.button(label, key=f"nav_{page}", use_container_width=True):
                    st.session_state.current_page = page
                    st.rerun()
        
        st.markdown("---")
        
        # Quick stats
        st.markdown('<p style="color: #718096; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem;">Quick Stats</p>', unsafe_allow_html=True)
        
        # Try to get real stats
        try:
            from src.registry.database import DatasetRegistry
            registry = DatasetRegistry()
            datasets = registry.list_datasets()
            total_datasets = len(datasets)
            total_samples = sum(d.get("size", 0) for d in datasets)
        except Exception:
            total_datasets = 0
            total_samples = 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Datasets", total_datasets)
        with col2:
            st.metric("Samples", f"{total_samples:,}")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 0.75rem;">
            <p>v1.0.0</p>
            <p>Built with Streamlit</p>
        </div>
        """, unsafe_allow_html=True)


def render_home():
    """Render the home page."""
    # Hero Section
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0;">
        <h1 class="hero-title">GENESIS LAB</h1>
        <p class="hero-subtitle" style="margin: 0 auto;">
            Enterprise-grade synthetic data generation platform. 
            Create realistic, diverse, and bias-free datasets for training AI models.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Domain Cards
    st.markdown("## Data Domains")
    st.markdown('<p style="color: #a0aec0; margin-bottom: 2rem;">Choose a domain to generate synthetic data tailored to your use case.</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        domain_card(
            title="Customer Service",
            description="Multi-turn banking conversations using Banking77 taxonomy with 77 distinct intents. Bilingual support for English and Spanish.",
            features=[
                "77 Banking intents (Banking77)",
                "Multi-turn dialogues",
                "Bilingual EN/ES support",
                "Sentiment & emotion arcs",
            ],
            status="active",
            badge_text="Active",
            icon="💬",
        )
        if st.button("Start Generating", key="cta_cs", use_container_width=True):
            st.session_state.current_page = "Generate"
            st.rerun()
    
    with col2:
        domain_card(
            title="Time-Series",
            description="Synthetic temporal data for electricity consumption, IoT sensors, and business metrics. Includes validation for stationarity and anomalies.",
            features=[
                "370 reference series",
                "4 sub-domains",
                "Seasonal patterns",
                "Anomaly injection",
            ],
            status="coming_soon",
            badge_text="Coming Q1 2025",
            icon="📈",
        )
        st.button("Notify Me", key="cta_ts", disabled=True, use_container_width=True)
    
    with col3:
        domain_card(
            title="Financial Transactions",
            description="Realistic transaction patterns for fraud detection and banking system testing. PCI-DSS compliant synthetic data generation.",
            features=[
                "Realistic patterns",
                "Fraud scenarios",
                "PCI-DSS compliant",
                "Multi-currency",
            ],
            status="coming_soon",
            badge_text="Coming Q2 2025",
            icon="💳",
        )
        st.button("Notify Me", key="cta_fin", disabled=True, use_container_width=True)
    
    get_divider()
    
    # Platform Features
    st.markdown("## Platform Features")
    
    features = [
        {
            "icon": "🎯",
            "title": "Quality Validation",
            "description": "Automated checks for completeness, consistency, realism, and diversity."
        },
        {
            "icon": "⚖️",
            "title": "Bias Detection",
            "description": "Identify and mitigate bias in sentiment, topics, and demographics."
        },
        {
            "icon": "🤖",
            "title": "ML-Ready Output",
            "description": "Export in JSON, CSV, or Parquet formats ready for model training."
        },
        {
            "icon": "📊",
            "title": "Dataset Registry",
            "description": "Track all generated datasets with full metadata and lineage."
        },
        {
            "icon": "🔄",
            "title": "Real vs Synthetic",
            "description": "Compare synthetic data distribution against reference datasets."
        },
        {
            "icon": "🌐",
            "title": "Multilingual",
            "description": "Generate data in multiple languages with consistent quality."
        },
    ]
    
    feature_list(features, columns=3)
    
    get_divider()
    
    # Quick Actions
    st.markdown("## Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: rgba(30, 30, 50, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">⚡</span>
            <h4 style="color: white; margin: 0.5rem 0;">Generate</h4>
            <p style="color: #718096; font-size: 0.85rem;">Create new synthetic data</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Generate", key="qa_gen", use_container_width=True):
            st.session_state.current_page = "Generate"
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="background: rgba(30, 30, 50, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">📊</span>
            <h4 style="color: white; margin: 0.5rem 0;">Validate</h4>
            <p style="color: #718096; font-size: 0.85rem;">Check data quality</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Validate", key="qa_val", use_container_width=True):
            st.session_state.current_page = "Validate"
            st.rerun()
    
    with col3:
        st.markdown("""
        <div style="background: rgba(30, 30, 50, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">📁</span>
            <h4 style="color: white; margin: 0.5rem 0;">Registry</h4>
            <p style="color: #718096; font-size: 0.85rem;">Browse datasets</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Registry", key="qa_reg", use_container_width=True):
            st.session_state.current_page = "Registry"
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="background: rgba(30, 30, 50, 0.5); border-radius: 12px; padding: 1.5rem; text-align: center;">
            <span style="font-size: 2rem;">🔄</span>
            <h4 style="color: white; margin: 0.5rem 0;">Compare</h4>
            <p style="color: #718096; font-size: 0.85rem;">Analyze distributions</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Compare", key="qa_cmp", use_container_width=True):
            st.session_state.current_page = "Compare"
            st.rerun()


def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()
    
    # Route to current page
    page = st.session_state.current_page
    
    if page == "Home":
        render_home()
    elif page == "Generate":
        from ui.pages.generate import render_generate_page
        render_generate_page()
    elif page == "Validate":
        from ui.pages.validate import render_validate_page
        render_validate_page()
    elif page == "Training":
        from ui.pages.training import render_training_page
        render_training_page()
    elif page == "Registry":
        from ui.pages.registry import render_registry_page
        render_registry_page()
    elif page == "Compare":
        from ui.pages.compare import render_compare_page
        render_compare_page()
    elif page == "Help":
        from ui.pages.help import render_help_page
        render_help_page()
    else:
        render_home()


if __name__ == "__main__":
    main()
