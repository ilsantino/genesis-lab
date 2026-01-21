"""
Generate Page - Synthetic data generation interface.

Provides a form for configuring and triggering data generation.
"""

import streamlit as st
import json
import time
from pathlib import Path
from datetime import datetime

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, stat_card, metric_card, page_header
from ui.components.charts import intent_distribution_chart, sentiment_pie_chart

# Try to import backend
BACKEND_AVAILABLE = False
try:
    from src.pipelines.customer_service_pipeline import CustomerServicePipeline, PipelineConfig
    from src.generation.generator import CustomerServiceGenerator
    BACKEND_AVAILABLE = True
except ImportError:
    pass


def render_generate_page():
    """Render the generation page."""
    page_header(
        icon="⚡",
        title="Generate Synthetic Data",
        subtitle="Configure and generate customer service conversations."
    )
    
    # Check backend availability
    if not BACKEND_AVAILABLE:
        info_banner(
            "Demo Mode - Backend not connected. Generation is simulated.",
            type="warning",
            icon="🔧"
        )
    
    # Main layout
    col_config, col_preview = st.columns([1, 1])
    
    with col_config:
        st.markdown('<p class="section-header">Configuration</p>', unsafe_allow_html=True)
        
        # Configuration form in glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Number of samples
        num_samples = st.slider(
            "Number of Conversations",
            min_value=5,
            max_value=1000,
            value=100,
            step=5,
            help="Number of synthetic conversations to generate"
        )
        
        # Language selection
        language = st.selectbox(
            "Language",
            options=["en", "es"],
            format_func=lambda x: "English" if x == "en" else "Spanish",
            help="Target language for generated conversations"
        )
        
        # Quality threshold
        quality_threshold = st.slider(
            "Quality Threshold",
            min_value=50.0,
            max_value=100.0,
            value=70.0,
            step=5.0,
            help="Minimum quality score for production readiness"
        )
        
        # Bias threshold
        bias_threshold = st.selectbox(
            "Bias Threshold",
            options=["none", "low", "medium", "high"],
            index=2,
            help="Maximum acceptable bias severity"
        )
        
        # Advanced options
        with st.expander("Advanced Options"):
            use_nlp_sentiment = st.checkbox(
                "Use NLP Sentiment Analysis",
                value=False,
                help="Use TextBlob for more accurate sentiment detection"
            )
            
            batch_mode = st.checkbox(
                "Batch Mode (requires AWS setup)",
                value=False,
                disabled=True,
                help="Use batch inference for large datasets"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Cost/Time Estimator
        st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Estimation</p>', unsafe_allow_html=True)
        
        # Calculate estimates
        time_per_sample = 60  # seconds (due to throttling)
        estimated_time = num_samples * time_per_sample
        estimated_cost = num_samples * 0.003  # $0.003 per conversation (rough estimate)
        
        col_est1, col_est2 = st.columns(2)
        
        with col_est1:
            hours = estimated_time // 3600
            minutes = (estimated_time % 3600) // 60
            time_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            metric_card(
                title="Estimated Time",
                value=time_str,
                subtitle=f"~{time_per_sample}s per sample",
                status="warning" if estimated_time > 3600 else "neutral",
                icon="⏱️"
            )
        
        with col_est2:
            metric_card(
                title="Estimated Cost",
                value=f"${estimated_cost:.2f}",
                subtitle="AWS Bedrock usage",
                status="neutral",
                icon="💰"
            )
        
        if estimated_time > 3600:
            info_banner(
                f"Large dataset ({num_samples} samples) may take {hours}+ hours due to API rate limits. "
                "Consider using batch mode when available.",
                type="warning"
            )
        
        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        
        # Generate button
        generate_clicked = st.button(
            "🚀 Start Generation",
            use_container_width=True,
            type="primary"
        )
        
        if generate_clicked:
            run_generation(
                num_samples=num_samples,
                language=language,
                quality_threshold=quality_threshold,
                bias_threshold=bias_threshold,
                use_nlp_sentiment=use_nlp_sentiment
            )
    
    with col_preview:
        st.markdown('<p class="section-header">Preview</p>', unsafe_allow_html=True)
        
        # Show sample conversation structure in glass card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Sample Output Structure**")
        
        sample_conv = {
            "conversation_id": "conv_abc123",
            "intent": "card_arrival",
            "category": "card_management",
            "sentiment": "neutral",
            "complexity": "medium",
            "language": language,
            "turn_count": 4,
            "turns": [
                {"speaker": "customer", "text": "Hi, I'm waiting for my new card..."},
                {"speaker": "agent", "text": "I'd be happy to help check on that!"},
                {"speaker": "customer", "text": "How long does delivery take?"},
                {"speaker": "agent", "text": "Usually 5-7 business days..."},
            ]
        }
        
        st.json(sample_conv)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Intent distribution preview
        st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Intent Categories</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: #a0aec0; margin-bottom: 1rem;">Your data will cover these Banking77 categories:</p>', unsafe_allow_html=True)
        
        categories = {
            "Card Management": 12,
            "Account Services": 15,
            "Payments & Transfers": 18,
            "Technical Support": 8,
            "General Inquiries": 10,
            "Disputes & Fraud": 7,
            "Other": 7,
        }
        
        fig = intent_distribution_chart(
            categories,
            title="",
            top_n=7,
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Results section (shown after generation)
    if "last_generation_results" in st.session_state:
        show_generation_results(st.session_state.last_generation_results)


def run_generation(
    num_samples: int,
    language: str,
    quality_threshold: float,
    bias_threshold: str,
    use_nlp_sentiment: bool
):
    """Run the generation process with progress tracking."""
    
    st.markdown("---")
    st.markdown("### Generation Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results_container = st.container()
    
    if BACKEND_AVAILABLE:
        try:
            # Create pipeline config
            config = PipelineConfig(
                quality_threshold=quality_threshold,
                bias_threshold=bias_threshold,
                use_nlp_sentiment=use_nlp_sentiment
            )
            
            # Initialize pipeline
            status_text.text("Initializing pipeline...")
            progress_bar.progress(10)
            
            pipeline = CustomerServicePipeline(config=config)
            
            # Run generation
            status_text.text(f"Generating {num_samples} conversations...")
            progress_bar.progress(20)
            
            # Note: The actual pipeline runs synchronously
            # For a better UX, we'd need to run this in a background thread
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"data/synthetic/customer_service_{num_samples}_{timestamp}.json"
            
            results = pipeline.run(
                count=num_samples,
                save_path=save_path,
                language=language,
                save_combined_report=True
            )
            
            progress_bar.progress(100)
            status_text.text("Generation complete!")
            
            # Store results
            st.session_state.last_generation_results = {
                "success": True,
                "count": results.get("generation", {}).get("count", 0),
                "quality": results.get("quality", {}).get("overall", 0),
                "bias_severity": results.get("bias", {}).get("severity", "unknown"),
                "file_path": save_path,
                "production_ready": results.get("summary", {}).get("production_ready", False),
            }
            
            st.success(f"Generated {results.get('generation', {}).get('count', 0)} conversations!")
            
        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Generation failed")
            st.error(f"Error: {str(e)}")
            
            st.session_state.last_generation_results = {
                "success": False,
                "error": str(e)
            }
    else:
        # Demo mode - simulate generation
        for i in range(5):
            progress_bar.progress((i + 1) * 20)
            status_text.text(f"Simulating generation... Step {i + 1}/5")
            time.sleep(0.5)
        
        status_text.text("Demo generation complete!")
        
        # Simulated results
        st.session_state.last_generation_results = {
            "success": True,
            "count": num_samples,
            "quality": 75.5,
            "bias_severity": "low",
            "file_path": "data/synthetic/demo_output.json",
            "production_ready": True,
            "demo": True
        }
        
        info_banner(
            "This is a demo. Connect the backend to generate real data.",
            type="info"
        )


def show_generation_results(results: dict):
    """Display generation results."""
    
    get_divider()
    
    st.markdown("### Generation Results")
    
    if not results.get("success", False):
        st.error(f"Generation failed: {results.get('error', 'Unknown error')}")
        return
    
    # Results cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stat_card(
            value=str(results.get("count", 0)),
            label="Conversations",
            icon="💬"
        )
    
    with col2:
        quality = results.get("quality", 0)
        stat_card(
            value=f"{quality:.1f}%",
            label="Quality Score",
            icon="✨"
        )
    
    with col3:
        severity = results.get("bias_severity", "unknown")
        stat_card(
            value=severity.title(),
            label="Bias Severity",
            icon="⚖️"
        )
    
    with col4:
        ready = results.get("production_ready", False)
        stat_card(
            value="Yes" if ready else "No",
            label="Production Ready",
            icon="🚀" if ready else "⚠️"
        )
    
    # File location
    if results.get("file_path"):
        st.markdown(f"**Output file:** `{results['file_path']}`")
    
    # Actions
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        if st.button("📊 Validate Data", use_container_width=True):
            st.session_state.current_page = "Validate"
            st.session_state.validate_file = results.get("file_path")
            st.rerun()
    
    with col_a2:
        if st.button("📁 View in Registry", use_container_width=True):
            st.session_state.current_page = "Registry"
            st.rerun()
    
    with col_a3:
        if st.button("🔄 Generate More", use_container_width=True):
            if "last_generation_results" in st.session_state:
                del st.session_state.last_generation_results
            st.rerun()
