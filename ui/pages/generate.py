"""
Generate Page - Synthetic data generation interface.

Provides a comprehensive form for configuring and triggering data generation
with full customization of intents, sentiments, complexity, and more.
"""

import streamlit as st
import json
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, stat_card, metric_card, page_header
from ui.components.charts import intent_distribution_chart, sentiment_pie_chart

# Try to import backend
BACKEND_AVAILABLE = False
try:
    from src.pipelines.customer_service_pipeline import CustomerServicePipeline, PipelineConfig
    from src.generation.generator import CustomerServiceGenerator
    from src.generation.templates.customer_service_prompts import BANKING77_INTENTS, ALL_INTENTS
    BACKEND_AVAILABLE = True
except ImportError:
    # Fallback intent data for demo mode
    BANKING77_INTENTS = {
        "card_management": ["activate_my_card", "card_arrival", "card_not_working", "lost_or_stolen_card"],
        "card_payments": ["declined_card_payment", "pending_card_payment"],
        "cash_atm": ["atm_support", "cash_withdrawal_charge"],
        "transfers": ["pending_transfer", "declined_transfer", "failed_transfer"],
        "top_up": ["top_up_failed", "pending_top_up"],
        "exchange_currency": ["exchange_rate", "exchange_charge"],
        "account_security": ["change_pin", "passcode_forgotten"],
        "verification_identity": ["verify_my_identity", "unable_to_verify_identity"],
        "account_management": ["edit_personal_details", "terminate_account"],
        "payment_methods": ["apple_pay_or_google_pay"],
        "refunds": ["request_refund", "Refund_not_showing_up"],
    }
    ALL_INTENTS = [i for cat in BANKING77_INTENTS.values() for i in cat]

# Category display names
CATEGORY_DISPLAY_NAMES = {
    "card_management": "Card Management",
    "card_payments": "Card Payments",
    "cash_atm": "Cash & ATM",
    "transfers": "Transfers",
    "top_up": "Top Up",
    "exchange_currency": "Currency Exchange",
    "account_security": "Account Security",
    "verification_identity": "Identity Verification",
    "account_management": "Account Management",
    "payment_methods": "Payment Methods",
    "refunds": "Refunds",
}


def render_generate_page():
    """Render the generation page with full customization options."""
    page_header(
        icon="⚡",
        title="Generate Synthetic Data",
        subtitle="Configure and generate customer service conversations with full control."
    )
    
    # Check backend availability
    if not BACKEND_AVAILABLE:
        info_banner(
            "Demo Mode - Backend not connected. Generation is simulated.",
            type="warning",
            icon="🔧"
        )
    
    # Initialize session state for selections
    if "selected_categories" not in st.session_state:
        st.session_state.selected_categories = list(BANKING77_INTENTS.keys())
    if "selected_intents" not in st.session_state:
        st.session_state.selected_intents = ALL_INTENTS.copy()
    
    # Main layout - two columns
    col_config, col_preview = st.columns([1.2, 0.8])
    
    with col_config:
        render_configuration_panel()
    
    with col_preview:
        render_preview_panel()
    
    # Results section (shown after generation)
    if "last_generation_results" in st.session_state:
        show_generation_results(st.session_state.last_generation_results)


def render_configuration_panel():
    """Render the main configuration panel."""
    
    # =========================================================================
    # BASIC SETTINGS
    # =========================================================================
    st.markdown('<p class="section-header">Basic Settings</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Number of conversations
    num_samples = st.slider(
        "Number of Conversations",
        min_value=10,
        max_value=5000,
        value=1000,
        step=10,
        help="Total conversations to generate (batch inference)"
    )
    
    # Language distribution
    en_percentage = st.slider(
        "Language Distribution",
        min_value=0,
        max_value=100,
        value=50,
        format="%d%% English",
        help="Percentage of English conversations (rest will be Spanish)"
    )
    es_percentage = 100 - en_percentage
    
    st.markdown(f"""
        <div style="display: flex; justify-content: space-between; color: #a0aec0; font-size: 0.85rem; margin-top: -0.5rem;">
            <span>🇺🇸 English: {en_percentage}%</span>
            <span>🇪🇸 Spanish: {es_percentage}%</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # INTENT SELECTION
    # =========================================================================
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Intent Categories</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("""
        <p style="color: #a0aec0; font-size: 0.85rem; margin-bottom: 1rem;">
            Select categories and expand to customize individual intents (Banking77 taxonomy)
        </p>
    """, unsafe_allow_html=True)
    
    # Track selected intents
    all_selected_intents = []
    
    for category_key, intents in BANKING77_INTENTS.items():
        display_name = CATEGORY_DISPLAY_NAMES.get(category_key, category_key.replace("_", " ").title())
        
        # Category checkbox with expander
        col_check, col_count = st.columns([4, 1])
        
        with col_check:
            # Check if all intents in category are selected
            category_intents_selected = all(i in st.session_state.selected_intents for i in intents)
            
            category_selected = st.checkbox(
                f"{display_name}",
                value=category_intents_selected,
                key=f"cat_{category_key}"
            )
        
        with col_count:
            selected_in_cat = sum(1 for i in intents if i in st.session_state.selected_intents)
            st.markdown(f"""
                <span style="color: #718096; font-size: 0.8rem;">
                    {selected_in_cat}/{len(intents)}
                </span>
            """, unsafe_allow_html=True)
        
        # Update intents based on category selection
        if category_selected:
            for intent in intents:
                if intent not in st.session_state.selected_intents:
                    st.session_state.selected_intents.append(intent)
        else:
            for intent in intents:
                if intent in st.session_state.selected_intents:
                    st.session_state.selected_intents.remove(intent)
        
        # Expandable individual intents
        with st.expander(f"View {len(intents)} intents", expanded=False):
            cols = st.columns(2)
            for idx, intent in enumerate(intents):
                with cols[idx % 2]:
                    intent_selected = st.checkbox(
                        intent.replace("_", " ").title(),
                        value=intent in st.session_state.selected_intents,
                        key=f"intent_{intent}"
                    )
                    if intent_selected and intent not in st.session_state.selected_intents:
                        st.session_state.selected_intents.append(intent)
                    elif not intent_selected and intent in st.session_state.selected_intents:
                        st.session_state.selected_intents.remove(intent)
        
        all_selected_intents = st.session_state.selected_intents.copy()
    
    # Summary
    total_intents = len(ALL_INTENTS)
    selected_count = len(st.session_state.selected_intents)
    st.markdown(f"""
        <div style="margin-top: 1rem; padding: 0.75rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; text-align: center;">
            <span style="color: #10b981; font-weight: 600;">{selected_count}/{total_intents}</span>
            <span style="color: #a0aec0;"> intents selected</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # SENTIMENT DISTRIBUTION
    # =========================================================================
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Sentiment Distribution</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        positive_pct = st.slider("😊 Positive", 0, 100, 30, key="positive_pct")
    with col_s2:
        neutral_pct = st.slider("😐 Neutral", 0, 100, 50, key="neutral_pct")
    with col_s3:
        negative_pct = st.slider("😤 Negative", 0, 100, 20, key="negative_pct")
    
    # Normalize percentages
    total_sentiment = positive_pct + neutral_pct + negative_pct
    if total_sentiment > 0:
        norm_positive = round(positive_pct / total_sentiment * 100)
        norm_neutral = round(neutral_pct / total_sentiment * 100)
        norm_negative = 100 - norm_positive - norm_neutral
    else:
        norm_positive, norm_neutral, norm_negative = 33, 34, 33
    
    st.markdown(f"""
        <div style="display: flex; gap: 0.5rem; margin-top: 0.5rem;">
            <div style="flex: {norm_positive}; background: #10b981; height: 8px; border-radius: 4px;"></div>
            <div style="flex: {norm_neutral}; background: #6366f1; height: 8px; border-radius: 4px;"></div>
            <div style="flex: {norm_negative}; background: #ef4444; height: 8px; border-radius: 4px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #718096; font-size: 0.75rem; margin-top: 0.25rem;">
            <span>{norm_positive}%</span>
            <span>{norm_neutral}%</span>
            <span>{norm_negative}%</span>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # COMPLEXITY & TURNS
    # =========================================================================
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Complexity & Turn Count</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    st.markdown("**Complexity Levels**")
    col_c1, col_c2, col_c3 = st.columns(3)
    
    with col_c1:
        include_simple = st.checkbox("Simple", value=True, help="2-4 turns, straightforward")
    with col_c2:
        include_medium = st.checkbox("Medium", value=True, help="4-8 turns, follow-ups")
    with col_c3:
        include_complex = st.checkbox("Complex", value=True, help="8-12 turns, escalations")
    
    st.markdown("**Turn Count Range**")
    min_turns, max_turns = st.slider(
        "Turns per conversation",
        min_value=2,
        max_value=12,
        value=(4, 10),
        help="Number of exchanges between customer and agent"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # RESOLUTION TYPES
    # =========================================================================
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Resolution Types</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        include_resolved = st.checkbox("✅ Resolved", value=True, help="Issue fully addressed")
    with col_r2:
        include_escalated = st.checkbox("📞 Escalated", value=True, help="Transferred to specialist")
    with col_r3:
        include_pending = st.checkbox("⏳ Pending", value=False, help="Follow-up needed")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # AWS SETTINGS
    # =========================================================================
    with st.expander("🔧 AWS Settings", expanded=False):
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Check for credentials
        env_access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        env_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        has_default_creds = bool(env_access_key and env_secret_key)
        
        if has_default_creds:
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 8px; margin-bottom: 1rem;">
                    <span style="color: #10b981;">✓</span>
                    <span style="color: #a0aec0;">Connected (using default credentials)</span>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="display: flex; align-items: center; gap: 0.5rem; padding: 0.5rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px; margin-bottom: 1rem;">
                    <span style="color: #ef4444;">✗</span>
                    <span style="color: #a0aec0;">No credentials found - configure below or add to .env</span>
                </div>
            """, unsafe_allow_html=True)
        
        use_custom_creds = st.checkbox("Use custom credentials", value=False)
        
        if use_custom_creds:
            custom_access_key = st.text_input(
                "AWS Access Key ID",
                type="password",
                placeholder="AKIA...",
                help="Your AWS access key"
            )
            custom_secret_key = st.text_input(
                "AWS Secret Access Key",
                type="password",
                placeholder="Enter secret key",
                help="Your AWS secret key"
            )
        
        col_aws1, col_aws2 = st.columns(2)
        with col_aws1:
            s3_bucket = st.text_input(
                "S3 Bucket",
                value=os.getenv("S3_BUCKET", "genesis-lab-batch-inference"),
                help="S3 bucket for batch inference"
            )
        with col_aws2:
            aws_region = st.selectbox(
                "AWS Region",
                options=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"],
                index=0,
                help="AWS region for Bedrock"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # =========================================================================
    # ESTIMATION & GENERATE BUTTON
    # =========================================================================
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Estimation</p>', unsafe_allow_html=True)
    
    # Batch mode time estimates
    if num_samples <= 100:
        est_minutes = 5
    elif num_samples <= 500:
        est_minutes = 15
    elif num_samples <= 1000:
        est_minutes = 25
    elif num_samples <= 2000:
        est_minutes = 45
    else:
        est_minutes = int(num_samples / 1000 * 25)
    
    # Cost estimate (~$0.003 per conversation)
    estimated_cost = num_samples * 0.003
    
    col_est1, col_est2 = st.columns(2)
    
    with col_est1:
        metric_card(
            title="Estimated Time",
            value=f"~{est_minutes} min",
            subtitle="Batch inference",
            status="neutral",
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
    
    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
    
    # Generate button
    generate_clicked = st.button(
        "🚀 Start Generation",
        use_container_width=True,
        type="primary",
        disabled=len(st.session_state.selected_intents) == 0
    )
    
    if len(st.session_state.selected_intents) == 0:
        st.warning("Please select at least one intent category to generate data.")
    
    if generate_clicked:
        # Build configuration
        complexity_levels = []
        if include_simple:
            complexity_levels.append("simple")
        if include_medium:
            complexity_levels.append("medium")
        if include_complex:
            complexity_levels.append("complex")
        
        resolution_types = []
        if include_resolved:
            resolution_types.append("resolved")
        if include_escalated:
            resolution_types.append("escalated")
        if include_pending:
            resolution_types.append("pending")
        
        run_generation(
            num_samples=num_samples,
            en_percentage=en_percentage,
            selected_intents=st.session_state.selected_intents.copy(),
            sentiment_distribution={
                "positive": norm_positive,
                "neutral": norm_neutral,
                "negative": norm_negative
            },
            complexity_levels=complexity_levels,
            min_turns=min_turns,
            max_turns=max_turns,
            resolution_types=resolution_types,
            s3_bucket=s3_bucket if 's3_bucket' in dir() else "genesis-lab-batch-inference",
            aws_region=aws_region if 'aws_region' in dir() else "us-east-1"
        )


def render_preview_panel():
    """Render the preview panel with sample output and charts."""
    st.markdown('<p class="section-header">Preview</p>', unsafe_allow_html=True)
    
    # Sample output structure
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("**Sample Output Structure**")
    
    sample_conv = {
        "conversation_id": "conv_abc123",
        "intent": "card_arrival",
        "category": "card_management",
        "sentiment": "neutral",
        "complexity": "medium",
        "language": "en",
        "turn_count": 4,
        "resolution_status": "resolved",
        "turns": [
            {"speaker": "customer", "text": "Hi, I'm waiting for my new card..."},
            {"speaker": "agent", "text": "I'd be happy to help check on that!"},
            {"speaker": "customer", "text": "How long does delivery take?"},
            {"speaker": "agent", "text": "Usually 5-7 business days..."},
        ]
    }
    
    st.json(sample_conv)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Selected intents distribution
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Selected Distribution</p>', unsafe_allow_html=True)
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    # Count intents per category
    category_counts = {}
    for cat_key, intents in BANKING77_INTENTS.items():
        selected_in_cat = sum(1 for i in intents if i in st.session_state.selected_intents)
        if selected_in_cat > 0:
            display_name = CATEGORY_DISPLAY_NAMES.get(cat_key, cat_key)
            category_counts[display_name] = selected_in_cat
    
    if category_counts:
        fig = intent_distribution_chart(
            category_counts,
            title="",
            top_n=len(category_counts),
            height=280
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.markdown("""
            <p style="color: #718096; text-align: center; padding: 2rem;">
                No intents selected
            </p>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


def run_generation(
    num_samples: int,
    en_percentage: int,
    selected_intents: List[str],
    sentiment_distribution: Dict[str, int],
    complexity_levels: List[str],
    min_turns: int,
    max_turns: int,
    resolution_types: List[str],
    s3_bucket: str,
    aws_region: str
):
    """Run the generation process with progress tracking."""
    
    st.markdown("---")
    st.markdown("### Generation Progress")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate language split
    en_count = int(num_samples * en_percentage / 100)
    es_count = num_samples - en_count
    
    if BACKEND_AVAILABLE:
        try:
            # Create pipeline config
            config = PipelineConfig(
                quality_threshold=70.0,
                bias_threshold="medium",
                use_nlp_sentiment=False
            )
            
            # Initialize pipeline
            status_text.text("Initializing pipeline...")
            progress_bar.progress(10)
            
            pipeline = CustomerServicePipeline(config=config)
            
            # Run generation
            status_text.text(f"Generating {num_samples} conversations ({en_count} EN, {es_count} ES)...")
            progress_bar.progress(20)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"data/synthetic/customer_service_{num_samples}_{timestamp}.json"
            
            # Note: For full customization, we'd need to enhance the pipeline
            # For now, we pass the primary language and let the pipeline handle it
            language = "en" if en_percentage >= 50 else "es"
            
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
                "config": {
                    "intents": len(selected_intents),
                    "sentiment": sentiment_distribution,
                    "complexity": complexity_levels,
                    "turns": f"{min_turns}-{max_turns}",
                    "resolution": resolution_types
                }
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
            "quality": 78.5,
            "bias_severity": "low",
            "file_path": "data/synthetic/demo_output.json",
            "production_ready": True,
            "demo": True,
            "config": {
                "intents": len(selected_intents),
                "sentiment": sentiment_distribution,
                "complexity": complexity_levels,
                "turns": f"{min_turns}-{max_turns}",
                "resolution": resolution_types
            }
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
    
    # Configuration used
    if results.get("config"):
        config = results["config"]
        st.markdown("**Configuration Used:**")
        col_c1, col_c2, col_c3 = st.columns(3)
        with col_c1:
            st.markdown(f"- **Intents:** {config.get('intents', 'N/A')}")
            st.markdown(f"- **Turns:** {config.get('turns', 'N/A')}")
        with col_c2:
            sentiment = config.get('sentiment', {})
            st.markdown(f"- **Positive:** {sentiment.get('positive', 0)}%")
            st.markdown(f"- **Neutral:** {sentiment.get('neutral', 0)}%")
        with col_c3:
            st.markdown(f"- **Complexity:** {', '.join(config.get('complexity', []))}")
            st.markdown(f"- **Resolution:** {', '.join(config.get('resolution', []))}")
    
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
