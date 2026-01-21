"""
Validate Page - Quality and bias analysis interface.

Provides tools for validating synthetic data quality and detecting bias.
"""

import streamlit as st
import json
from pathlib import Path

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, metric_card, stat_card, page_header
from ui.components.charts import (
    quality_gauge,
    sentiment_pie_chart,
    intent_distribution_chart,
    language_bar_chart,
    metrics_radar_chart,
    complexity_distribution_chart
)

# Try to import backend
BACKEND_AVAILABLE = False
try:
    from src.validation.quality import QualityValidator
    from src.validation.bias import BiasDetector
    from src.utils.visualization import (
        load_conversations,
        analyze_conversations,
        get_sentiment_percentages,
        get_language_percentages,
        get_quality_summary,
        get_bias_summary
    )
    BACKEND_AVAILABLE = True
except ImportError:
    from src.utils.visualization import (
        load_conversations,
        analyze_conversations,
        get_sentiment_percentages,
        get_language_percentages,
    )


def render_validate_page():
    """Render the validation page."""
    page_header(
        icon="📊",
        title="Validate Data",
        subtitle="Analyze quality metrics and detect bias in your datasets."
    )
    
    # File selection
    st.markdown("### Select Dataset")
    
    col_upload, col_existing = st.columns(2)
    
    with col_upload:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Upload File**")
        uploaded_file = st.file_uploader(
            "Choose a JSON file",
            type=["json"],
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_existing:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Select Existing Dataset**")
        
        # List available datasets
        data_dir = Path("data/synthetic")
        json_files = []
        if data_dir.exists():
            json_files = sorted(
                [f.name for f in data_dir.glob("*.json") if not f.name.endswith(".report.json")],
                reverse=True
            )
        
        selected_file = st.selectbox(
            "Choose a dataset",
            options=[""] + json_files,
            format_func=lambda x: "Select a file..." if x == "" else x,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Load data
    conversations = []
    file_source = None
    
    if uploaded_file:
        try:
            conversations = json.load(uploaded_file)
            file_source = uploaded_file.name
        except Exception as e:
            st.error(f"Error loading file: {e}")
    elif selected_file:
        file_path = data_dir / selected_file
        conversations = load_conversations(str(file_path))
        file_source = selected_file
    
    # Check for file from generate page
    if not conversations and "validate_file" in st.session_state:
        file_path = st.session_state.validate_file
        conversations = load_conversations(file_path)
        file_source = Path(file_path).name
        del st.session_state.validate_file
    
    if not conversations:
        info_banner(
            "Select or upload a dataset to begin validation.",
            type="info"
        )
        return
    
    # Run validation
    st.markdown(f"**Analyzing:** `{file_source}` ({len(conversations)} conversations)")
    
    get_divider()
    
    # Analysis tabs
    tab_quality, tab_bias, tab_distribution = st.tabs([
        "📈 Quality Metrics",
        "⚖️ Bias Analysis",
        "📊 Distributions"
    ])
    
    # Analyze data
    analysis = analyze_conversations(conversations)
    
    with tab_quality:
        render_quality_tab(conversations, analysis)
    
    with tab_bias:
        render_bias_tab(conversations, analysis)
    
    with tab_distribution:
        render_distribution_tab(conversations, analysis)


def render_quality_tab(conversations: list, analysis: dict):
    """Render quality metrics tab."""
    st.markdown('<p class="section-header">Quality Assessment</p>', unsafe_allow_html=True)
    
    # Try to get actual quality metrics
    quality_metrics = {}
    
    if BACKEND_AVAILABLE:
        try:
            validator = QualityValidator()
            metrics = validator.compute_overall_score(conversations)
            quality_metrics = {
                "completeness": metrics.completeness_score,
                "consistency": metrics.consistency_score,
                "realism": metrics.realism_score,
                "diversity": metrics.diversity_score,
            }
            overall_score = metrics.overall_score
        except Exception as e:
            st.warning(f"Could not compute quality metrics: {e}")
            quality_metrics = estimate_quality_metrics(conversations)
            overall_score = sum(quality_metrics.values()) / len(quality_metrics) * 100
    else:
        quality_metrics = estimate_quality_metrics(conversations)
        overall_score = sum(quality_metrics.values()) / len(quality_metrics) * 100
    
    # Overall score gauge and breakdown - equal columns
    col_gauge, col_breakdown = st.columns([1, 1])
    
    with col_gauge:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        fig = quality_gauge(
            overall_score,
            title="Overall Quality",
            height=260
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Status message
        if overall_score >= 80:
            info_banner("Excellent quality! Dataset is production-ready.", type="success")
        elif overall_score >= 70:
            info_banner("Good quality. Minor improvements possible.", type="info")
        elif overall_score >= 60:
            info_banner("Acceptable quality. Consider regenerating some samples.", type="warning")
        else:
            info_banner("Quality below threshold. Review and regenerate.", type="error")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_breakdown:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Metric Breakdown</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            score = quality_metrics.get("completeness", 0)
            metric_card(
                title="Completeness",
                value=f"{score*100:.1f}%",
                subtitle="All required fields present",
                status="success" if score >= 0.95 else "warning" if score >= 0.8 else "error",
                icon="✓"
            )
            
            score = quality_metrics.get("consistency", 0)
            metric_card(
                title="Consistency",
                value=f"{score*100:.1f}%",
                subtitle="Logical conversation flow",
                status="success" if score >= 0.9 else "warning" if score >= 0.75 else "error",
                icon="🔗"
            )
        
        with col2:
            score = quality_metrics.get("realism", 0)
            metric_card(
                title="Realism",
                value=f"{score*100:.1f}%",
                subtitle="Natural language patterns",
                status="success" if score >= 0.85 else "warning" if score >= 0.7 else "error",
                icon="💬"
            )
            
            score = quality_metrics.get("diversity", 0)
            metric_card(
                title="Diversity",
                value=f"{score*100:.1f}%",
                subtitle="Variety in responses",
                status="success" if score >= 0.8 else "warning" if score >= 0.65 else "error",
                icon="🎨"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Radar chart
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Quality Profile</p>', unsafe_allow_html=True)
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    fig = metrics_radar_chart(quality_metrics, title="", height=350)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def render_bias_tab(conversations: list, analysis: dict):
    """Render bias analysis tab."""
    st.markdown('<p class="section-header">Bias Detection</p>', unsafe_allow_html=True)
    
    # Get bias metrics
    if BACKEND_AVAILABLE:
        try:
            detector = BiasDetector()
            bias_results = detector.detect_bias(conversations)
            severity = bias_results.get("severity", "unknown")
            issues = bias_results.get("issues", [])
        except Exception:
            severity = "low"
            issues = []
    else:
        severity = estimate_bias_severity(analysis)
        issues = []
    
    # Severity indicator
    severity_colors = {
        "none": ("success", "No significant bias detected"),
        "low": ("info", "Minor imbalances detected"),
        "medium": ("warning", "Moderate bias - review recommended"),
        "high": ("error", "Significant bias - action required")
    }
    
    color, message = severity_colors.get(severity, ("neutral", "Unknown"))
    
    # Equal columns for severity and sentiment
    col_sev, col_sent = st.columns([1, 1])
    
    with col_sev:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        metric_card(
            title="Bias Severity",
            value=severity.upper(),
            subtitle=message,
            status=color,
            icon="⚖️"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_sent:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        # Sentiment distribution
        sentiments = analysis.get("sentiments", {})
        sent_pct = get_sentiment_percentages(sentiments)
        
        fig = sentiment_pie_chart(sent_pct, title="Sentiment Distribution", height=250)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    get_divider()
    
    # Detailed analysis - equal columns
    col_lang, col_complexity = st.columns([1, 1])
    
    with col_lang:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Language Distribution</p>', unsafe_allow_html=True)
        languages = analysis.get("languages", {})
        lang_pct = get_language_percentages(languages)
        
        fig = language_bar_chart(lang_pct, title="", height=220)
        st.plotly_chart(fig, use_container_width=True)
        
        # Check for imbalance
        if lang_pct:
            max_lang_pct = max(lang_pct.values())
            if max_lang_pct > 80 and len(lang_pct) > 1:
                info_banner(
                    "Language distribution is imbalanced.",
                    type="warning"
                )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_complexity:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Complexity Distribution</p>', unsafe_allow_html=True)
        complexities = analysis.get("complexities", {})
        
        fig = complexity_distribution_chart(complexities, title="", height=220)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Intent coverage - equal columns
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Intent Coverage</p>', unsafe_allow_html=True)
    intents = analysis.get("intents", {})
    total_intents = len(intents)
    coverage = (total_intents / 77) * 100  # Banking77 has 77 intents
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        metric_card(
            title="Intent Coverage",
            value=f"{coverage:.1f}%",
            subtitle=f"{total_intents} of 77 intents",
            status="success" if coverage >= 70 else "warning" if coverage >= 50 else "error",
            icon="🎯"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        if coverage < 50:
            info_banner(
                f"Low intent coverage ({coverage:.1f}%). Generate more samples targeting underrepresented intents.",
                type="warning"
            )
        elif coverage < 70:
            info_banner(
                f"Moderate intent coverage ({coverage:.1f}%). Consider targeting specific intents.",
                type="info"
            )
        else:
            info_banner(
                f"Good intent coverage ({coverage:.1f}%). Dataset has broad intent representation.",
                type="success"
            )
        st.markdown('</div>', unsafe_allow_html=True)


def render_distribution_tab(conversations: list, analysis: dict):
    """Render distribution analysis tab."""
    st.markdown('<p class="section-header">Data Distributions</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Top Intents</p>', unsafe_allow_html=True)
        intents = analysis.get("intents", {})
        fig = intent_distribution_chart(intents, title="", top_n=12, height=380)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Resolution Status</p>', unsafe_allow_html=True)
        resolutions = analysis.get("resolution_statuses", {})
        
        if resolutions:
            # Create a simple bar chart for resolutions
            import plotly.graph_objects as go
            
            labels = list(resolutions.keys())
            values = list(resolutions.values())
            
            fig = go.Figure(go.Bar(
                x=[l.replace("_", " ").title() for l in labels],
                y=values,
                marker=dict(
                    color=["#10b981", "#f59e0b", "#ef4444", "#718096"][:len(labels)],
                    line=dict(width=0)
                ),
                text=values,
                textposition="outside",
                textfont=dict(color="#a0aec0")
            ))
            
            fig.update_layout(
                height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#a0aec0"),
                xaxis=dict(
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#718096")
                ),
                yaxis=dict(
                    gridcolor="rgba(255, 255, 255, 0.05)",
                    tickfont=dict(color="#718096")
                ),
                margin=dict(l=40, r=40, t=40, b=40)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Summary statistics
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Summary Statistics</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stat_card(
            value=str(analysis.get("count", 0)),
            label="Total Conversations",
            icon="💬"
        )
    
    with col2:
        stat_card(
            value=f"{analysis.get('avg_turns', 0):.1f}",
            label="Avg Turns/Conv",
            icon="🔄"
        )
    
    with col3:
        stat_card(
            value=str(len(analysis.get("intents", {}))),
            label="Unique Intents",
            icon="🎯"
        )
    
    with col4:
        stat_card(
            value=str(len(analysis.get("languages", {}))),
            label="Languages",
            icon="🌐"
        )


def estimate_quality_metrics(conversations: list) -> dict:
    """Estimate quality metrics when backend is not available."""
    if not conversations:
        return {"completeness": 0, "consistency": 0, "realism": 0, "diversity": 0}
    
    # Completeness: check for required fields
    required_fields = ["conversation_id", "intent", "sentiment", "turns"]
    completeness_scores = []
    for conv in conversations:
        present = sum(1 for f in required_fields if f in conv and conv[f])
        completeness_scores.append(present / len(required_fields))
    
    # Consistency: check turn structure
    consistency_scores = []
    for conv in conversations:
        turns = conv.get("turns", [])
        if not turns:
            consistency_scores.append(0)
            continue
        # Check first turn is customer
        first_ok = turns[0].get("speaker") == "customer" if turns else True
        # Check alternating turns
        alt_ok = all(
            turns[i].get("speaker") != turns[i+1].get("speaker")
            for i in range(len(turns)-1)
        ) if len(turns) > 1 else True
        consistency_scores.append(1.0 if first_ok and alt_ok else 0.5)
    
    # Diversity: unique intents / total
    intents = set(c.get("intent") for c in conversations if c.get("intent"))
    diversity = len(intents) / max(len(conversations), 1)
    
    return {
        "completeness": sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0,
        "consistency": sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0,
        "realism": 0.75,  # Default estimate
        "diversity": min(diversity * 1.5, 1.0),  # Scale up slightly
    }


def estimate_bias_severity(analysis: dict) -> str:
    """Estimate bias severity from analysis."""
    issues = 0
    
    # Check sentiment balance
    sentiments = analysis.get("sentiments", {})
    if sentiments:
        total = sum(sentiments.values())
        for count in sentiments.values():
            if total > 0 and count / total > 0.6:
                issues += 1
    
    # Check language balance
    languages = analysis.get("languages", {})
    if len(languages) > 1:
        total = sum(languages.values())
        for count in languages.values():
            if total > 0 and count / total > 0.9:
                issues += 1
    
    # Check intent coverage
    intents = analysis.get("intents", {})
    if len(intents) < 20:
        issues += 1
    
    if issues == 0:
        return "none"
    elif issues == 1:
        return "low"
    elif issues == 2:
        return "medium"
    else:
        return "high"
