"""
Compare Page - Side-by-side dataset comparison.

Provides tools for comparing synthetic data against reference datasets
or comparing multiple synthetic datasets.
"""

import streamlit as st
import json
from pathlib import Path
import random

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, stat_card, metric_card, conversation_preview, page_header
from ui.components.charts import (
    comparison_chart,
    sentiment_pie_chart,
    intent_distribution_chart
)

# Import utilities
from src.utils.visualization import (
    load_conversations,
    analyze_conversations,
    get_sentiment_percentages,
    compare_datasets
)


def render_compare_page():
    """Render the comparison page."""
    page_header(
        icon="🔄",
        title="Compare Datasets",
        subtitle="Compare synthetic data distributions against reference or other datasets."
    )
    
    # Dataset selection
    st.markdown("### Select Datasets")
    
    # Get available files
    json_files = get_available_files()
    reference_files = get_reference_files()
    
    all_files = {
        "Synthetic Datasets": json_files,
        "Reference Datasets": reference_files
    }
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Dataset 1**")
        
        source1 = st.selectbox(
            "Source",
            options=list(all_files.keys()),
            key="source1",
            label_visibility="collapsed"
        )
        
        file1 = st.selectbox(
            "File",
            options=[""] + all_files.get(source1, []),
            format_func=lambda x: "Select a file..." if x == "" else Path(x).name,
            key="file1",
            label_visibility="collapsed"
        )
        
        # Check for pre-selected file
        if "compare_file1" in st.session_state and not file1:
            file1 = st.session_state.compare_file1
            del st.session_state.compare_file1
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Dataset 2**")
        
        source2 = st.selectbox(
            "Source",
            options=list(all_files.keys()),
            key="source2",
            label_visibility="collapsed"
        )
        
        file2 = st.selectbox(
            "File",
            options=[""] + all_files.get(source2, []),
            format_func=lambda x: "Select a file..." if x == "" else Path(x).name,
            key="file2",
            label_visibility="collapsed"
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if not file1 or not file2:
        info_banner(
            "Select two datasets to compare.",
            type="info"
        )
        return
    
    # Load datasets
    data1 = load_data(file1)
    data2 = load_data(file2)
    
    if not data1 or not data2:
        st.error("Failed to load one or both datasets.")
        return
    
    # Run comparison
    comparison = compare_datasets(data1, data2)
    
    get_divider()
    
    # Comparison tabs
    tab_overview, tab_distributions, tab_samples = st.tabs([
        "📊 Overview",
        "📈 Distributions",
        "💬 Sample Comparison"
    ])
    
    with tab_overview:
        render_overview_tab(comparison, Path(file1).name, Path(file2).name)
    
    with tab_distributions:
        render_distributions_tab(comparison, Path(file1).name, Path(file2).name)
    
    with tab_samples:
        render_samples_tab(data1, data2, Path(file1).name, Path(file2).name)


def get_available_files() -> list:
    """Get available synthetic dataset files."""
    data_dir = Path("data/synthetic")
    if not data_dir.exists():
        return []
    
    files = []
    for f in sorted(data_dir.glob("*.json"), reverse=True):
        if not f.name.endswith(".report.json"):
            files.append(str(f))
    
    return files


def get_reference_files() -> list:
    """Get available reference dataset files."""
    ref_dir = Path("data/reference")
    if not ref_dir.exists():
        return []
    
    return [str(f) for f in sorted(ref_dir.glob("*.json"))]


def load_data(file_path: str) -> list:
    """Load data from file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def render_overview_tab(comparison: dict, name1: str, name2: str):
    """Render overview comparison."""
    st.markdown('<p class="section-header">Comparison Overview</p>', unsafe_allow_html=True)
    
    # Side-by-side stats - equal main columns with minimal VS column
    col1, col_mid, col2 = st.columns([5, 1, 5])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<p style="color: white; font-weight: 600; margin-bottom: 1rem;">{name1[:30]}</p>', unsafe_allow_html=True)
        analysis1 = comparison.get("analysis1", {})
        
        stat_card(
            value=str(analysis1.get("count", 0)),
            label="Samples",
            icon="💬"
        )
        
        metric_card(
            title="Avg Turns",
            value=f"{analysis1.get('avg_turns', 0):.1f}",
            status="neutral",
            icon="🔄"
        )
        
        metric_card(
            title="Unique Intents",
            value=str(len(analysis1.get("intents", {}))),
            status="neutral",
            icon="🎯"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_mid:
        st.markdown("""
        <div style="display: flex; align-items: center; justify-content: center; height: 100%; min-height: 300px;">
            <div style="text-align: center;">
                <span style="font-size: 2.5rem;">⚡</span>
                <p style="color: #667eea; font-size: 1rem; font-weight: 600; margin-top: 0.5rem;">VS</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown(f'<p style="color: white; font-weight: 600; margin-bottom: 1rem;">{name2[:30]}</p>', unsafe_allow_html=True)
        analysis2 = comparison.get("analysis2", {})
        
        stat_card(
            value=str(analysis2.get("count", 0)),
            label="Samples",
            icon="💬"
        )
        
        metric_card(
            title="Avg Turns",
            value=f"{analysis2.get('avg_turns', 0):.1f}",
            status="neutral",
            icon="🔄"
        )
        
        metric_card(
            title="Unique Intents",
            value=str(len(analysis2.get("intents", {}))),
            status="neutral",
            icon="🎯"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    get_divider()
    
    # Similarity metrics
    st.markdown('<p class="section-header">Similarity Metrics</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        overlap = comparison.get("intent_overlap", 0)
        metric_card(
            title="Intent Overlap",
            value=f"{overlap*100:.1f}%",
            subtitle="Shared intents between datasets",
            status="success" if overlap >= 0.7 else "warning" if overlap >= 0.5 else "error",
            icon="🎯"
        )
    
    with col2:
        turns_diff = comparison.get("avg_turns_diff", 0)
        metric_card(
            title="Avg Turns Difference",
            value=f"{turns_diff:.1f}",
            subtitle="Difference in conversation length",
            status="success" if turns_diff < 1 else "warning" if turns_diff < 2 else "error",
            icon="🔄"
        )
    
    with col3:
        # Calculate sentiment similarity
        sent_comp = comparison.get("sentiment_comparison", {})
        if sent_comp:
            avg_diff = sum(s.get("diff", 0) for s in sent_comp.values()) / len(sent_comp)
            similarity = max(0, 100 - avg_diff)
        else:
            similarity = 0
        
        metric_card(
            title="Sentiment Similarity",
            value=f"{similarity:.1f}%",
            subtitle="Distribution similarity",
            status="success" if similarity >= 80 else "warning" if similarity >= 60 else "error",
            icon="💭"
        )


def render_distributions_tab(comparison: dict, name1: str, name2: str):
    """Render distribution comparisons."""
    
    analysis1 = comparison.get("analysis1", {})
    analysis2 = comparison.get("analysis2", {})
    
    # Sentiment comparison
    st.markdown("### Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sent1 = get_sentiment_percentages(analysis1.get("sentiments", {}))
        fig = sentiment_pie_chart(sent1, title=name1[:25], height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        sent2 = get_sentiment_percentages(analysis2.get("sentiments", {}))
        fig = sentiment_pie_chart(sent2, title=name2[:25], height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    get_divider()
    
    # Intent comparison
    st.markdown("### Intent Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = intent_distribution_chart(
            analysis1.get("intents", {}),
            title=name1[:25],
            top_n=10,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = intent_distribution_chart(
            analysis2.get("intents", {}),
            title=name2[:25],
            top_n=10,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    get_divider()
    
    # Complexity comparison
    st.markdown("### Complexity Comparison")
    
    # Prepare data for grouped bar chart
    complexities1 = analysis1.get("complexities", {})
    complexities2 = analysis2.get("complexities", {})
    
    all_complexities = sorted(set(complexities1.keys()) | set(complexities2.keys()))
    
    if all_complexities:
        total1 = sum(complexities1.values()) or 1
        total2 = sum(complexities2.values()) or 1
        
        values1 = [(complexities1.get(c, 0) / total1) * 100 for c in all_complexities]
        values2 = [(complexities2.get(c, 0) / total2) * 100 for c in all_complexities]
        
        fig = comparison_chart(
            categories=[c.title() for c in all_complexities],
            dataset1_values=values1,
            dataset2_values=values2,
            dataset1_name=name1[:20],
            dataset2_name=name2[:20],
            title="Complexity Distribution (%)",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)


def render_samples_tab(data1: list, data2: list, name1: str, name2: str):
    """Render sample comparison."""
    st.markdown("### Sample Conversations")
    
    # Filter options
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        # Get common intents
        intents1 = set(c.get("intent") for c in data1 if c.get("intent"))
        intents2 = set(c.get("intent") for c in data2 if c.get("intent"))
        common_intents = sorted(intents1 & intents2)
        
        selected_intent = st.selectbox(
            "Filter by Intent",
            options=["Any"] + common_intents,
            format_func=lambda x: x.replace("_", " ").title() if x != "Any" else "Any Intent"
        )
    
    with col_filter2:
        if st.button("🔀 Random Samples", use_container_width=True):
            st.session_state.sample_seed = random.randint(0, 10000)
    
    # Get samples
    seed = st.session_state.get("sample_seed", 42)
    random.seed(seed)
    
    # Filter by intent if selected
    filtered1 = data1
    filtered2 = data2
    
    if selected_intent != "Any":
        filtered1 = [c for c in data1 if c.get("intent") == selected_intent]
        filtered2 = [c for c in data2 if c.get("intent") == selected_intent]
    
    # Get random samples
    sample1 = random.choice(filtered1) if filtered1 else None
    sample2 = random.choice(filtered2) if filtered2 else None
    
    # Display side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"#### {name1[:25]}")
        
        if sample1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            st.markdown(f"**Intent:** `{sample1.get('intent', 'N/A')}`")
            st.markdown(f"**Sentiment:** {sample1.get('sentiment', 'N/A')}")
            st.markdown(f"**Complexity:** {sample1.get('complexity', 'N/A')}")
            
            st.markdown("---")
            
            turns = sample1.get("turns", [])
            conversation_preview(turns, max_turns=6)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            info_banner("No matching samples found.", type="warning")
    
    with col2:
        st.markdown(f"#### {name2[:25]}")
        
        if sample2:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            # Handle reference format (text/intent) vs synthetic format
            if "text" in sample2 and "turns" not in sample2:
                # Reference format
                st.markdown(f"**Intent:** `{sample2.get('intent', 'N/A')}`")
                st.markdown(f"**Text:** {sample2.get('text', 'N/A')}")
            else:
                # Synthetic format
                st.markdown(f"**Intent:** `{sample2.get('intent', 'N/A')}`")
                st.markdown(f"**Sentiment:** {sample2.get('sentiment', 'N/A')}")
                st.markdown(f"**Complexity:** {sample2.get('complexity', 'N/A')}")
                
                st.markdown("---")
                
                turns = sample2.get("turns", [])
                conversation_preview(turns, max_turns=6)
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            info_banner("No matching samples found.", type="warning")
    
    # Similarity assessment
    if sample1 and sample2:
        get_divider()
        st.markdown("### Quick Assessment")
        
        assessment = assess_samples(sample1, sample2)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            metric_card(
                title="Length Match",
                value="Yes" if assessment["length_match"] else "No",
                status="success" if assessment["length_match"] else "warning",
                icon="📏"
            )
        
        with col2:
            metric_card(
                title="Sentiment Match",
                value="Yes" if assessment["sentiment_match"] else "No",
                status="success" if assessment["sentiment_match"] else "warning",
                icon="💭"
            )
        
        with col3:
            metric_card(
                title="Intent Match",
                value="Yes" if assessment["intent_match"] else "No",
                status="success" if assessment["intent_match"] else "warning",
                icon="🎯"
            )


def assess_samples(sample1: dict, sample2: dict) -> dict:
    """Assess similarity between two samples."""
    # Length comparison
    turns1 = len(sample1.get("turns", []))
    turns2 = len(sample2.get("turns", [])) if "turns" in sample2 else 1
    length_match = abs(turns1 - turns2) <= 2
    
    # Sentiment comparison
    sent1 = sample1.get("sentiment", "").lower()
    sent2 = sample2.get("sentiment", "").lower()
    sentiment_match = sent1 == sent2 if sent1 and sent2 else False
    
    # Intent comparison
    intent1 = sample1.get("intent", "").lower()
    intent2 = sample2.get("intent", "").lower()
    intent_match = intent1 == intent2 if intent1 and intent2 else False
    
    return {
        "length_match": length_match,
        "sentiment_match": sentiment_match,
        "intent_match": intent_match,
    }
