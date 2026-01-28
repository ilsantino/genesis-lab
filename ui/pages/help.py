"""
Help Page - Documentation and user guide for Genesis Lab.

Provides comprehensive documentation on how to use the platform,
including getting started guide, features reference, concepts, and FAQ.
"""

import streamlit as st

# Import components
from ui.components.styles import get_divider, COLORS
from ui.components.cards import page_header, info_banner, stat_card


def render_help_page():
    """Render the help/documentation page."""
    page_header(
        icon="📚",
        title="Documentation",
        subtitle="Learn how to use Genesis Lab for synthetic data generation."
    )
    
    # Documentation tabs
    tab_start, tab_features, tab_concepts, tab_models, tab_faq = st.tabs([
        "🚀 Getting Started",
        "⚙️ Features Guide",
        "📖 Concepts",
        "🤖 Models & Training",
        "❓ FAQ"
    ])
    
    with tab_start:
        render_getting_started()
    
    with tab_features:
        render_features_guide()
    
    with tab_concepts:
        render_concepts()
    
    with tab_models:
        render_models_training()
    
    with tab_faq:
        render_faq()


def render_getting_started():
    """Render the getting started tab."""
    
    # Welcome section
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: white; margin-bottom: 1rem;">Welcome to Genesis Lab</h3>
        <p style="color: #a0aec0; line-height: 1.7;">
            Genesis Lab is an enterprise-grade synthetic data generation platform designed for 
            creating realistic, diverse, and bias-free datasets for training AI models. 
            Built on AWS Bedrock with Claude 3.5 Sonnet, it specializes in generating 
            customer service conversations following the Banking77 taxonomy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    get_divider()
    
    # Quick Start Steps
    st.markdown('<p class="section-header">Quick Start Guide</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">1️⃣</div>
            <h4 style="color: white; margin-bottom: 0.5rem;">Generate Data</h4>
            <p style="color: #a0aec0; font-size: 0.9rem;">
                Go to the <strong>Generate</strong> page, configure your settings 
                (samples, language, thresholds), and click Start Generation.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">2️⃣</div>
            <h4 style="color: white; margin-bottom: 0.5rem;">Validate Quality</h4>
            <p style="color: #a0aec0; font-size: 0.9rem;">
                Use the <strong>Validate</strong> page to check quality metrics, 
                detect bias, and analyze distributions in your dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 200px;">
            <div style="font-size: 2.5rem; margin-bottom: 1rem;">3️⃣</div>
            <h4 style="color: white; margin-bottom: 0.5rem;">Train & Export</h4>
            <p style="color: #a0aec0; font-size: 0.9rem;">
                Train intent classifiers on the <strong>Training</strong> page, 
                then export your data from the <strong>Registry</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Prerequisites
    st.markdown('<p class="section-header">Prerequisites</p>', unsafe_allow_html=True)
    
    col_req1, col_req2 = st.columns([1, 1])
    
    with col_req1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">🔧 System Requirements</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li>Python 3.10 or higher</li>
                <li>UV package manager (recommended)</li>
                <li>8GB RAM minimum</li>
                <li>Internet connection for AWS</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col_req2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">🔑 AWS Setup</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li>AWS Account with Bedrock access</li>
                <li>IAM credentials configured</li>
                <li>Claude 3.5 Sonnet model enabled</li>
                <li>Region: us-east-1 (recommended)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Installation
    st.markdown('<p class="section-header">Installation</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0aec0; margin-bottom: 1rem;">Install dependencies and run the UI:</p>
        <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; color: #10b981; overflow-x: auto;">
# Clone the repository
git clone https://github.com/ilsantino/genesis-lab.git
cd genesis-lab

# Install dependencies with UV
uv sync

# Create .env file from template
cp .env.template .env
# Edit .env with your AWS credentials

# Run the UI
uv run streamlit run ui/app.py</pre>
    </div>
    """, unsafe_allow_html=True)


def render_features_guide():
    """Render the features guide tab."""
    
    # Generate Feature
    st.markdown('<p class="section-header">Generate Synthetic Data</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">⚡ Configuration Options</h4>
            <table style="width: 100%; color: #a0aec0;">
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Samples</strong></td>
                    <td>5 - 1,000 conversations</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Language</strong></td>
                    <td>English or Spanish</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Quality Threshold</strong></td>
                    <td>50% - 100%</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Bias Threshold</strong></td>
                    <td>none / low / medium / high</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">💰 Cost & Time Estimates</h4>
            <table style="width: 100%; color: #a0aec0;">
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Time per sample</strong></td>
                    <td>~60 seconds (with rate limiting)</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Cost per sample</strong></td>
                    <td>~$0.003 USD</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>100 samples</strong></td>
                    <td>~1.5 hours, ~$0.30</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>1,000 samples</strong></td>
                    <td>~16 hours, ~$3.00</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Validate Feature
    st.markdown('<p class="section-header">Validate Datasets</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">📊 Quality Metrics</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li><strong>Completeness:</strong> All required fields present</li>
                <li><strong>Consistency:</strong> Logical conversation flow</li>
                <li><strong>Realism:</strong> Natural language patterns</li>
                <li><strong>Diversity:</strong> Variety in responses</li>
                <li><strong>Overall Score:</strong> Weighted average (0-100)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">⚖️ Bias Detection</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li><strong>Sentiment Balance:</strong> pos/neu/neg distribution</li>
                <li><strong>Language Balance:</strong> EN/ES ratio</li>
                <li><strong>Complexity Balance:</strong> simple/medium/complex</li>
                <li><strong>Intent Coverage:</strong> % of 77 intents covered</li>
                <li><strong>Severity Levels:</strong> none / low / medium / high</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Training Feature
    st.markdown('<p class="section-header">Train Models</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">🎓 Model Types</h4>
            <table style="width: 100%; color: #a0aec0;">
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Logistic Regression</strong></td>
                    <td>Fast, good baseline</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Random Forest</strong></td>
                    <td>Balanced performance</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>XGBoost</strong></td>
                    <td>Highest accuracy</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">⚡ Quick Presets</h4>
            <table style="width: 100%; color: #a0aec0;">
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Fast</strong></td>
                    <td>LogReg, quick training</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Balanced</strong></td>
                    <td>RandomForest, good accuracy</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem 0;"><strong>Best</strong></td>
                    <td>XGBoost, max performance</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Registry & Compare
    st.markdown('<p class="section-header">Registry & Compare</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">📁 Dataset Registry</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li>Browse all generated datasets</li>
                <li>Search by name, domain, or ID</li>
                <li>View detailed metadata</li>
                <li>Track quality scores over time</li>
                <li>Export as JSON or CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">🔄 Compare Datasets</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li>Side-by-side comparison</li>
                <li>Similarity metrics (intent overlap)</li>
                <li>Distribution comparison charts</li>
                <li>Sample conversation previews</li>
                <li>Compare synthetic vs reference</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)


def render_concepts():
    """Render the concepts tab."""
    
    # Quality Metrics
    st.markdown('<p class="section-header">Quality Metrics</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0aec0; margin-bottom: 1.5rem;">
            Genesis Lab evaluates synthetic data quality using four key metrics, 
            each scored from 0.0 to 1.0. The overall score is a weighted average 
            converted to a 0-100 scale.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #10b981; margin-bottom: 0.5rem;">✓ Completeness</h4>
            <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
                Measures whether all required fields are present and non-empty.
                Checks for conversation_id, intent, sentiment, turns, etc.
            </p>
            <p style="color: #718096; font-size: 0.85rem;">Target: 95%+</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="color: #f59e0b; margin-bottom: 0.5rem;">💬 Realism</h4>
            <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
                Evaluates how natural and realistic the generated text appears.
                Compares distribution against reference Banking77 dataset.
            </p>
            <p style="color: #718096; font-size: 0.85rem;">Target: 70%+</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #667eea; margin-bottom: 0.5rem;">🔗 Consistency</h4>
            <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
                Verifies logical conversation flow: first turn is customer,
                turns alternate correctly, no duplicate messages.
            </p>
            <p style="color: #718096; font-size: 0.85rem;">Target: 90%+</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="color: #ec4899; margin-bottom: 0.5rem;">🎨 Diversity</h4>
            <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
                Measures variety in vocabulary, intents, and response patterns.
                Higher diversity prevents model overfitting.
            </p>
            <p style="color: #718096; font-size: 0.85rem;">Target: 80%+</p>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Banking77 Taxonomy
    st.markdown('<p class="section-header">Banking77 Intent Taxonomy</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0aec0; margin-bottom: 1.5rem;">
            Genesis Lab uses the Banking77 taxonomy - a standard benchmark for intent 
            classification in banking/fintech customer service. It contains 
            <strong style="color: white;">77 intents</strong> organized into 
            <strong style="color: white;">11 categories</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    categories = [
        ("💳 Card Management", ["card_arrival", "card_linking", "card_not_working", "card_payment_fee_charged"]),
        ("🏦 Account Services", ["activate_my_card", "age_limit", "apple_pay_or_google_pay", "atm_support"]),
        ("💸 Payments", ["balance_not_updated", "beneficiary_not_allowed", "cancel_transfer", "cash_withdrawal_charge"]),
        ("🔧 Technical", ["card_swallowed", "compromised_card", "contactless_not_working", "country_support"]),
        ("📊 Balance & Statements", ["balance_not_updated", "getting_spare_card", "getting_virtual_card"]),
        ("🔐 Security", ["compromised_card", "lost_or_stolen_card", "pin_blocked", "wrong_amount_of_cash"]),
    ]
    
    for i, col in enumerate([col1, col2, col3]):
        with col:
            for j in range(2):
                idx = i * 2 + j
                if idx < len(categories):
                    cat_name, intents = categories[idx]
                    st.markdown(f"""
                    <div class="glass-card" style="min-height: 140px;">
                        <h5 style="color: white; margin-bottom: 0.5rem;">{cat_name}</h5>
                        <p style="color: #718096; font-size: 0.8rem; line-height: 1.6;">
                            {', '.join(intents[:3])}...
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
    
    get_divider()
    
    # Dataset Schema
    st.markdown('<p class="section-header">Dataset Schema</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">📄 Conversation Structure</h4>
            <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; color: #10b981; font-size: 0.8rem; overflow-x: auto;">
{
  "conversation_id": "conv_abc123",
  "intent": "card_arrival",
  "category": "card_management",
  "sentiment": "neutral",
  "complexity": "medium",
  "language": "en",
  "turn_count": 4,
  "resolution_status": "resolved",
  "turns": [...]
}</pre>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">💬 Turn Structure</h4>
            <pre style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; color: #10b981; font-size: 0.8rem; overflow-x: auto;">
{
  "turns": [
    {
      "speaker": "customer",
      "text": "Hi, when will my..."
    },
    {
      "speaker": "agent", 
      "text": "Hello! I'd be happy..."
    }
  ]
}</pre>
        </div>
        """, unsafe_allow_html=True)


def render_models_training():
    """Render the Models & Training deep-dive tab."""
    
    # Main explainer
    st.markdown("""
    <div class="glass-card" style="border-left: 4px solid #667eea;">
        <h3 style="color: white; margin-bottom: 1rem;">Understanding the Genesis Lab Pipeline</h3>
        <p style="color: #a0aec0; line-height: 1.8;">
            Genesis Lab is a <strong style="color: white;">synthetic data generation</strong> platform. 
            This page explains exactly what happens at each stage, what models are used, and how 
            training works. <strong style="color: #f59e0b;">Important:</strong> You do NOT train the AI 
            that generates data - you train a separate classifier.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    get_divider()
    
    # Generation vs Training - The key distinction
    st.markdown('<p class="section-header">Generation vs Training: The Key Distinction</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #667eea; min-height: 320px;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">🤖 GENERATION (Claude AI)</h4>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">What it is:</p>
            <p style="color: #a0aec0; margin-bottom: 1rem;">
                Claude 3.5 Sonnet (via AWS Bedrock) creates synthetic conversations based on your settings.
            </p>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Is it trained?</p>
            <p style="color: #a0aec0; margin-bottom: 1rem;">
                <strong style="color: #10b981;">Already trained by Anthropic.</strong> You cannot and do not need to train it.
            </p>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Your role:</p>
            <p style="color: #a0aec0;">
                Configure settings (samples, language, thresholds) and let Claude generate data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card" style="border-top: 4px solid #ec4899; min-height: 320px;">
            <h4 style="color: #ec4899; margin-bottom: 1rem;">📊 TRAINING (Classifier)</h4>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">What it is:</p>
            <p style="color: #a0aec0; margin-bottom: 1rem;">
                A machine learning model that learns to classify intents from the synthetic data.
            </p>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Is it trained?</p>
            <p style="color: #a0aec0; margin-bottom: 1rem;">
                <strong style="color: #f59e0b;">You train it!</strong> Using the synthetic data Claude generated.
            </p>
            <p style="color: white; font-weight: 600; margin-bottom: 0.5rem;">Your role:</p>
            <p style="color: #a0aec0;">
                Choose a model (LogReg, RF, XGBoost), configure settings, and train on your data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Pipeline visualization
    st.markdown("""
    <div class="glass-card" style="text-align: center; margin-top: 1rem;">
        <p style="color: #718096; margin-bottom: 0.5rem;">THE COMPLETE FLOW:</p>
        <p style="color: #a0aec0; font-size: 1.1rem; letter-spacing: 0.3px;">
            <span style="background: rgba(102, 126, 234, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Your Settings</span>
            <span style="color: #4a5568;"> → </span>
            <span style="background: rgba(102, 126, 234, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Claude AI</span>
            <span style="color: #4a5568;"> → </span>
            <span style="background: rgba(16, 185, 129, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Synthetic Data</span>
            <span style="color: #4a5568;"> → </span>
            <span style="background: rgba(16, 185, 129, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Validation</span>
            <span style="color: #4a5568;"> → </span>
            <span style="background: rgba(236, 72, 153, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Train Classifier</span>
            <span style="color: #4a5568;"> → </span>
            <span style="background: rgba(245, 158, 11, 0.2); padding: 0.3rem 0.6rem; border-radius: 4px;">Predict Intents!</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    get_divider()
    
    # Available Models Deep Dive
    st.markdown('<p class="section-header">Classification Models Reference</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0aec0; margin-bottom: 1rem;">
            All classifiers use <strong style="color: white;">TF-IDF vectorization</strong> to convert text to numbers. 
            TF-IDF measures how important each word is by combining word frequency with how unique the word is across all documents.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Logistic Regression
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <h4 style="color: #667eea; margin-bottom: 1rem;">⚡ Logistic Regression</h4>
        <table style="width: 100%; color: #a0aec0;">
            <tr>
                <td style="padding: 0.5rem 0; width: 30%;"><strong>Speed</strong></td>
                <td style="color: #10b981;">Very Fast (seconds)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Accuracy</strong></td>
                <td>Good baseline</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>How it works</strong></td>
                <td>Finds a linear boundary to separate intent classes. Simple but effective.</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Best for</strong></td>
                <td>Quick experiments, debugging, establishing a baseline</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Preset</strong></td>
                <td>⚡ Fast</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Random Forest
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <h4 style="color: #10b981; margin-bottom: 1rem;">🌲 Random Forest</h4>
        <table style="width: 100%; color: #a0aec0;">
            <tr>
                <td style="padding: 0.5rem 0; width: 30%;"><strong>Speed</strong></td>
                <td style="color: #f59e0b;">Medium (10-30 seconds)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Accuracy</strong></td>
                <td>Better than LogReg, more robust</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>How it works</strong></td>
                <td>Builds many decision trees and combines their votes. Reduces overfitting.</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Best for</strong></td>
                <td>Production with limited data, when you need feature importance</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Preset</strong></td>
                <td>⚖️ Balanced</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # XGBoost
    st.markdown("""
    <div class="glass-card" style="margin-top: 1rem;">
        <h4 style="color: #ec4899; margin-bottom: 1rem;">🎯 XGBoost</h4>
        <table style="width: 100%; color: #a0aec0;">
            <tr>
                <td style="padding: 0.5rem 0; width: 30%;"><strong>Speed</strong></td>
                <td style="color: #ef4444;">Slower (30-60 seconds)</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Accuracy</strong></td>
                <td>Best overall performance</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>How it works</strong></td>
                <td>Gradient boosting: builds trees sequentially, each correcting previous errors.</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Best for</strong></td>
                <td>Final production model, when accuracy is critical</td>
            </tr>
            <tr>
                <td style="padding: 0.5rem 0;"><strong>Preset</strong></td>
                <td>🎯 Best</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    get_divider()
    
    # Validation Methods
    st.markdown('<p class="section-header">Validation Methods Explained</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #a0aec0; margin-bottom: 1rem;">
            Before training, Genesis Lab validates your synthetic data using these techniques:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col_val1, col_val2 = st.columns(2)
    
    with col_val1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #10b981; margin-bottom: 1rem;">📊 Quality Metrics</h4>
            <ul style="color: #a0aec0; line-height: 2.2;">
                <li><strong>Completeness:</strong> Are all required fields present?</li>
                <li><strong>Consistency:</strong> Does the conversation flow logically?</li>
                <li><strong>Realism:</strong> Does the distribution match Banking77?</li>
                <li><strong>Diversity:</strong> Is there variety in responses?</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="color: #667eea; margin-bottom: 1rem;">📏 Distribution Matching</h4>
            <p style="color: #a0aec0; line-height: 1.8;">
                Uses <strong style="color: white;">Jensen-Shannon Divergence</strong> to compare your 
                synthetic data distribution against the Banking77 reference dataset. 
                Lower divergence = more realistic data.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_val2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #f59e0b; margin-bottom: 1rem;">⚖️ Bias Detection</h4>
            <ul style="color: #a0aec0; line-height: 2.2;">
                <li><strong>Sentiment Balance:</strong> pos/neutral/neg distribution</li>
                <li><strong>Language Balance:</strong> English/Spanish ratio</li>
                <li><strong>Intent Coverage:</strong> % of 77 intents represented</li>
                <li><strong>Complexity Balance:</strong> simple/medium/complex</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="glass-card" style="margin-top: 1rem;">
            <h4 style="color: #ec4899; margin-bottom: 1rem;">🔍 Semantic Coherence</h4>
            <p style="color: #a0aec0; line-height: 1.8;">
                Optionally uses <strong style="color: white;">sentence embeddings</strong> (BERT-based) 
                to check if agent responses are semantically related to customer queries.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    get_divider()
    
    # Complete Glossary
    st.markdown('<p class="section-header">Complete Glossary</p>', unsafe_allow_html=True)
    
    with st.expander("🔤 A-F", expanded=False):
        st.markdown("""
        | Term | Definition |
        |------|------------|
        | **Accuracy** | Percentage of predictions that are correct |
        | **AWS Bedrock** | Amazon's managed service for accessing LLMs like Claude |
        | **Banking77** | Benchmark dataset with 77 customer service intents |
        | **Bias** | Systematic skew in data that can affect model fairness |
        | **Classification** | Task of assigning categories (intents) to inputs |
        | **Claude** | Anthropic's AI model used for data generation |
        | **Cross-Validation** | Testing on multiple data splits for reliable evaluation |
        | **F1 Score** | Harmonic mean of precision and recall |
        """)
    
    with st.expander("🔤 G-N", expanded=False):
        st.markdown("""
        | Term | Definition |
        |------|------------|
        | **Generation** | Process of creating synthetic data using Claude |
        | **Gradient Boosting** | ML technique where models are trained sequentially to correct errors |
        | **Intent** | The purpose or goal behind a customer's message |
        | **Jensen-Shannon Divergence** | Measure of similarity between two probability distributions |
        | **Logistic Regression** | Linear model for classification |
        | **Max Features** | Maximum number of vocabulary terms in TF-IDF |
        | **N-gram** | Sequence of N consecutive words (unigram=1, bigram=2, trigram=3) |
        """)
    
    with st.expander("🔤 O-Z", expanded=False):
        st.markdown("""
        | Term | Definition |
        |------|------------|
        | **Overfitting** | When a model memorizes training data instead of learning patterns |
        | **Precision** | Of all positive predictions, how many are correct |
        | **Random Forest** | Ensemble of decision trees that vote on predictions |
        | **Recall** | Of all actual positives, how many were correctly identified |
        | **Synthetic Data** | Artificially generated data that mimics real data |
        | **TF-IDF** | Term Frequency-Inverse Document Frequency - text vectorization method |
        | **Training** | Process of teaching a model to recognize patterns in data |
        | **Validation** | Checking data quality before using it for training |
        | **XGBoost** | Extreme Gradient Boosting - powerful tree-based algorithm |
        """)
    
    # Quick Reference Card
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Quick Reference</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="border: 1px solid #4a5568;">
        <h4 style="color: white; margin-bottom: 1rem;">🎯 TL;DR - The 30-Second Summary</h4>
        <ol style="color: #a0aec0; line-height: 2;">
            <li><strong>Generate:</strong> Claude creates fake but realistic conversations</li>
            <li><strong>Validate:</strong> Check the data is good quality and unbiased</li>
            <li><strong>Train:</strong> Teach a classifier (LogReg/RF/XGBoost) to recognize intents</li>
            <li><strong>Use:</strong> The trained classifier predicts intents for new messages</li>
        </ol>
        <p style="color: #f59e0b; margin-top: 1rem;">
            <strong>Remember:</strong> You train the classifier, NOT Claude. Claude is already trained.
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_faq():
    """Render the FAQ tab."""
    
    st.markdown('<p class="section-header">Frequently Asked Questions</p>', unsafe_allow_html=True)
    
    # General Questions
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white; margin-bottom: 1rem;">📋 General</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("What is Genesis Lab?"):
        st.markdown("""
        Genesis Lab is a synthetic data generation platform that creates realistic 
        customer service conversations for training AI models. It uses AWS Bedrock 
        with Claude 3.5 Sonnet to generate high-quality, diverse, and bias-free 
        datasets following the Banking77 taxonomy.
        """)
    
    with st.expander("What can I use the generated data for?"):
        st.markdown("""
        - **Intent Classification**: Train models to classify customer intents
        - **Chatbot Training**: Build customer service chatbots
        - **NLU Systems**: Develop natural language understanding pipelines
        - **Testing**: Create test datasets for QA
        - **Research**: Academic research on conversational AI
        """)
    
    with st.expander("Is the data production-ready?"):
        st.markdown("""
        Data with a quality score above 70% and bias severity of "low" or "none" 
        is considered production-ready. Always validate your data using the 
        Validate page before deploying to production systems.
        """)
    
    get_divider()
    
    # Training Questions - NEW SECTION
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white; margin-bottom: 1rem;">🎓 Training</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Do I need to train the AI model (Claude)?"):
        st.markdown("""
        **No!** This is the most common misconception.
        
        - **Claude** (the AI that generates data) is already trained by Anthropic
        - You cannot and do not need to train Claude
        - What you *can* train is a **separate classifier** that learns from the synthetic data
        
        Think of it this way:
        - Claude = An author writing example conversations (already skilled)
        - Classifier = A student learning from those examples (you train this)
        """)
    
    with st.expander("What's the difference between generation and training?"):
        st.markdown("""
        | | Generation | Training |
        |---|---|---|
        | **What** | Creating synthetic data | Teaching a model to classify |
        | **Who** | Claude AI (AWS Bedrock) | Classifier (LogReg/RF/XGBoost) |
        | **Input** | Your settings | Synthetic conversations |
        | **Output** | Conversations | Trained model |
        | **Your role** | Configure settings | Choose model, run training |
        
        **Flow:** Generation → Validation → Training → Use model
        """)
    
    with st.expander("Why do I need to validate before training?"):
        st.markdown("""
        **Garbage in, garbage out.**
        
        If you train on bad data, your model will learn bad patterns:
        - Missing fields → Model crashes during training
        - Biased sentiment → Model predicts same sentiment always
        - Low diversity → Model can't generalize to new intents
        - Inconsistent conversations → Model learns wrong patterns
        
        Validation catches these issues **before** they waste your training time.
        
        **Rule of thumb:** Quality score > 70%, Bias severity = "low" or "none"
        """)
    
    with st.expander("What accuracy should I expect?"):
        st.markdown("""
        With 77 intents, random guessing = 1.3% accuracy. Here's what's realistic:
        
        | Dataset Size | Expected Accuracy | Notes |
        |--------------|------------------|-------|
        | 100 samples | 10-20% | Limited, but 10x better than random |
        | 500 samples | 40-50% | Good for prototypes |
        | 1,000 samples | 60-70% | Ready for initial production |
        | 5,000 samples | 80-85% | Production-ready |
        | 10,000+ samples | 90%+ | State-of-the-art |
        
        **Tip:** Quality matters more than quantity. Focus on diverse, high-quality data.
        """)
    
    with st.expander("Which model should I use?"):
        st.markdown("""
        **Start with Balanced (Random Forest) if unsure.**
        
        | Situation | Recommended |
        |-----------|-------------|
        | Just testing, need speed | ⚡ Fast (Logistic Regression) |
        | Production, general use | ⚖️ Balanced (Random Forest) |
        | Need max accuracy | 🎯 Best (XGBoost) |
        | Less than 500 samples | Random Forest |
        | More than 1000 samples | XGBoost |
        
        All models use the same TF-IDF text processing, so the data prep is identical.
        """)
    
    with st.expander("What is cross-validation and should I use it?"):
        st.markdown("""
        **Cross-validation tests your model multiple times for more reliable results.**
        
        **How 5-fold CV works:**
        1. Split data into 5 equal parts
        2. Train on 4 parts, test on 1
        3. Repeat 5 times with different test parts
        4. Average the results
        
        **When to use:**
        - Final evaluation before production
        - When you want confidence intervals (±)
        - When dataset is small (< 500 samples)
        
        **When to skip:**
        - Quick experiments
        - Large datasets (5000+) - single split is reliable enough
        - Time is critical
        """)
    
    get_divider()
    
    # Technical Questions
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white; margin-bottom: 1rem;">🔧 Technical</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("Why is generation slow?"):
        st.markdown("""
        Generation includes a 5-second delay between API calls to respect 
        AWS Bedrock rate limits. This prevents throttling errors and ensures 
        100% success rate. For large datasets (1000+), consider running 
        generation overnight.
        """)
    
    with st.expander("What model is used for generation?"):
        st.markdown("""
        Genesis Lab uses **Claude 3.5 Sonnet** (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`) 
        via AWS Bedrock. This model excels at generating natural, contextually 
        appropriate customer service conversations.
        """)
    
    with st.expander("How accurate is the intent classifier?"):
        st.markdown("""
        Classifier accuracy depends on dataset size:
        - **100 samples**: ~15% accuracy (baseline, 77 classes)
        - **500 samples**: ~40-50% accuracy
        - **1000+ samples**: ~60-70% accuracy
        - **5000+ samples**: ~85%+ accuracy (production-grade)
        
        Random baseline for 77 classes is 1.3%, so even 15% is 11x better than random.
        """)
    
    with st.expander("What are the AWS requirements?"):
        st.markdown("""
        - AWS account with Bedrock access enabled
        - IAM user with `AmazonBedrockFullAccess` permission
        - Claude 3.5 Sonnet model enabled in Bedrock console
        - Region: `us-east-1` recommended (supports cross-region inference)
        - Credentials configured via `.env` file or AWS CLI
        """)
    
    get_divider()
    
    # Troubleshooting
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: white; margin-bottom: 1rem;">🛠️ Troubleshooting</h4>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("ThrottlingException errors"):
        st.markdown("""
        **Problem**: AWS Bedrock rate limit exceeded.
        
        **Solution**: 
        - Increase delay between calls (default: 5 seconds)
        - Reduce batch size
        - Wait and retry later
        - Request quota increase from AWS
        """)
    
    with st.expander("ValidationException: Model not supported"):
        st.markdown("""
        **Problem**: Model ID not recognized or not enabled.
        
        **Solution**:
        - Ensure Claude 3.5 Sonnet is enabled in Bedrock console
        - Use the correct model ID with `us.` prefix: 
          `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
        - Check your region supports the model
        """)
    
    with st.expander("Low quality scores"):
        st.markdown("""
        **Problem**: Generated data has quality score below 70%.
        
        **Solutions**:
        - Regenerate failed samples
        - Check for API errors in logs
        - Verify prompt templates are intact
        - Ensure reference dataset is loaded correctly
        """)
    
    get_divider()
    
    # Resources
    st.markdown('<p class="section-header">External Resources</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">📚 Documentation</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li><a href="https://docs.aws.amazon.com/bedrock/" style="color: #667eea;">AWS Bedrock Documentation</a></li>
                <li><a href="https://huggingface.co/datasets/PolyAI/banking77" style="color: #667eea;">Banking77 Dataset</a></li>
                <li><a href="https://arxiv.org/abs/2003.04807" style="color: #667eea;">Banking77 Paper</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: white; margin-bottom: 1rem;">🔗 Project Links</h4>
            <ul style="color: #a0aec0; line-height: 2;">
                <li><a href="https://github.com/ilsantino/genesis-lab" style="color: #667eea;">GitHub Repository</a></li>
                <li><a href="https://github.com/ilsantino/genesis-lab/blob/main/docs/ARCHITECTURE.md" style="color: #667eea;">Architecture Docs</a></li>
                <li><a href="https://github.com/ilsantino/genesis-lab/blob/main/README.md" style="color: #667eea;">README</a></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
