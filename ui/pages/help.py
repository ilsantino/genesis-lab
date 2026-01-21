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
    tab_start, tab_features, tab_concepts, tab_faq = st.tabs([
        "🚀 Getting Started",
        "⚙️ Features Guide",
        "📖 Concepts",
        "❓ FAQ"
    ])
    
    with tab_start:
        render_getting_started()
    
    with tab_features:
        render_features_guide()
    
    with tab_concepts:
        render_concepts()
    
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
