"""
Custom CSS styles and theme for GENESIS LAB.

Provides a dark mode theme with glassmorphism effects,
gradient accents, and modern UI styling.
"""

import streamlit as st

# Color Palette
COLORS = {
    # Primary gradient
    "primary_start": "#667eea",
    "primary_end": "#764ba2",
    
    # Accent colors
    "accent_cyan": "#06b6d4",
    "accent_emerald": "#10b981",
    "accent_amber": "#f59e0b",
    "accent_rose": "#f43f5e",
    
    # Background colors
    "bg_dark": "#0f0f23",
    "bg_card": "rgba(30, 30, 50, 0.7)",
    "bg_hover": "rgba(102, 126, 234, 0.1)",
    
    # Text colors
    "text_primary": "#ffffff",
    "text_secondary": "#a0aec0",
    "text_muted": "#718096",
    
    # Status colors
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "info": "#3b82f6",
    
    # Border
    "border": "rgba(255, 255, 255, 0.1)",
}


def get_gradient_css(direction: str = "135deg") -> str:
    """Get CSS gradient string."""
    return f"linear-gradient({direction}, {COLORS['primary_start']}, {COLORS['primary_end']})"


def inject_custom_css():
    """Inject custom CSS for dark theme and glassmorphism effects."""
    
    css = f"""
    <style>
        /* ===== ROOT VARIABLES ===== */
        :root {{
            --primary-start: {COLORS['primary_start']};
            --primary-end: {COLORS['primary_end']};
            --bg-dark: {COLORS['bg_dark']};
            --bg-card: {COLORS['bg_card']};
            --text-primary: {COLORS['text_primary']};
            --text-secondary: {COLORS['text_secondary']};
            --border-color: {COLORS['border']};
            --accent-cyan: {COLORS['accent_cyan']};
            --accent-emerald: {COLORS['accent_emerald']};
            --accent-amber: {COLORS['accent_amber']};
            --accent-rose: {COLORS['accent_rose']};
        }}
        
        /* ===== GLOBAL STYLES ===== */
        .stApp {{
            background: linear-gradient(180deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        }}
        
        /* Hide Streamlit branding */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}
        
        /* ===== TYPOGRAPHY ===== */
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 700 !important;
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .main h1 {{
            font-size: 3rem !important;
            margin-bottom: 1rem;
        }}
        
        p, span, label {{
            color: var(--text-secondary) !important;
        }}
        
        /* ===== GLASSMORPHISM CARDS ===== */
        .glass-card {{
            background: rgba(30, 30, 50, 0.6);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            transition: all 0.3s ease;
        }}
        
        .glass-card:hover {{
            background: rgba(30, 30, 50, 0.8);
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 32px rgba(102, 126, 234, 0.15);
        }}
        
        /* ===== DOMAIN CARDS ===== */
        .domain-card {{
            background: rgba(30, 30, 50, 0.6);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 20px;
            padding: 2rem;
            height: 100%;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }}
        
        .domain-card::before {{
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, var(--primary-start), var(--primary-end));
        }}
        
        .domain-card.active {{
            border-color: rgba(102, 126, 234, 0.4);
        }}
        
        .domain-card.coming-soon {{
            opacity: 0.7;
        }}
        
        .domain-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        }}
        
        .domain-title {{
            font-size: 1.5rem;
            font-weight: 700;
            color: white !important;
            margin-bottom: 0.5rem;
        }}
        
        .domain-description {{
            color: var(--text-secondary);
            font-size: 0.95rem;
            line-height: 1.6;
            margin-bottom: 1rem;
        }}
        
        /* ===== STAT CARDS ===== */
        .stat-card {{
            background: rgba(30, 30, 50, 0.5);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.25rem;
            text-align: center;
        }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.25rem;
        }}
        
        .stat-label {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        /* ===== BADGES ===== */
        .badge {{
            display: inline-block;
            padding: 0.35rem 0.75rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .badge-active {{
            background: linear-gradient(135deg, var(--accent-emerald), #059669);
            color: white;
        }}
        
        .badge-coming-soon {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--accent-amber);
            border: 1px solid rgba(245, 158, 11, 0.3);
        }}
        
        .badge-beta {{
            background: rgba(102, 126, 234, 0.2);
            color: var(--primary-start);
            border: 1px solid rgba(102, 126, 234, 0.3);
        }}
        
        /* ===== BUTTONS ===== */
        .stButton > button {{
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end)) !important;
            color: white !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.75rem 2rem !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
        }}
        
        .stButton > button:active {{
            transform: translateY(0) !important;
        }}
        
        /* Secondary button style */
        .secondary-btn > button {{
            background: transparent !important;
            border: 2px solid var(--primary-start) !important;
            color: var(--primary-start) !important;
            box-shadow: none !important;
        }}
        
        .secondary-btn > button:hover {{
            background: rgba(102, 126, 234, 0.1) !important;
        }}
        
        /* ===== FORM ELEMENTS ===== */
        .stSelectbox > div > div {{
            background: rgba(30, 30, 50, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
        }}
        
        .stTextInput > div > div > input {{
            background: rgba(30, 30, 50, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        
        .stNumberInput > div > div > input {{
            background: rgba(30, 30, 50, 0.6) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 10px !important;
            color: white !important;
        }}
        
        .stSlider > div > div {{
            background: rgba(102, 126, 234, 0.3) !important;
        }}
        
        .stSlider > div > div > div {{
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end)) !important;
        }}
        
        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {{
            background: rgba(30, 30, 50, 0.4);
            border-radius: 10px;
            padding: 0.25rem;
            gap: 0.5rem;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 8px;
            color: var(--text-secondary);
            font-weight: 500;
            padding: 0.75rem 1.5rem;
        }}
        
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            color: white !important;
        }}
        
        /* ===== SIDEBAR ===== */
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #1a1a2e 0%, #0f0f23 100%);
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }}
        
        [data-testid="stSidebar"] .stRadio > label {{
            color: white !important;
            font-weight: 500;
        }}
        
        /* ===== METRICS ===== */
        [data-testid="stMetricValue"] {{
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 700;
        }}
        
        [data-testid="stMetricDelta"] {{
            color: var(--accent-emerald) !important;
        }}
        
        /* ===== DATA FRAMES ===== */
        .stDataFrame {{
            background: rgba(30, 30, 50, 0.4) !important;
            border-radius: 12px;
            overflow: hidden;
        }}
        
        .stDataFrame thead th {{
            background: rgba(102, 126, 234, 0.2) !important;
            color: white !important;
            font-weight: 600;
        }}
        
        .stDataFrame tbody td {{
            color: var(--text-secondary) !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
        }}
        
        /* ===== EXPANDER ===== */
        .streamlit-expanderHeader {{
            background: rgba(30, 30, 50, 0.4) !important;
            border-radius: 10px !important;
            color: white !important;
            font-weight: 500;
        }}
        
        .streamlit-expanderContent {{
            background: rgba(30, 30, 50, 0.2) !important;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 0 0 10px 10px;
        }}
        
        /* ===== PROGRESS BAR ===== */
        .stProgress > div > div {{
            background: linear-gradient(90deg, var(--primary-start), var(--primary-end)) !important;
            border-radius: 10px;
        }}
        
        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {{
            width: 8px;
            height: 8px;
        }}
        
        ::-webkit-scrollbar-track {{
            background: rgba(30, 30, 50, 0.4);
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb {{
            background: linear-gradient(135deg, var(--primary-start), var(--primary-end));
            border-radius: 4px;
        }}
        
        ::-webkit-scrollbar-thumb:hover {{
            background: var(--primary-end);
        }}
        
        /* ===== ALERTS ===== */
        .stAlert {{
            border-radius: 12px !important;
            border: none !important;
        }}
        
        .stSuccess {{
            background: rgba(16, 185, 129, 0.1) !important;
            border-left: 4px solid var(--accent-emerald) !important;
        }}
        
        .stWarning {{
            background: rgba(245, 158, 11, 0.1) !important;
            border-left: 4px solid var(--accent-amber) !important;
        }}
        
        .stError {{
            background: rgba(239, 68, 68, 0.1) !important;
            border-left: 4px solid var(--accent-rose) !important;
        }}
        
        .stInfo {{
            background: rgba(59, 130, 246, 0.1) !important;
            border-left: 4px solid var(--info) !important;
        }}
        
        /* ===== FEATURE LIST ===== */
        .feature-item {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            padding: 0.5rem 0;
            color: var(--text-secondary);
        }}
        
        .feature-icon {{
            width: 24px;
            height: 24px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: rgba(102, 126, 234, 0.2);
            border-radius: 6px;
            font-size: 0.8rem;
        }}
        
        /* ===== SECTION HEADERS ===== */
        .section-header {{
            font-size: 1.25rem !important;
            font-weight: 600 !important;
            color: white !important;
            margin-bottom: 1rem !important;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}
        
        /* ===== HERO SECTION ===== */
        .hero-title {{
            font-size: 4rem !important;
            font-weight: 800 !important;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            line-height: 1.1;
        }}
        
        .hero-subtitle {{
            font-size: 1.25rem;
            color: var(--text-secondary);
            max-width: 600px;
            line-height: 1.7;
        }}
        
        /* ===== CONVERSATION PREVIEW ===== */
        .conversation-turn {{
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 12px;
            max-width: 85%;
        }}
        
        .conversation-turn.customer {{
            background: rgba(102, 126, 234, 0.15);
            border: 1px solid rgba(102, 126, 234, 0.2);
            margin-right: auto;
        }}
        
        .conversation-turn.agent {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.2);
            margin-left: auto;
        }}
        
        .speaker-label {{
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }}
        
        .speaker-label.customer {{
            color: var(--primary-start);
        }}
        
        .speaker-label.agent {{
            color: var(--accent-emerald);
        }}
        
        /* ===== ANIMATIONS ===== */
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .fade-in {{
            animation: fadeIn 0.5s ease-out;
        }}
        
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.5; }}
        }}
        
        .pulse {{
            animation: pulse 2s infinite;
        }}
        
        @keyframes gradient-shift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        
        .gradient-animate {{
            background: linear-gradient(270deg, var(--primary-start), var(--primary-end), var(--accent-cyan));
            background-size: 200% 200%;
            animation: gradient-shift 5s ease infinite;
        }}
        
        /* ===== RESPONSIVE DESIGN ===== */
        @media (max-width: 1024px) {{
            .hero-title {{
                font-size: 3rem !important;
            }}
            
            .hero-subtitle {{
                font-size: 1.1rem;
            }}
            
            .domain-card {{
                padding: 1.5rem;
            }}
            
            .stat-value {{
                font-size: 2rem;
            }}
        }}
        
        @media (max-width: 768px) {{
            .hero-title {{
                font-size: 2.5rem !important;
            }}
            
            .hero-subtitle {{
                font-size: 1rem;
                max-width: 100%;
            }}
            
            .domain-card {{
                padding: 1.25rem;
                margin-bottom: 1rem;
            }}
            
            .domain-title {{
                font-size: 1.25rem;
            }}
            
            .domain-description {{
                font-size: 0.9rem;
            }}
            
            .glass-card {{
                padding: 1rem;
                border-radius: 12px;
            }}
            
            .stat-card {{
                padding: 1rem;
            }}
            
            .stat-value {{
                font-size: 1.75rem;
            }}
            
            .stat-label {{
                font-size: 0.75rem;
            }}
            
            .section-header {{
                font-size: 1.1rem !important;
            }}
            
            .conversation-turn {{
                padding: 0.75rem;
                max-width: 95%;
            }}
        }}
        
        @media (max-width: 480px) {{
            .hero-title {{
                font-size: 2rem !important;
            }}
            
            .domain-card {{
                padding: 1rem;
            }}
            
            .badge {{
                font-size: 0.65rem;
                padding: 0.25rem 0.5rem;
            }}
            
            .feature-item {{
                font-size: 0.85rem;
            }}
        }}
    </style>
    """
    
    st.markdown(css, unsafe_allow_html=True)


def get_page_header(title: str, subtitle: str = "", icon: str = ""):
    """Create a styled page header."""
    header_html = f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="font-size: 2.5rem; margin-bottom: 0.5rem;">
            {icon} {title}
        </h1>
        <p style="font-size: 1.1rem; color: #a0aec0; margin: 0;">
            {subtitle}
        </p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def get_divider():
    """Create a styled divider."""
    st.markdown(
        '<hr style="border: none; height: 1px; background: linear-gradient(90deg, transparent, rgba(102, 126, 234, 0.3), transparent); margin: 2rem 0;">',
        unsafe_allow_html=True
    )
