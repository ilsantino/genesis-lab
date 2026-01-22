"""
Reusable card components for GENESIS LAB UI.

Provides domain cards, stat cards, feature lists, and metric displays.
"""

import streamlit as st
from typing import List, Optional, Dict, Any


def domain_card(
    title: str,
    description: str,
    features: List[str],
    status: str = "active",  # "active", "coming_soon", "beta"
    badge_text: str = "",
    icon: str = "",
):
    """
    Create a domain card with glassmorphism styling.
    
    Uses separate st.markdown calls to avoid nested HTML interpolation issues.
    
    Args:
        title: Card title
        description: Card description
        features: List of feature strings
        status: "active", "coming_soon", or "beta"
        badge_text: Text for status badge
        icon: Emoji icon for the card
    """
    # Badge colors based on status
    badge_colors = {
        "active": ("#10b981", "rgba(16, 185, 129, 0.2)"),
        "coming_soon": ("#f59e0b", "rgba(245, 158, 11, 0.2)"),
        "beta": ("#3b82f6", "rgba(59, 130, 246, 0.2)")
    }
    color, bg = badge_colors.get(status, badge_colors["active"])
    
    # Card header (single markdown call - no nesting)
    st.markdown(f'''
    <div style="background: rgba(30, 30, 50, 0.6); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 20px; padding: 1.5rem; margin-bottom: 0.5rem;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 1rem;">
            <span style="font-size: 2.5rem;">{icon}</span>
            <span style="background: {bg}; color: {color}; padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.75rem; font-weight: 600;">{badge_text}</span>
        </div>
        <h3 style="font-size: 1.4rem; font-weight: 700; color: white; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #a0aec0; font-size: 0.9rem; line-height: 1.6; margin-bottom: 1rem;">{description}</p>
        <div style="border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1rem; margin-top: 0.5rem;">
    ''', unsafe_allow_html=True)
    
    # Each feature as separate call (avoids nested HTML interpolation!)
    for feature in features:
        st.markdown(f'''
            <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem; color: #a0aec0; font-size: 0.85rem;">
                <span style="color: #10b981; font-size: 0.9rem;">✓</span>
                <span>{feature}</span>
            </div>
        ''', unsafe_allow_html=True)
    
    # Close container
    st.markdown('</div></div>', unsafe_allow_html=True)


def stat_card(
    value: str,
    label: str,
    icon: str = "",
    delta: str = "",
    delta_color: str = "normal"
):
    """
    Create a stat card with a prominent value.
    
    Args:
        value: Main value to display
        label: Label below the value
        icon: Emoji icon
        delta: Change indicator (e.g., "+5%")
        delta_color: "normal", "inverse", or "off"
    """
    delta_html = ""
    if delta:
        delta_style = {
            "normal": "color: #10b981;",  # Green for positive
            "inverse": "color: #ef4444;",  # Red
            "off": "color: #718096;"  # Gray
        }.get(delta_color, "color: #10b981;")
        
        delta_html = f'<span style="font-size: 0.9rem; {delta_style}">{delta}</span>'
    
    card_html = f"""
    <div class="stat-card">
        {f'<span style="font-size: 1.5rem; margin-bottom: 0.5rem; display: block;">{icon}</span>' if icon else ''}
        <div class="stat-value">{value}</div>
        <div class="stat-label">{label}</div>
        {delta_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def metric_card(
    title: str,
    value: Any,
    subtitle: str = "",
    status: str = "neutral",  # "success", "warning", "error", "neutral"
    icon: str = ""
):
    """
    Create a metric card with status indicator.
    
    Args:
        title: Metric title
        value: Metric value
        subtitle: Additional context
        status: Status for coloring
        icon: Emoji icon
    """
    status_colors = {
        "success": ("#10b981", "rgba(16, 185, 129, 0.1)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.1)"),
        "neutral": ("#667eea", "rgba(102, 126, 234, 0.1)")
    }
    
    color, bg_color = status_colors.get(status, status_colors["neutral"])
    
    card_html = f"""
    <div style="
        background: {bg_color};
        border: 1px solid {color}33;
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    ">
        <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.5rem;">
            {f'<span style="font-size: 1.25rem;">{icon}</span>' if icon else ''}
            <span style="color: #a0aec0; font-size: 0.9rem; font-weight: 500;">{title}</span>
        </div>
        <div style="font-size: 2rem; font-weight: 700; color: {color}; margin-bottom: 0.25rem;">
            {value}
        </div>
        {f'<div style="color: #718096; font-size: 0.85rem;">{subtitle}</div>' if subtitle else ''}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def feature_list(features: List[Dict[str, str]], columns: int = 2):
    """
    Create a feature list with icons in a symmetric grid.
    
    Args:
        features: List of dicts with 'icon', 'title', 'description'
        columns: Number of columns
    """
    # Process features in rows for better symmetry
    rows = [features[i:i + columns] for i in range(0, len(features), columns)]
    
    for row in rows:
        cols = st.columns(columns)
        for idx, feature in enumerate(row):
            with cols[idx]:
                feature_html = f"""
                <div style="
                    background: rgba(30, 30, 50, 0.4);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 1.25rem;
                    margin-bottom: 1rem;
                    min-height: 140px;
                ">
                    <div style="font-size: 1.5rem; margin-bottom: 0.75rem;">
                        {feature.get('icon', '✨')}
                    </div>
                    <h4 style="color: white; font-size: 1rem; font-weight: 600; margin-bottom: 0.5rem;">
                        {feature.get('title', '')}
                    </h4>
                    <p style="color: #a0aec0; font-size: 0.9rem; margin: 0; line-height: 1.5;">
                        {feature.get('description', '')}
                    </p>
                </div>
                """
                st.markdown(feature_html, unsafe_allow_html=True)


def info_banner(
    message: str,
    type: str = "info",  # "info", "success", "warning", "error"
    icon: str = ""
):
    """
    Create an info banner.
    
    Args:
        message: Banner message
        type: Banner type for styling
        icon: Emoji icon
    """
    colors = {
        "info": ("#3b82f6", "rgba(59, 130, 246, 0.1)"),
        "success": ("#10b981", "rgba(16, 185, 129, 0.1)"),
        "warning": ("#f59e0b", "rgba(245, 158, 11, 0.1)"),
        "error": ("#ef4444", "rgba(239, 68, 68, 0.1)")
    }
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    color, bg_color = colors.get(type, colors["info"])
    default_icon = icons.get(type, "ℹ️")
    
    banner_html = f"""
    <div style="
        background: {bg_color};
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    ">
        <span style="font-size: 1.25rem;">{icon or default_icon}</span>
        <span style="color: #e2e8f0;">{message}</span>
    </div>
    """
    
    st.markdown(banner_html, unsafe_allow_html=True)


def conversation_preview(turns: List[Dict[str, str]], max_turns: int = 6):
    """
    Display a conversation preview.
    
    Args:
        turns: List of turn dicts with 'speaker' and 'text'
        max_turns: Maximum turns to display
    """
    for turn in turns[:max_turns]:
        speaker = turn.get('speaker', 'customer')
        text = turn.get('text', '')
        
        speaker_class = "customer" if speaker == "customer" else "agent"
        speaker_label = "Customer" if speaker == "customer" else "Agent"
        
        turn_html = f"""
        <div class="conversation-turn {speaker_class}">
            <div class="speaker-label {speaker_class}">{speaker_label}</div>
            <div style="color: #e2e8f0; font-size: 0.95rem; line-height: 1.5;">
                {text}
            </div>
        </div>
        """
        
        st.markdown(turn_html, unsafe_allow_html=True)
    
    if len(turns) > max_turns:
        st.markdown(
            f'<p style="text-align: center; color: #718096; font-size: 0.9rem; margin-top: 1rem;">'
            f'... and {len(turns) - max_turns} more turns</p>',
            unsafe_allow_html=True
        )


def loading_spinner(message: str = "Loading..."):
    """
    Display a loading spinner with message.
    
    Args:
        message: Loading message to display
    """
    spinner_html = f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 3rem;
        text-align: center;
    ">
        <div style="
            width: 48px;
            height: 48px;
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <p style="color: #a0aec0; margin-top: 1rem; font-size: 1rem;">{message}</p>
    </div>
    <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
    """
    st.markdown(spinner_html, unsafe_allow_html=True)


def skeleton_card(height: int = 120, count: int = 1):
    """
    Display skeleton placeholder cards during loading.
    
    Args:
        height: Height of each skeleton card in pixels
        count: Number of skeleton cards to display
    """
    for _ in range(count):
        skeleton_html = f"""
        <div style="
            background: linear-gradient(90deg, rgba(30, 30, 50, 0.6) 25%, rgba(50, 50, 70, 0.6) 50%, rgba(30, 30, 50, 0.6) 75%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 12px;
            height: {height}px;
            margin-bottom: 1rem;
        "></div>
        <style>
            @keyframes shimmer {{
                0% {{ background-position: -200% 0; }}
                100% {{ background-position: 200% 0; }}
            }}
        </style>
        """
        st.markdown(skeleton_html, unsafe_allow_html=True)


def error_state(
    title: str = "Something went wrong",
    message: str = "An error occurred. Please try again.",
    icon: str = "❌",
    show_retry: bool = True,
    retry_key: str = "retry_btn"
) -> bool:
    """
    Display an error state with optional retry button.
    
    Args:
        title: Error title
        message: Error message
        icon: Emoji icon
        show_retry: Whether to show retry button
        retry_key: Unique key for retry button
    
    Returns:
        True if retry button was clicked
    """
    error_html = f"""
    <div style="
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="font-size: 3rem; margin-bottom: 1rem;">{icon}</div>
        <h3 style="color: #ef4444; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #a0aec0; font-size: 0.95rem; margin-bottom: 1.5rem;">{message}</p>
    </div>
    """
    st.markdown(error_html, unsafe_allow_html=True)
    
    if show_retry:
        return st.button("🔄 Try Again", key=retry_key, use_container_width=True)
    return False


def empty_state(
    icon: str = "📭",
    title: str = "No data found",
    message: str = "There's nothing here yet.",
    action_text: str = None,
    action_key: str = "empty_action_btn"
) -> bool:
    """
    Display an empty state placeholder.
    
    Args:
        icon: Emoji icon
        title: Empty state title
        message: Descriptive message
        action_text: Optional action button text
        action_key: Unique key for action button
    
    Returns:
        True if action button was clicked
    """
    empty_html = f"""
    <div style="
        background: rgba(30, 30, 50, 0.4);
        border: 2px dashed rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 3rem;
        text-align: center;
        margin: 1rem 0;
    ">
        <div style="font-size: 4rem; margin-bottom: 1rem; opacity: 0.7;">{icon}</div>
        <h3 style="color: white; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;">{title}</h3>
        <p style="color: #718096; font-size: 0.95rem; margin-bottom: 1.5rem;">{message}</p>
    </div>
    """
    st.markdown(empty_html, unsafe_allow_html=True)
    
    if action_text:
        return st.button(action_text, key=action_key, use_container_width=True)
    return False


def loading_overlay(message: str = "Processing..."):
    """
    Display a full-width loading overlay.
    
    Args:
        message: Loading message to display
    """
    overlay_html = f"""
    <div style="
        background: rgba(15, 15, 35, 0.9);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.3);
    ">
        <div style="
            width: 56px;
            height: 56px;
            margin: 0 auto 1rem;
            border: 4px solid rgba(102, 126, 234, 0.2);
            border-top: 4px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        "></div>
        <p style="color: white; font-size: 1.1rem; font-weight: 500; margin-bottom: 0.5rem;">{message}</p>
        <p style="color: #718096; font-size: 0.9rem;">Please wait...</p>
    </div>
    """
    st.markdown(overlay_html, unsafe_allow_html=True)


def page_header(icon: str, title: str, subtitle: str):
    """
    Create a standardized page header.
    
    Args:
        icon: Emoji icon for the page
        title: Page title
        subtitle: Page description
    """
    header_html = f"""
    <div style="margin-bottom: 2rem;">
        <h1 style="
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        ">{icon} {title}</h1>
        <p style="
            color: #a0aec0;
            font-size: 1.1rem;
            margin: 0;
            line-height: 1.6;
        ">{subtitle}</p>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def quick_action_card(
    title: str,
    description: str,
    icon: str,
    button_text: str,
    button_key: str
) -> bool:
    """
    Create a quick action card with button.
    
    Args:
        title: Card title
        description: Card description
        icon: Emoji icon
        button_text: Button text
        button_key: Unique key for button
    
    Returns:
        True if button was clicked
    """
    card_html = f"""
    <div style="
        background: rgba(30, 30, 50, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        height: 100%;
        transition: all 0.3s ease;
    ">
        <div style="font-size: 2rem; margin-bottom: 1rem;">{icon}</div>
        <h4 style="color: white; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">
            {title}
        </h4>
        <p style="color: #a0aec0; font-size: 0.9rem; margin-bottom: 1rem;">
            {description}
        </p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    return st.button(button_text, key=button_key, use_container_width=True)
