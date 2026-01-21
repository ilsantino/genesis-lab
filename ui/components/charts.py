"""
Plotly chart components for GENESIS LAB UI.

All charts use dark theme styling consistent with the app design.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import pandas as pd


# Color palette matching the app theme
CHART_COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#06b6d4",
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "neutral": "#718096",
}

# Gradient color scale
GRADIENT_COLORS = [
    "#667eea",  # Primary start
    "#764ba2",  # Primary end
    "#9f7aea",  # Purple
    "#06b6d4",  # Cyan
    "#10b981",  # Emerald
]

# Dark theme layout defaults
DARK_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "font": {"color": "#a0aec0", "family": "system-ui, sans-serif"},
    "title": {"font": {"color": "#ffffff", "size": 18}},
    "legend": {
        "bgcolor": "rgba(30, 30, 50, 0.5)",
        "bordercolor": "rgba(255, 255, 255, 0.1)",
        "borderwidth": 1,
        "font": {"color": "#a0aec0"}
    },
    "xaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.05)",
        "linecolor": "rgba(255, 255, 255, 0.1)",
        "tickfont": {"color": "#718096"},
        "title": {"font": {"color": "#a0aec0"}}
    },
    "yaxis": {
        "gridcolor": "rgba(255, 255, 255, 0.05)",
        "linecolor": "rgba(255, 255, 255, 0.1)",
        "tickfont": {"color": "#718096"},
        "title": {"font": {"color": "#a0aec0"}}
    },
    "margin": {"l": 60, "r": 30, "t": 60, "b": 60},
}

# Chart configuration for interactivity
CHART_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "modeBarButtonsToAdd": ["resetScale2d"],
    "toImageButtonOptions": {
        "format": "png",
        "filename": "genesis_chart",
        "height": 600,
        "width": 1000,
        "scale": 2
    },
    "scrollZoom": True,
}


def get_chart_config(enable_zoom: bool = True, enable_download: bool = True) -> dict:
    """
    Get chart configuration for Plotly.
    
    Args:
        enable_zoom: Enable scroll zoom
        enable_download: Enable download button
    
    Returns:
        Config dict for st.plotly_chart
    """
    config = CHART_CONFIG.copy()
    config["scrollZoom"] = enable_zoom
    
    if not enable_download:
        config["modeBarButtonsToRemove"] = config.get("modeBarButtonsToRemove", []) + ["toImage"]
    
    return config


def apply_dark_theme(fig: go.Figure, enable_hover: bool = True) -> go.Figure:
    """
    Apply dark theme to a Plotly figure.
    
    Args:
        fig: Plotly figure
        enable_hover: Enable hover interactions
    
    Returns:
        Themed figure
    """
    fig.update_layout(**DARK_LAYOUT)
    
    if enable_hover:
        fig.update_layout(
            hoverlabel=dict(
                bgcolor="rgba(30, 30, 50, 0.9)",
                font_size=13,
                font_family="system-ui, sans-serif",
                font_color="white",
                bordercolor="rgba(102, 126, 234, 0.5)"
            )
        )
    
    return fig


def intent_distribution_chart(
    intents: Dict[str, int],
    title: str = "Intent Distribution",
    top_n: int = 15,
    height: int = 400
) -> go.Figure:
    """
    Create a horizontal bar chart for intent distribution.
    
    Args:
        intents: Dict mapping intent names to counts
        title: Chart title
        top_n: Number of top intents to show
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    # Sort and get top N
    sorted_intents = sorted(intents.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    if not sorted_intents:
        return _empty_chart(title, height)
    
    labels = [item[0].replace("_", " ").title() for item in sorted_intents]
    values = [item[1] for item in sorted_intents]
    
    # Reverse for horizontal bar (top at top)
    labels = labels[::-1]
    values = values[::-1]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(
            color=values,
            colorscale=[[0, CHART_COLORS["primary"]], [1, CHART_COLORS["secondary"]]],
            line=dict(width=0)
        ),
        hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Count",
        yaxis_title="",
        showlegend=False,
    )
    
    return apply_dark_theme(fig)


def sentiment_pie_chart(
    sentiment_data: Dict[str, float],
    title: str = "Sentiment Distribution",
    height: int = 350
) -> go.Figure:
    """
    Create a donut chart for sentiment distribution.
    
    Args:
        sentiment_data: Dict with sentiment labels and percentages
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not sentiment_data:
        return _empty_chart(title, height)
    
    labels = list(sentiment_data.keys())
    values = list(sentiment_data.values())
    
    # Color mapping for sentiments
    color_map = {
        "positive": CHART_COLORS["success"],
        "negative": CHART_COLORS["error"],
        "neutral": CHART_COLORS["neutral"],
        "mixed": CHART_COLORS["warning"],
    }
    
    colors = [color_map.get(label.lower(), CHART_COLORS["primary"]) for label in labels]
    
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker=dict(colors=colors, line=dict(color='rgba(0,0,0,0)', width=0)),
        textinfo='percent',
        textfont=dict(color='white', size=12),
        hovertemplate="<b>%{label}</b><br>%{percent}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        annotations=[dict(
            text="Sentiment",
            x=0.5, y=0.5,
            font_size=14,
            font_color="#a0aec0",
            showarrow=False
        )]
    )
    
    return apply_dark_theme(fig)


def quality_gauge(
    value: float,
    title: str = "Quality Score",
    max_value: float = 100,
    thresholds: Optional[Dict[str, float]] = None,
    height: int = 250
) -> go.Figure:
    """
    Create a gauge chart for quality scores.
    
    Args:
        value: Current value
        title: Chart title
        max_value: Maximum value for the gauge
        thresholds: Dict with 'good', 'warning' threshold values
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if thresholds is None:
        thresholds = {"good": 80, "warning": 60}
    
    # Determine color based on value
    if value >= thresholds["good"]:
        bar_color = CHART_COLORS["success"]
    elif value >= thresholds["warning"]:
        bar_color = CHART_COLORS["warning"]
    else:
        bar_color = CHART_COLORS["error"]
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "%", "font": {"size": 40, "color": "white"}},
        title={"text": title, "font": {"size": 16, "color": "#a0aec0"}},
        gauge={
            "axis": {
                "range": [0, max_value],
                "tickwidth": 1,
                "tickcolor": "rgba(255,255,255,0.3)",
                "tickfont": {"color": "#718096"}
            },
            "bar": {"color": bar_color, "thickness": 0.75},
            "bgcolor": "rgba(30, 30, 50, 0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, thresholds["warning"]], "color": "rgba(239, 68, 68, 0.2)"},
                {"range": [thresholds["warning"], thresholds["good"]], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [thresholds["good"], max_value], "color": "rgba(16, 185, 129, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": value
            }
        }
    ))
    
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
    )
    
    return apply_dark_theme(fig)


def language_bar_chart(
    language_data: Dict[str, float],
    title: str = "Language Distribution",
    height: int = 300
) -> go.Figure:
    """
    Create a bar chart for language distribution.
    
    Args:
        language_data: Dict with language codes and percentages
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not language_data:
        return _empty_chart(title, height)
    
    labels = list(language_data.keys())
    values = list(language_data.values())
    
    # Map language codes to full names
    lang_names = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
    }
    
    display_labels = [lang_names.get(l.lower(), l.upper()) for l in labels]
    
    fig = go.Figure(go.Bar(
        x=display_labels,
        y=values,
        marker=dict(
            color=[CHART_COLORS["primary"], CHART_COLORS["secondary"]] * len(labels),
            line=dict(width=0)
        ),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(color="#a0aec0"),
        hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Language",
        yaxis_title="Percentage",
        yaxis=dict(range=[0, max(values) * 1.2] if values else [0, 100]),
        showlegend=False,
    )
    
    return apply_dark_theme(fig)


def timeline_chart(
    dates: List[str],
    values: List[float],
    title: str = "Dataset Timeline",
    y_label: str = "Count",
    height: int = 350
) -> go.Figure:
    """
    Create a timeline/line chart.
    
    Args:
        dates: List of date strings
        values: List of values
        title: Chart title
        y_label: Y-axis label
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not dates or not values:
        return _empty_chart(title, height)
    
    fig = go.Figure()
    
    # Add area fill
    fig.add_trace(go.Scatter(
        x=dates,
        y=values,
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.2)',
        line=dict(color=CHART_COLORS["primary"], width=2),
        mode='lines+markers',
        marker=dict(size=8, color=CHART_COLORS["primary"]),
        hovertemplate="<b>%{x}</b><br>" + y_label + ": %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Date",
        yaxis_title=y_label,
        showlegend=False,
    )
    
    return apply_dark_theme(fig)


def comparison_chart(
    categories: List[str],
    dataset1_values: List[float],
    dataset2_values: List[float],
    dataset1_name: str = "Dataset 1",
    dataset2_name: str = "Dataset 2",
    title: str = "Dataset Comparison",
    height: int = 400
) -> go.Figure:
    """
    Create a grouped bar chart comparing two datasets.
    
    Args:
        categories: List of category names
        dataset1_values: Values for first dataset
        dataset2_values: Values for second dataset
        dataset1_name: Name of first dataset
        dataset2_name: Name of second dataset
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not categories:
        return _empty_chart(title, height)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name=dataset1_name,
        x=categories,
        y=dataset1_values,
        marker_color=CHART_COLORS["primary"],
        hovertemplate=f"<b>{dataset1_name}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>"
    ))
    
    fig.add_trace(go.Bar(
        name=dataset2_name,
        x=categories,
        y=dataset2_values,
        marker_color=CHART_COLORS["secondary"],
        hovertemplate=f"<b>{dataset2_name}</b><br>%{{x}}: %{{y:.2f}}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        barmode='group',
        xaxis_title="Category",
        yaxis_title="Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    
    return apply_dark_theme(fig)


def complexity_distribution_chart(
    complexity_data: Dict[str, int],
    title: str = "Complexity Distribution",
    height: int = 300
) -> go.Figure:
    """
    Create a chart for complexity distribution.
    
    Args:
        complexity_data: Dict with complexity levels and counts
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not complexity_data:
        return _empty_chart(title, height)
    
    # Order complexity levels
    order = ["simple", "medium", "complex"]
    labels = [l for l in order if l in complexity_data]
    values = [complexity_data[l] for l in labels]
    
    colors = [CHART_COLORS["success"], CHART_COLORS["warning"], CHART_COLORS["error"]]
    
    fig = go.Figure(go.Bar(
        x=[l.title() for l in labels],
        y=values,
        marker=dict(color=colors[:len(labels)], line=dict(width=0)),
        text=values,
        textposition="outside",
        textfont=dict(color="#a0aec0"),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis_title="Complexity",
        yaxis_title="Count",
        showlegend=False,
    )
    
    return apply_dark_theme(fig)


def metrics_radar_chart(
    metrics: Dict[str, float],
    title: str = "Quality Metrics",
    height: int = 400
) -> go.Figure:
    """
    Create a radar chart for multiple metrics.
    
    Args:
        metrics: Dict with metric names and values (0-1 scale)
        title: Chart title
        height: Chart height in pixels
    
    Returns:
        Plotly figure
    """
    if not metrics:
        return _empty_chart(title, height)
    
    categories = list(metrics.keys())
    values = [v * 100 for v in metrics.values()]  # Convert to percentage
    
    # Close the radar chart
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(102, 126, 234, 0.3)',
        line=dict(color=CHART_COLORS["primary"], width=2),
        marker=dict(size=8, color=CHART_COLORS["primary"]),
        hovertemplate="<b>%{theta}</b><br>%{r:.1f}%<extra></extra>"
    ))
    
    fig.update_layout(
        title=title,
        height=height,
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="#718096"),
                gridcolor="rgba(255, 255, 255, 0.1)",
            ),
            angularaxis=dict(
                tickfont=dict(color="#a0aec0"),
                gridcolor="rgba(255, 255, 255, 0.1)",
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    
    return apply_dark_theme(fig)


def _empty_chart(title: str, height: int) -> go.Figure:
    """Create an empty chart with a 'No Data' message."""
    fig = go.Figure()
    
    fig.add_annotation(
        text="No data available",
        x=0.5, y=0.5,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=16, color="#718096")
    )
    
    fig.update_layout(
        title=title,
        height=height,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )
    
    return apply_dark_theme(fig)
