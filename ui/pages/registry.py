"""
Registry Page - Dataset browser and management.

Provides an interface to browse, search, and manage generated datasets.
"""

import streamlit as st
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, stat_card, metric_card, page_header, empty_state
from ui.components.charts import timeline_chart

# Try to import backend
BACKEND_AVAILABLE = False
try:
    from src.registry.database import DatasetRegistry
    BACKEND_AVAILABLE = True
except ImportError:
    pass


def render_registry_page():
    """Render the registry page."""
    page_header(
        icon="📁",
        title="Dataset Registry",
        subtitle="Browse, search, and manage your generated datasets."
    )
    
    # Load datasets
    datasets = load_datasets()
    
    if not datasets:
        if empty_state(
            icon="📭",
            title="No Datasets Found",
            message="Generate some synthetic data to populate the registry.",
            action_text="⚡ Go to Generate",
            action_key="empty_go_generate"
        ):
            st.session_state.current_page = "Generate"
            st.rerun()
        return
    
    # Summary stats
    render_summary_stats(datasets)
    
    get_divider()
    
    # View toggle
    col_toggle, col_search = st.columns([1, 2])
    
    with col_toggle:
        view_mode = st.radio(
            "View Mode",
            options=["Table", "Cards"],
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_search:
        search_query = st.text_input(
            "Search datasets",
            placeholder="Search by name, domain, or ID...",
            label_visibility="collapsed"
        )
    
    # Filter datasets
    if search_query:
        datasets = filter_datasets(datasets, search_query)
    
    if not datasets:
        info_banner("No datasets match your search.", type="info")
        return
    
    # Render based on view mode
    if view_mode == "Table":
        render_table_view(datasets)
    else:
        render_card_view(datasets)


def load_datasets() -> list:
    """Load datasets from registry and file system."""
    datasets = []
    
    # Try to load from database registry
    if BACKEND_AVAILABLE:
        try:
            registry = DatasetRegistry()
            db_datasets = registry.list_datasets()
            datasets.extend(db_datasets)
        except Exception:
            pass
    
    # Also scan file system for JSON files
    data_dir = Path("data/synthetic")
    if data_dir.exists():
        for json_file in data_dir.glob("*.json"):
            # Skip report files
            if json_file.name.endswith(".report.json"):
                continue
            
            # Check if already in registry
            file_path = str(json_file)
            if any(d.get("file_path") == file_path for d in datasets):
                continue
            
            # Load basic info from file
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    # Extract info from filename
                    name = json_file.stem
                    parts = name.split("_")
                    
                    # Try to get date from filename
                    gen_date = None
                    for part in parts:
                        if len(part) == 8 and part.isdigit():
                            try:
                                gen_date = datetime.strptime(part, "%Y%m%d").isoformat()
                            except ValueError:
                                pass
                    
                    datasets.append({
                        "id": f"file_{json_file.stem}",
                        "domain": "customer_service",
                        "size": len(data),
                        "generation_date": gen_date or json_file.stat().st_mtime,
                        "file_path": file_path,
                        "file_format": "json",
                        "quality_score": None,
                        "source": "file"
                    })
            except Exception:
                continue
    
    # Sort by date (newest first)
    datasets.sort(key=lambda x: x.get("generation_date") or "", reverse=True)
    
    return datasets


def filter_datasets(datasets: list, query: str) -> list:
    """Filter datasets by search query."""
    query = query.lower()
    filtered = []
    
    for ds in datasets:
        searchable = " ".join([
            str(ds.get("id", "")),
            str(ds.get("domain", "")),
            str(ds.get("file_path", "")),
            str(ds.get("notes", ""))
        ]).lower()
        
        if query in searchable:
            filtered.append(ds)
    
    return filtered


def render_summary_stats(datasets: list):
    """Render summary statistics."""
    total_datasets = len(datasets)
    total_samples = sum(d.get("size", 0) for d in datasets)
    
    # Calculate quality stats
    quality_scores = [d.get("quality_score") for d in datasets if d.get("quality_score")]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    # Get domains
    domains = set(d.get("domain", "unknown") for d in datasets)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stat_card(
            value=str(total_datasets),
            label="Total Datasets",
            icon="📁"
        )
    
    with col2:
        stat_card(
            value=f"{total_samples:,}",
            label="Total Samples",
            icon="💬"
        )
    
    with col3:
        stat_card(
            value=f"{avg_quality:.1f}%" if avg_quality else "N/A",
            label="Avg Quality",
            icon="✨"
        )
    
    with col4:
        stat_card(
            value=str(len(domains)),
            label="Domains",
            icon="🎯"
        )


def render_table_view(datasets: list):
    """Render datasets as a table."""
    st.markdown("### Datasets")
    
    # Prepare dataframe
    df_data = []
    for ds in datasets:
        df_data.append({
            "ID": ds.get("id", "")[:15] + "..." if len(ds.get("id", "")) > 15 else ds.get("id", ""),
            "Domain": ds.get("domain", "").replace("_", " ").title(),
            "Samples": ds.get("size", 0),
            "Quality": f"{ds.get('quality_score', 0):.1f}%" if ds.get("quality_score") else "N/A",
            "Date": format_date(ds.get("generation_date")),
            "File": Path(ds.get("file_path", "")).name if ds.get("file_path") else "N/A",
        })
    
    df = pd.DataFrame(df_data)
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Domain": st.column_config.TextColumn("Domain", width="medium"),
            "Samples": st.column_config.NumberColumn("Samples", width="small"),
            "Quality": st.column_config.TextColumn("Quality", width="small"),
            "Date": st.column_config.TextColumn("Date", width="medium"),
            "File": st.column_config.TextColumn("File", width="large"),
        }
    )
    
    get_divider()
    
    # Dataset details
    st.markdown("### Dataset Details")
    
    selected_idx = st.selectbox(
        "Select a dataset to view details",
        options=range(len(datasets)),
        format_func=lambda i: f"{datasets[i].get('id', 'Unknown')} - {datasets[i].get('size', 0)} samples"
    )
    
    if selected_idx is not None:
        render_dataset_details(datasets[selected_idx])


def render_card_view(datasets: list):
    """Render datasets as cards."""
    st.markdown("### Datasets")
    
    # Create grid
    cols_per_row = 3
    
    for i in range(0, len(datasets), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(datasets):
                break
            
            ds = datasets[idx]
            
            with col:
                render_dataset_card(ds, idx)


def render_dataset_card(dataset: dict, idx: int):
    """Render a single dataset card."""
    quality = dataset.get("quality_score")
    quality_str = f"{quality:.1f}%" if quality else "N/A"
    
    quality_color = "#10b981" if quality and quality >= 80 else "#f59e0b" if quality and quality >= 70 else "#718096"
    
    # Truncate ID smartly
    dataset_id = dataset.get('id', 'Unknown')
    display_id = dataset_id[:20] + "..." if len(dataset_id) > 20 else dataset_id
    
    card_html = f"""
    <div style="
        background: rgba(30, 30, 50, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 0.75rem;
        min-height: 180px;
        transition: all 0.3s ease;
    ">
        <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 1rem;">
            <span style="font-size: 1.5rem;">📁</span>
            <span style="
                background: rgba(102, 126, 234, 0.2);
                color: #667eea;
                padding: 0.25rem 0.5rem;
                border-radius: 6px;
                font-size: 0.75rem;
                font-weight: 600;
            ">{dataset.get('domain', 'unknown').replace('_', ' ').title()}</span>
        </div>
        
        <h4 style="color: white; font-size: 1rem; margin-bottom: 0.5rem; word-break: break-all; min-height: 1.5rem;">
            {display_id}
        </h4>
        
        <div style="display: flex; gap: 1.5rem; margin-bottom: 1rem;">
            <div>
                <span style="color: #718096; font-size: 0.75rem;">Samples</span>
                <p style="color: white; font-weight: 600; margin: 0;">{dataset.get('size', 0):,}</p>
            </div>
            <div>
                <span style="color: #718096; font-size: 0.75rem;">Quality</span>
                <p style="color: {quality_color}; font-weight: 600; margin: 0;">{quality_str}</p>
            </div>
        </div>
        
        <p style="color: #718096; font-size: 0.8rem; margin: 0;">
            {format_date(dataset.get('generation_date'))}
        </p>
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Validate", key=f"val_{idx}", use_container_width=True):
            st.session_state.current_page = "Validate"
            st.session_state.validate_file = dataset.get("file_path")
            st.rerun()
    with col2:
        if st.button("⬇️ Download", key=f"dl_{idx}", use_container_width=True):
            download_dataset(dataset)


def render_dataset_details(dataset: dict):
    """Render detailed view of a dataset."""
    file_path = dataset.get('file_path', '')
    file_size = get_file_size_str(file_path) if file_path else "Unknown"
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Dataset Info</p>', unsafe_allow_html=True)
        
        st.markdown(f"**ID:** `{dataset.get('id', 'Unknown')}`")
        st.markdown(f"**Domain:** {dataset.get('domain', 'Unknown').replace('_', ' ').title()}")
        st.markdown(f"**File:** `{Path(file_path).name if file_path else 'N/A'}`")
        st.markdown(f"**Size:** {file_size}")
        st.markdown(f"**Created:** {format_date(dataset.get('generation_date'))}")
        
        if dataset.get("notes"):
            st.markdown(f"**Notes:** {dataset.get('notes')}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<p style="color: white; font-weight: 600; margin-bottom: 1rem;">Statistics</p>', unsafe_allow_html=True)
        
        # Quick stats
        metric_card(
            title="Samples",
            value=f"{dataset.get('size', 0):,}",
            status="neutral",
            icon="💬"
        )
        
        quality = dataset.get("quality_score")
        if quality:
            metric_card(
                title="Quality Score",
                value=f"{quality:.1f}%",
                status="success" if quality >= 80 else "warning" if quality >= 70 else "error",
                icon="✨"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Actions and Export in two rows
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Actions</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Validate Dataset", key="detail_validate", use_container_width=True):
            st.session_state.current_page = "Validate"
            st.session_state.validate_file = dataset.get("file_path")
            st.rerun()
    
    with col2:
        if st.button("🔄 Compare with Another", key="detail_compare", use_container_width=True):
            st.session_state.current_page = "Compare"
            st.session_state.compare_file1 = dataset.get("file_path")
            st.rerun()
    
    # Export section
    st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Export</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare JSON download
        content, filename, mime = prepare_download_data(file_path, "json")
        if content:
            st.download_button(
                label="⬇️ Download JSON",
                data=content,
                file_name=filename,
                mime=mime,
                key=f"dl_json_{dataset.get('id')}",
                use_container_width=True
            )
    
    with col2:
        # Prepare CSV download
        content, filename, mime = prepare_download_data(file_path, "csv")
        if content:
            st.download_button(
                label="⬇️ Download CSV",
                data=content,
                file_name=filename,
                mime=mime,
                key=f"dl_csv_{dataset.get('id')}",
                use_container_width=True
            )


def get_file_size_str(file_path: str) -> str:
    """Get human-readable file size."""
    try:
        size_bytes = Path(file_path).stat().st_size
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    except Exception:
        return "Unknown"


def prepare_download_data(file_path: str, format: str) -> tuple:
    """
    Prepare data for download.
    
    Returns:
        Tuple of (content, filename, mime_type) or (None, None, None) on error
    """
    if not file_path or not Path(file_path).exists():
        return None, None, None
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if format == "json":
            content = json.dumps(data, indent=2, ensure_ascii=False)
            filename = Path(file_path).stem + ".json"
            mime = "application/json"
        elif format == "csv":
            if isinstance(data, list) and data:
                df = pd.json_normalize(data)
                content = df.to_csv(index=False)
                filename = Path(file_path).stem + ".csv"
                mime = "text/csv"
            else:
                return None, None, None
        else:
            return None, None, None
        
        return content, filename, mime
        
    except Exception:
        return None, None, None


def download_dataset(dataset: dict, format: str = "json"):
    """Trigger dataset download."""
    file_path = dataset.get("file_path")
    
    content, filename, mime = prepare_download_data(file_path, format)
    
    if content is None:
        st.error("Could not prepare download. File may not exist or is invalid.")
        return
    
    st.download_button(
        label=f"⬇️ Download {format.upper()}",
        data=content,
        file_name=filename,
        mime=mime,
        key=f"download_{dataset.get('id')}_{format}",
        use_container_width=True
    )


def format_date(date_value) -> str:
    """Format a date value for display."""
    if not date_value:
        return "Unknown"
    
    if isinstance(date_value, str):
        try:
            dt = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M")
        except ValueError:
            return date_value[:19] if len(date_value) > 19 else date_value
    
    if isinstance(date_value, (int, float)):
        try:
            dt = datetime.fromtimestamp(date_value)
            return dt.strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            return "Unknown"
    
    return str(date_value)
