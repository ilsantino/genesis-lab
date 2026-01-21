"""
Training Page - Model training and evaluation interface.

Provides tools for training intent classifiers with different configurations,
running cross-validation, and tracking experiments.
"""

import streamlit as st
import json
from pathlib import Path
from datetime import datetime

# Import components
from ui.components.styles import get_divider
from ui.components.cards import info_banner, stat_card, metric_card, page_header, empty_state
from ui.components.charts import quality_gauge, metrics_radar_chart

# Try to import backend
BACKEND_AVAILABLE = False
REGISTRY_AVAILABLE = False

try:
    from src.training import (
        Trainer,
        IntentClassifier,
        get_preset,
        list_presets,
        PRESETS,
        CVResult,
        ExperimentResult,
    )
    from src.utils.visualization import load_conversations
    BACKEND_AVAILABLE = True
except ImportError:
    pass

try:
    from src.registry import DatasetRegistry
    REGISTRY_AVAILABLE = True
except ImportError:
    pass


def render_training_page():
    """Render the training page."""
    page_header(
        icon="🎓",
        title="Model Training",
        subtitle="Train and evaluate intent classification models."
    )
    
    # Check backend availability
    if not BACKEND_AVAILABLE:
        info_banner(
            "Demo Mode - Training backend not connected. Features are simulated.",
            type="warning",
            icon="🔧"
        )
    
    # Training tabs
    tab_train, tab_evaluate, tab_history = st.tabs([
        "🚀 Train Model",
        "📊 Evaluate",
        "📜 History"
    ])
    
    with tab_train:
        render_train_tab()
    
    with tab_evaluate:
        render_evaluate_tab()
    
    with tab_history:
        render_history_tab()


def render_train_tab():
    """Render the training configuration and execution tab."""
    
    # Main layout
    col_config, col_results = st.columns([1, 1])
    
    with col_config:
        st.markdown('<p class="section-header">Training Configuration</p>', unsafe_allow_html=True)
        
        # Dataset selection
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Select Dataset**")
        
        # Option to upload or select existing
        data_source = st.radio(
            "Data Source",
            options=["existing", "upload"],
            format_func=lambda x: "Select Existing" if x == "existing" else "Upload File",
            horizontal=True,
            label_visibility="collapsed"
        )
        
        dataset_path = None
        conversations = []
        
        if data_source == "existing":
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
            
            if selected_file:
                dataset_path = str(data_dir / selected_file)
                if BACKEND_AVAILABLE:
                    conversations = load_conversations(dataset_path)
                else:
                    try:
                        with open(dataset_path) as f:
                            conversations = json.load(f)
                    except Exception:
                        pass
        else:
            uploaded_file = st.file_uploader(
                "Upload JSON file",
                type=["json"],
                label_visibility="collapsed"
            )
            if uploaded_file:
                try:
                    conversations = json.load(uploaded_file)
                    dataset_path = uploaded_file.name
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        if conversations:
            st.success(f"Loaded {len(conversations)} conversations")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Preset selection
        st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Quick Presets</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        preset_col1, preset_col2, preset_col3 = st.columns(3)
        
        # Initialize session state for preset
        if "selected_preset" not in st.session_state:
            st.session_state.selected_preset = "balanced"
        
        with preset_col1:
            if st.button("⚡ Fast", use_container_width=True, 
                        type="primary" if st.session_state.selected_preset == "fast" else "secondary"):
                st.session_state.selected_preset = "fast"
                st.rerun()
            st.caption("LogisticRegression\nQuick training")
        
        with preset_col2:
            if st.button("⚖️ Balanced", use_container_width=True,
                        type="primary" if st.session_state.selected_preset == "balanced" else "secondary"):
                st.session_state.selected_preset = "balanced"
                st.rerun()
            st.caption("RandomForest\nGood accuracy")
        
        with preset_col3:
            if st.button("🎯 Best", use_container_width=True,
                        type="primary" if st.session_state.selected_preset == "best" else "secondary"):
                st.session_state.selected_preset = "best"
                st.rerun()
            st.caption("XGBoost\nHighest accuracy")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Model configuration
        st.markdown('<p class="section-header" style="margin-top: 1.5rem;">Model Configuration</p>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        # Model type
        model_type = st.selectbox(
            "Model Type",
            options=["logistic_regression", "random_forest", "xgboost"],
            format_func=lambda x: {
                "logistic_regression": "Logistic Regression",
                "random_forest": "Random Forest",
                "xgboost": "XGBoost"
            }.get(x, x),
            index=["logistic_regression", "random_forest", "xgboost"].index(
                _get_preset_model_type(st.session_state.selected_preset)
            )
        )
        
        # Cross-validation
        use_cv = st.checkbox("Enable Cross-Validation", value=False)
        cv_folds = 5
        if use_cv:
            cv_folds = st.slider("Number of Folds", min_value=3, max_value=10, value=5)
        
        # Advanced options
        with st.expander("Advanced Options"):
            max_features = st.slider(
                "Max Features (TF-IDF)",
                min_value=1000,
                max_value=10000,
                value=5000,
                step=1000
            )
            
            ngram_min, ngram_max = st.columns(2)
            with ngram_min:
                ngram_lower = st.number_input("N-gram Min", min_value=1, max_value=3, value=1)
            with ngram_max:
                ngram_upper = st.number_input("N-gram Max", min_value=1, max_value=3, value=2)
            
            test_size = st.slider(
                "Test Split",
                min_value=0.1,
                max_value=0.4,
                value=0.2,
                step=0.05,
                format="%.2f"
            )
            
            save_model = st.checkbox("Save Trained Model", value=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Train button
        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
        
        train_disabled = not conversations
        train_clicked = st.button(
            "🚀 Start Training",
            use_container_width=True,
            type="primary",
            disabled=train_disabled
        )
        
        if train_disabled:
            info_banner("Select a dataset to enable training.", type="info")
        
        if train_clicked and conversations:
            run_training(
                conversations=conversations,
                model_type=model_type,
                use_cv=use_cv,
                cv_folds=cv_folds,
                max_features=max_features,
                ngram_range=(int(ngram_lower), int(ngram_upper)),
                test_size=test_size,
                save_model=save_model,
                dataset_path=dataset_path
            )
    
    with col_results:
        st.markdown('<p class="section-header">Training Results</p>', unsafe_allow_html=True)
        
        if "last_training_results" in st.session_state:
            show_training_results(st.session_state.last_training_results)
        else:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            empty_state(
                title="No Training Results Yet",
                message="Configure and run training to see results here.",
                icon="🎓"
            )
            st.markdown('</div>', unsafe_allow_html=True)


def render_evaluate_tab():
    """Render the model evaluation tab."""
    st.markdown('<p class="section-header">Model Evaluation</p>', unsafe_allow_html=True)
    
    col_model, col_data = st.columns([1, 1])
    
    with col_model:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Select Model**")
        
        # List available models
        models_dir = Path("models/trained")
        model_files = []
        if models_dir.exists():
            model_files = sorted([f.name for f in models_dir.glob("*.pkl")], reverse=True)
        
        if not model_files:
            info_banner("No trained models found. Train a model first.", type="info")
        else:
            selected_model = st.selectbox(
                "Choose a model",
                options=model_files,
                label_visibility="collapsed"
            )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_data:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Select Test Data**")
        
        # List available datasets
        data_dir = Path("data/synthetic")
        json_files = []
        if data_dir.exists():
            json_files = sorted(
                [f.name for f in data_dir.glob("*.json") if not f.name.endswith(".report.json")],
                reverse=True
            )
        
        selected_test_file = st.selectbox(
            "Choose test dataset",
            options=[""] + json_files,
            format_func=lambda x: "Select a file..." if x == "" else x,
            label_visibility="collapsed",
            key="eval_dataset"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    get_divider()
    
    # Evaluate button
    can_evaluate = model_files and selected_test_file
    
    if st.button("📊 Evaluate Model", use_container_width=True, type="primary", disabled=not can_evaluate):
        if BACKEND_AVAILABLE:
            run_evaluation(
                model_path=str(models_dir / selected_model),
                test_data_path=str(data_dir / selected_test_file)
            )
        else:
            # Demo evaluation
            st.session_state.last_evaluation_results = {
                "accuracy": 0.72,
                "f1_score": 0.68,
                "precision": 0.70,
                "recall": 0.66,
                "demo": True
            }
            st.rerun()
    
    # Show evaluation results
    if "last_evaluation_results" in st.session_state:
        show_evaluation_results(st.session_state.last_evaluation_results)


def render_history_tab():
    """Render the training history tab."""
    st.markdown('<p class="section-header">Training History</p>', unsafe_allow_html=True)
    
    if not REGISTRY_AVAILABLE:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        info_banner("Registry not available. Training history requires the registry module.", type="warning")
        st.markdown('</div>', unsafe_allow_html=True)
        return
    
    try:
        registry = DatasetRegistry()
        
        # Get all datasets with training runs
        datasets = registry.list_datasets()
        
        if not datasets:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            empty_state(
                title="No Training History",
                message="Train models on registered datasets to see history here.",
                icon="📜"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Collect all training runs
        all_runs = []
        for dataset in datasets:
            history = registry.get_training_history(dataset["id"])
            for run in history:
                run["dataset_name"] = dataset.get("name", dataset["id"])
                all_runs.append(run)
        
        if not all_runs:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            empty_state(
                title="No Training Runs Found",
                message="Training runs will appear here after you train models.",
                icon="📜"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # Sort by date
        all_runs.sort(key=lambda x: x.get("trained_at", ""), reverse=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            stat_card(
                value=str(len(all_runs)),
                label="Total Runs",
                icon="🔄"
            )
        
        with col2:
            # Best accuracy
            best_acc = max(
                (r.get("results", {}).get("accuracy", 0) for r in all_runs),
                default=0
            )
            stat_card(
                value=f"{best_acc*100:.1f}%",
                label="Best Accuracy",
                icon="🏆"
            )
        
        with col3:
            # Model types used
            model_types = set(r.get("model_type", "unknown") for r in all_runs)
            stat_card(
                value=str(len(model_types)),
                label="Model Types",
                icon="🤖"
            )
        
        with col4:
            # Datasets trained
            dataset_names = set(r.get("dataset_name", "unknown") for r in all_runs)
            stat_card(
                value=str(len(dataset_names)),
                label="Datasets",
                icon="📁"
            )
        
        get_divider()
        
        # Training runs table
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Recent Training Runs**")
        
        for run in all_runs[:10]:  # Show last 10
            results = run.get("results", {})
            accuracy = results.get("accuracy", 0)
            f1 = results.get("f1_score", 0)
            
            col_info, col_metrics = st.columns([2, 1])
            
            with col_info:
                st.markdown(f"""
                **{run.get('model_type', 'Unknown').replace('_', ' ').title()}** on `{run.get('dataset_name', 'Unknown')}`  
                <span style="color: #718096; font-size: 0.85rem;">{run.get('trained_at', 'Unknown date')}</span>
                """, unsafe_allow_html=True)
            
            with col_metrics:
                status = "success" if accuracy >= 0.7 else "warning" if accuracy >= 0.5 else "error"
                st.markdown(f"""
                <div style="text-align: right;">
                    <span style="color: {'#10b981' if status == 'success' else '#f59e0b' if status == 'warning' else '#ef4444'}; font-weight: 600;">
                        {accuracy*100:.1f}% acc
                    </span>
                    <span style="color: #718096;"> | </span>
                    <span style="color: #a0aec0;">{f1*100:.1f}% F1</span>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Error loading training history: {e}")


def run_training(
    conversations: list,
    model_type: str,
    use_cv: bool,
    cv_folds: int,
    max_features: int,
    ngram_range: tuple,
    test_size: float,
    save_model: bool,
    dataset_path: str = None
):
    """Execute the training process."""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    if BACKEND_AVAILABLE:
        try:
            status_text.text("Initializing classifier...")
            progress_bar.progress(10)
            
            # Create classifier
            classifier = IntentClassifier(
                model_type=model_type,
                max_features=max_features,
                ngram_range=ngram_range
            )
            
            if use_cv:
                status_text.text(f"Running {cv_folds}-fold cross-validation...")
                progress_bar.progress(30)
                
                # Use Trainer for CV
                trainer = Trainer()
                cv_result = trainer.cross_validate(
                    conversations=conversations,
                    k=cv_folds,
                    model_type=model_type
                )
                
                progress_bar.progress(90)
                
                results = {
                    "accuracy": cv_result.mean_accuracy,
                    "accuracy_std": cv_result.std_accuracy,
                    "f1_score": cv_result.mean_f1,
                    "f1_std": cv_result.std_f1,
                    "cv_folds": cv_folds,
                    "model_type": model_type,
                    "is_cv": True
                }
            else:
                status_text.text("Training classifier...")
                progress_bar.progress(30)
                
                # Regular training
                training_result = classifier.train(
                    conversations=conversations,
                    test_size=test_size
                )
                
                progress_bar.progress(70)
                
                if save_model:
                    status_text.text("Saving model...")
                    model_path = f"models/trained/intent_classifier.pkl"
                    classifier.save(model_path)
                
                progress_bar.progress(90)
                
                results = {
                    "accuracy": training_result.accuracy,
                    "f1_score": training_result.f1_score,
                    "precision": training_result.precision,
                    "recall": training_result.recall,
                    "train_samples": training_result.train_samples,
                    "test_samples": training_result.test_samples,
                    "unique_intents": training_result.unique_intents,
                    "model_type": model_type,
                    "is_cv": False,
                    "model_saved": save_model
                }
            
            progress_bar.progress(100)
            status_text.text("Training complete!")
            
            st.session_state.last_training_results = results
            
            # Register training run if registry available
            if REGISTRY_AVAILABLE and dataset_path:
                try:
                    registry = DatasetRegistry()
                    # Try to find dataset ID by path
                    datasets = registry.list_datasets()
                    for ds in datasets:
                        if ds.get("file_path") == dataset_path or dataset_path.endswith(ds.get("name", "")):
                            registry.register_training_run(
                                dataset_id=ds["id"],
                                model_type=model_type,
                                results=results
                            )
                            break
                except Exception:
                    pass  # Silent fail for registry
            
            st.rerun()
            
        except Exception as e:
            progress_bar.progress(100)
            status_text.text("Training failed")
            st.error(f"Error: {str(e)}")
    else:
        # Demo mode
        import time
        for i in range(5):
            progress_bar.progress((i + 1) * 20)
            status_text.text(f"Simulating training... Step {i + 1}/5")
            time.sleep(0.3)
        
        status_text.text("Demo training complete!")
        
        st.session_state.last_training_results = {
            "accuracy": 0.68,
            "f1_score": 0.62,
            "precision": 0.65,
            "recall": 0.60,
            "train_samples": int(len(conversations) * 0.8),
            "test_samples": int(len(conversations) * 0.2),
            "unique_intents": min(len(conversations), 77),
            "model_type": model_type,
            "is_cv": use_cv,
            "demo": True
        }
        
        if use_cv:
            st.session_state.last_training_results["accuracy_std"] = 0.05
            st.session_state.last_training_results["f1_std"] = 0.06
            st.session_state.last_training_results["cv_folds"] = cv_folds
        
        st.rerun()


def run_evaluation(model_path: str, test_data_path: str):
    """Run model evaluation on test data."""
    try:
        # Load model
        classifier = IntentClassifier.load(model_path)
        
        # Load test data
        conversations = load_conversations(test_data_path)
        
        # Evaluate
        result = classifier.evaluate(conversations)
        
        st.session_state.last_evaluation_results = {
            "accuracy": result.accuracy,
            "f1_score": result.f1_score,
            "precision": result.precision,
            "recall": result.recall,
            "test_samples": len(conversations)
        }
        
        st.rerun()
        
    except Exception as e:
        st.error(f"Evaluation error: {e}")


def show_training_results(results: dict):
    """Display training results."""
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    
    if results.get("demo"):
        info_banner("Demo results - connect backend for real training.", type="info")
    
    # Model info
    model_type = results.get("model_type", "unknown").replace("_", " ").title()
    is_cv = results.get("is_cv", False)
    
    st.markdown(f"**Model:** {model_type}")
    if is_cv:
        st.markdown(f"**Cross-Validation:** {results.get('cv_folds', 5)}-fold")
    
    st.markdown("---")
    
    # Metrics
    col1, col2 = st.columns(2)
    
    accuracy = results.get("accuracy", 0)
    f1 = results.get("f1_score", 0)
    
    with col1:
        if is_cv and "accuracy_std" in results:
            value = f"{accuracy*100:.1f}% ± {results['accuracy_std']*100:.1f}%"
        else:
            value = f"{accuracy*100:.1f}%"
        
        metric_card(
            title="Accuracy",
            value=value,
            subtitle="Test set performance",
            status="success" if accuracy >= 0.7 else "warning" if accuracy >= 0.5 else "error",
            icon="🎯"
        )
    
    with col2:
        if is_cv and "f1_std" in results:
            value = f"{f1*100:.1f}% ± {results['f1_std']*100:.1f}%"
        else:
            value = f"{f1*100:.1f}%"
        
        metric_card(
            title="F1 Score",
            value=value,
            subtitle="Macro average",
            status="success" if f1 >= 0.65 else "warning" if f1 >= 0.45 else "error",
            icon="📊"
        )
    
    if not is_cv:
        col3, col4 = st.columns(2)
        
        with col3:
            precision = results.get("precision", 0)
            metric_card(
                title="Precision",
                value=f"{precision*100:.1f}%",
                subtitle="Positive predictions",
                status="neutral",
                icon="✓"
            )
        
        with col4:
            recall = results.get("recall", 0)
            metric_card(
                title="Recall",
                value=f"{recall*100:.1f}%",
                subtitle="True positive rate",
                status="neutral",
                icon="🔍"
            )
    
    st.markdown("---")
    
    # Dataset info
    if "train_samples" in results:
        info_text = f"Trained on {results['train_samples']} samples, tested on {results['test_samples']} samples"
        if "unique_intents" in results:
            info_text += f" ({results['unique_intents']} intents)"
        st.caption(info_text)
    
    if results.get("model_saved"):
        st.success("Model saved to `models/trained/intent_classifier.pkl`")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Accuracy gauge
    st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
    fig = quality_gauge(
        accuracy * 100,
        title="Model Accuracy",
        height=250
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)


def show_evaluation_results(results: dict):
    """Display evaluation results."""
    st.markdown('<p class="section-header" style="margin-top: 2rem;">Evaluation Results</p>', unsafe_allow_html=True)
    
    if results.get("demo"):
        info_banner("Demo results - connect backend for real evaluation.", type="info")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        stat_card(
            value=f"{results.get('accuracy', 0)*100:.1f}%",
            label="Accuracy",
            icon="🎯"
        )
    
    with col2:
        stat_card(
            value=f"{results.get('f1_score', 0)*100:.1f}%",
            label="F1 Score",
            icon="📊"
        )
    
    with col3:
        stat_card(
            value=f"{results.get('precision', 0)*100:.1f}%",
            label="Precision",
            icon="✓"
        )
    
    with col4:
        stat_card(
            value=f"{results.get('recall', 0)*100:.1f}%",
            label="Recall",
            icon="🔍"
        )


def _get_preset_model_type(preset_name: str) -> str:
    """Get the model type for a given preset."""
    preset_models = {
        "fast": "logistic_regression",
        "balanced": "random_forest",
        "best": "xgboost",
        "quick_test": "logistic_regression",
        "cross_validation": "random_forest"
    }
    return preset_models.get(preset_name, "logistic_regression")
