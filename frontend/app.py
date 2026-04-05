"""
File Type Identification - Model Comparison Dashboard
Main Streamlit application
"""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from utils import clean_file_data, create_fragments
from models import load_models, predict_file
from visualizations import (
    plot_comparison_bars, plot_radar_comparison, plot_training_history,
    plot_accuracy_history, plot_confusion_matrix, plot_per_class_metrics,
    display_model_metrics, plot_confidence_gauge, plot_confidence_comparison,
    plot_prediction_pie
)

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="File Type Identification - Model Comparison Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0; }
    .section-header { border-bottom: 3px solid #1f77b4; padding-bottom: 10px; margin-top: 20px; }
    .model-name { font-size: 24px; font-weight: bold; color: #1f77b4; }
    </style>
    """, unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


def load_all_results():
    """Load all model results from JSON files"""
    models_data = {}
    json_files = sorted(RESULTS_DIR.glob("*_results.json"))
    
    for json_file in json_files:
        try:
            model_key = json_file.stem.replace("_results", "")
            data = json.loads(json_file.read_text(encoding="utf-8"))
            models_data[model_key] = data
        except Exception as e:
            st.warning(f"Could not load {json_file.name}: {e}")
    
    return models_data


def create_comparison_dataframe(models_data):
    """Create comparison DataFrame for all models"""
    rows = []
    
    for model_key, data in models_data.items():
        rows.append({
            "Model": model_key.replace("_", " ").upper(),
            "Accuracy": data.get("accuracy", 0),
            "Precision": data.get("precision", 0),
            "Recall": data.get("recall", 0),
            "F1 Score": data.get("f1_score", 0),
            "Val Accuracy": data.get("val_accuracy", 0),
            "Val F1 Score": data.get("val_f1_score", 0),
        })
    
    return pd.DataFrame(rows)


# Title and description
st.title("🎯 File Type Identification - Model Comparison Dashboard")
st.markdown("Compare performance metrics, visualizations, and training histories of different ML models")

# Sidebar
st.sidebar.header("📊 Dashboard Controls")
show_section = st.sidebar.radio(
    "Select Section:",
    ["📈 Model Comparison", "🔍 Individual Model Analysis", "📁 File Upload & Predict"],
    index=0
)

# Load all model data
models_data = load_all_results()

if not models_data:
    st.error("No model results found in the results directory!")
    st.stop()


# ==================== SECTION 1: MODEL COMPARISON ====================
if show_section == "📈 Model Comparison":
    st.markdown("### 📊 Overall Model Performance Comparison")
    
    df_comparison = create_comparison_dataframe(models_data)
    
    st.dataframe(
        df_comparison.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1 Score": "{:.2%}",
            "Val Accuracy": "{:.2%}",
            "Val F1 Score": "{:.2%}"
        }).highlight_max(subset=["Accuracy", "F1 Score"], color='yellow'),
        use_container_width=True
    )
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Bar Charts", "🎯 Radar Chart", "🏆 Best Models", "📈 Rankings"])
    
    with tab1:
        st.plotly_chart(plot_comparison_bars(df_comparison), use_container_width=True, key='comparison_bars')
    
    with tab2:
        st.plotly_chart(plot_radar_comparison(df_comparison), use_container_width=True, key='radar_comparison')
    
    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            best_accuracy = df_comparison.loc[df_comparison["Accuracy"].idxmax()]
            st.markdown("#### 🥇 Best Accuracy")
            st.success(f"**{best_accuracy['Model']}**: {best_accuracy['Accuracy']*100:.2f}%")
        
        with col2:
            best_f1 = df_comparison.loc[df_comparison["F1 Score"].idxmax()]
            st.markdown("#### 🥇 Best F1 Score")
            st.success(f"**{best_f1['Model']}**: {best_f1['F1 Score']*100:.2f}%")
    
    with tab4:
        st.markdown("#### Model Rankings by Metric")
        for metric in ["Accuracy", "F1 Score", "Precision", "Recall"]:
            ranking_df = df_comparison[["Model", metric]].sort_values(metric, ascending=False).reset_index(drop=True)
            ranking_df.index = ranking_df.index + 1
            st.markdown(f"**{metric}**")
            st.dataframe(ranking_df.style.format({metric: "{:.2%}"}))


# ==================== SECTION 2: INDIVIDUAL MODEL ANALYSIS ====================
elif show_section == "🔍 Individual Model Analysis":
    st.markdown("### 🔍 Detailed Model Analysis")
    
    selected_model = st.selectbox(
        "Select a Model to Analyze:",
        sorted(models_data.keys()),
        format_func=lambda x: x.replace("_", " ").upper()
    )
    
    model_data = models_data[selected_model]
    model_display_name = selected_model.replace("_", " ").upper()
    
    st.markdown(f"### {model_display_name}")
    
    st.markdown("#### 📊 Performance Metrics")
    display_model_metrics(model_data, model_display_name)
    
    if "parameters" in model_data:
        st.markdown("#### 🔧 Model Parameters")
        params = model_data["parameters"]
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Parameters", f"{params.get('total_params', 0):,}")
        with col2:
            st.metric("Trainable Parameters", f"{params.get('trainable_params', 0):,}")
        with col3:
            st.metric("Model Size", f"{params.get('model_size_mb', 0):.2f} MB")
    
    st.markdown("#### 📈 Training History")
    col1, col2 = st.columns(2)
    
    with col1:
        fig_loss = plot_training_history(model_data, model_display_name)
        if fig_loss:
            st.plotly_chart(fig_loss, use_container_width=True, key='training_loss')
    
    with col2:
        fig_acc = plot_accuracy_history(model_data, model_display_name)
        if fig_acc:
            st.plotly_chart(fig_acc, use_container_width=True, key='accuracy_history')
    
    st.markdown("#### 🔲 Confusion Matrix Heatmap")
    fig_cm = plot_confusion_matrix(model_data, model_display_name)
    if fig_cm:
        st.plotly_chart(fig_cm, use_container_width=True, key='confusion_matrix')
    
    st.markdown("#### 📊 Per-Class Performance Metrics")
    fig_per_class = plot_per_class_metrics(model_data, model_display_name)
    if fig_per_class:
        st.plotly_chart(fig_per_class, use_container_width=True, key='per_class_metrics')
    
    if "per_class_metrics" in model_data:
        st.markdown("#### 📋 Detailed Per-Class Metrics Table")
        per_class_df = pd.DataFrame([
            {
                "File Type": cls,
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1 Score": metrics.get("f1", 0),
                "Support": metrics.get("support", 0)
            }
            for cls, metrics in model_data["per_class_metrics"].items()
        ]).sort_values("F1 Score", ascending=False)
        
        st.dataframe(
            per_class_df.style.format({
                "Precision": "{:.2%}",
                "Recall": "{:.2%}",
                "F1 Score": "{:.2%}"
            }).highlight_max(subset=["F1 Score"], color='lightgreen'),
            use_container_width=True
        )


# ==================== SECTION 3: FILE UPLOAD & PREDICT ====================
elif show_section == "📁 File Upload & Predict":
    st.markdown("### 📁 File Upload & Prediction")
    
    with st.spinner("🔄 Loading models..."):
        loaded_models, class_labels = load_models()
    
    if not loaded_models:
        st.warning("⚠️ No models could be loaded. Please ensure models are saved in `saved_models/` directory.")
    
    st.info("📤 Choose an upload option and upload your file for type prediction using 8 different ML models.")
    
    # Two upload options
    upload_option = st.radio(
        "📋 Select Upload Option:",
        ["📦 Upload Binary (.bin) File", "📄 Upload Direct File"],
        horizontal=True,
        key="upload_option"
    )
    
    uploaded_file = None
    
    if upload_option == "📦 Upload Binary (.bin) File":
        st.markdown("#### 📦 Binary File Upload")
        st.markdown("Upload a **.bin** file (raw binary data)")
        uploaded_file = st.file_uploader("📁 Choose a .bin file:", type=["bin"], key="bin_uploader")
    
    else:  # Direct File
        st.markdown("#### 📄 Direct File Upload")
        st.markdown("Upload any file type (document, image, archive, etc.)")
        uploaded_file = st.file_uploader("📁 Choose a file:", type=None, key="direct_uploader")
    
    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        
        st.success(f"✅ File uploaded: **{uploaded_file.name}**")
        
        st.markdown("### 🔧 File Analysis & Fragmentation")
        
        # Display file info
        file_ext = uploaded_file.name.split('.')[-1].upper() if '.' in uploaded_file.name else "UNKNOWN"
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 File Name", uploaded_file.name.split('/')[-1][:30])
        with col2:
            st.metric("💾 File Size", f"{len(file_bytes) / 1024:.2f} KB")
        with col3:
            st.metric("🏷️ Extension", file_ext)
        with col4:
            st.metric("📊 Total Bytes", f"{len(file_bytes):,}")
        
        st.markdown("---")
        
        # File fragmentation analysis
        st.markdown("#### 🔨 Fragment Analysis")
        
        with st.spinner("📦 Creating file fragments..."):
            fragments = create_fragments(file_bytes, chunk_size=4096, num_fragments=5)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📦 Total Fragments", len(fragments))
        with col2:
            st.metric("📏 Fragment Size", "4096 bytes")
        with col3:
            st.metric("🔄 Sampling Method", "Start/Mid/End")
        
        # Show fragment details
        with st.expander("📋 Fragment Details", expanded=False):
            for i, fragment in enumerate(fragments):
                st.markdown(f"**Fragment {i+1}:**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Size", f"{len(fragment)} bytes")
                with col2:
                    st.metric("Non-Zero Bytes", np.count_nonzero(fragment))
                with col3:
                    entropy = -np.sum((np.bincount(fragment.astype(int), minlength=256) / len(fragment))[
                        np.nonzero(np.bincount(fragment.astype(int), minlength=256) / len(fragment))
                    ] * np.log2((np.bincount(fragment.astype(int), minlength=256) / len(fragment))[
                        np.nonzero(np.bincount(fragment.astype(int), minlength=256) / len(fragment))
                    ]))
                    st.metric("Entropy", f"{entropy:.2f}")
                st.divider()
        
        st.markdown("---")
        
        # Additional preprocessing only for direct files (not binary)
        if upload_option == "📄 Upload Direct File":
            st.markdown("### 🧹 File Preprocessing")
            
            with st.spinner("🔍 Detecting file type and cleaning headers/footers..."):
                cleaned_data, detection, clean_stats = clean_file_data(file_bytes, uploaded_file.name)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📄 Original Size", f"{clean_stats['original_size']:,} bytes")
            with col2:
                st.metric("🧹 Cleaned Size", f"{clean_stats['cleaned_size']:,} bytes")
            with col3:
                st.metric("🗑️ Bytes Removed", f"{clean_stats['bytes_removed']:,} bytes")
            with col4:
                st.metric("📊 Removal %", f"{clean_stats['removal_percentage']:.1f}%")
            
            with st.expander("📋 Preprocessing Details", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**Type Detection:**")
                    if detection['detected_type'] != 'Unknown':
                        st.success(f"✅ Detected as: **{detection['detected_type']}**")
                    else:
                        st.info("ℹ️ Type: Unknown (will use raw bytes)")
                
                with col2:
                    st.markdown("**Header Information:**")
                    if detection['has_header']:
                        st.success(f"✅ Header found & removed: **{detection['header_len']} bytes**")
                    else:
                        st.info("ℹ️ No recognizable header")
                
                with col3:
                    st.markdown("**Footer Information:**")
                    if detection['has_footer']:
                        st.success(f"✅ Footer found & removed: **{detection['footer_len']} bytes**")
                    else:
                        st.info("ℹ️ No recognizable footer")
        
        else:
            # For binary files, use raw data
            cleaned_data = file_bytes
            detection = {'detected_type': '.BIN (Raw Binary)'}
            clean_stats = {
                'original_size': len(file_bytes),
                'cleaned_size': len(file_bytes),
                'bytes_removed': 0,
                'removal_percentage': 0.0
            }
        
        st.markdown("---")
        
        if loaded_models and class_labels:
            with st.expander("📦 Model Loading Status", expanded=False):
                model_status = []
                for model_name in loaded_models.keys():
                    model_status.append({
                        'Model': model_name.upper(),
                        'Status': '✅ Loaded',
                        'Type': loaded_models[model_name]['type']
                    })
                st.dataframe(pd.DataFrame(model_status), use_container_width=True, hide_index=True)
            
            st.markdown("### 🔮 Predictions from All Models")
            
            st.warning("⚠️ **Model Accuracy Notice:** These models were trained on the available dataset and achieve 40-60% accuracy.")
            
            with st.spinner("🤖 Running predictions on cleaned file fragments..."):
                predictions = predict_file(file_bytes, loaded_models, class_labels, cleaned_data=cleaned_data)
            
            if predictions:
                # Predictions summary
                prediction_data = []
                for model_name, pred_info in predictions.items():
                    prediction_data.append({
                        'Model': model_name.replace('_', ' ').upper(),
                        'Predicted File Type': pred_info['predicted_class'].upper(),
                        'Confidence': f"{pred_info['confidence']*100:.2f}%",
                        'Confidence Score': pred_info['confidence']
                    })
                
                df_predictions = pd.DataFrame(prediction_data)
                
                st.markdown("#### 📊 Model Predictions Summary")
                st.dataframe(
                    df_predictions.style.format({}).highlight_max(subset=['Confidence Score'], color='lightgreen'),
                    use_container_width=True,
                    hide_index=True
                )
                
                # Debug section
                with st.expander("🔍 Debug: Full Prediction Probabilities (Top 5 Classes)", expanded=False):
                    for model_name, pred_info in predictions.items():
                        st.markdown(f"**{model_name.upper()}** - Top 5 Predictions:")
                        
                        top_5_indices = np.argsort(pred_info['probabilities'])[-5:][::-1]
                        
                        top_5_data = []
                        for rank, idx in enumerate(top_5_indices, 1):
                            if idx < len(class_labels):
                                top_5_data.append({
                                    'Rank': rank,
                                    'File Type': class_labels[idx].upper(),
                                    'Probability': f"{pred_info['probabilities'][idx]*100:.4f}%",
                                    'Score': pred_info['probabilities'][idx]
                                })
                        
                        if top_5_data:
                            df_top5 = pd.DataFrame(top_5_data)
                            st.dataframe(df_top5.style.format({'Score': '{:.6f}'}), use_container_width=True, hide_index=True)
                        st.divider()
                
                st.markdown("---")
                
                # Best prediction
                best_pred = max(predictions.items(), key=lambda x: x[1]['confidence'])
                
                st.markdown("### 🎯 Most Confident Prediction")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Model:** {best_pred[0].replace('_', ' ').upper()}\n\n**Predicted Type:** `{best_pred[1]['predicted_class'].upper()}`\n\n**Confidence:** {best_pred[1]['confidence']*100:.2f}%")
                    
                    if best_pred[1]['confidence'] > 0.8:
                        st.success("✅ High confidence prediction!")
                    elif best_pred[1]['confidence'] > 0.6:
                        st.info("ℹ️ Moderate confidence prediction")
                    else:
                        st.warning("⚠️ Low confidence - consider results with caution")
                
                with col2:
                    st.plotly_chart(plot_confidence_gauge(best_pred[1]['confidence']), use_container_width=False, key='confidence_gauge')
                
                st.markdown("---")
                
                st.markdown("#### 📈 Confidence Comparison Across Models")
                st.plotly_chart(plot_confidence_comparison(df_predictions), use_container_width=True, key='confidence_comparison')
                
                # Detailed predictions
                st.markdown("#### 🔬 Detailed Prediction Results")
                
                tabs = st.tabs([model_name.upper() for model_name in predictions.keys()])
                
                for idx, (model_name, pred_info) in enumerate(predictions.items()):
                    with tabs[idx]:
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            st.markdown(f"**Predicted File Type:** `{pred_info['predicted_class'].upper()}`")
                            st.markdown(f"**Confidence Score:** {pred_info['confidence']:.4f} ({pred_info['confidence']*100:.2f}%)")
                            
                            top_indices = np.argsort(pred_info['probabilities'])[-5:][::-1]
                            
                            st.markdown("**Top 5 Predictions:**")
                            top_pred_data = []
                            for rank, idx in enumerate(top_indices, 1):
                                if idx < len(class_labels):
                                    top_pred_data.append({
                                        'Rank': rank,
                                        'File Type': class_labels[idx].upper(),
                                        'Probability': f"{pred_info['probabilities'][idx]*100:.2f}%",
                                        'Score': pred_info['probabilities'][idx]
                                    })
                            
                            if top_pred_data:
                                df_top = pd.DataFrame(top_pred_data)
                                st.dataframe(df_top.style.format({}), use_container_width=True, hide_index=True)
                        
                        with col2:
                            if top_pred_data:
                                labels_top = [row['File Type'] for row in top_pred_data]
                                scores_top = [row['Score'] for row in top_pred_data]
                                st.plotly_chart(plot_prediction_pie(labels_top, scores_top), use_container_width=False, key=f"pie_{model_name}")
                
                st.markdown("---")
                
                # Voting summary
                st.markdown("#### 🗳️ Model Voting Summary")
                
                predictions_list = [pred_info['predicted_class'] for pred_info in predictions.values()]
                from collections import Counter
                vote_counts = Counter(predictions_list)
                
                consensus_label = vote_counts.most_common(1)[0][0]
                consensus_votes = vote_counts.most_common(1)[0][1]
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("🗳️ Consensus Prediction", consensus_label.upper())
                with col2:
                    st.metric("✅ Models in Agreement", f"{consensus_votes}/{len(predictions)}")
                with col3:
                    agreement_pct = (consensus_votes / len(predictions)) * 100
                    st.metric("📊 Agreement %", f"{agreement_pct:.1f}%")
                
                st.markdown("**All Model Predictions:**")
                
                predictions_display = []
                for model_name, pred_info in predictions.items():
                    predictions_display.append({
                        'Model': model_name.replace('_', ' ').upper(),
                        'Prediction': pred_info['predicted_class'].upper(),
                        'Confidence': f"{pred_info['confidence']*100:.1f}%"
                    })
                
                st.dataframe(pd.DataFrame(predictions_display), use_container_width=True, hide_index=True)
            
            else:
                st.error("❌ Could not generate predictions. Please check if file is valid.")
        
        else:
            st.error("❌ Models not properly loaded. Cannot make predictions.")
    
    else:
        st.info("👆 Upload a file above to get started with predictions!")
        
        st.markdown("#### 📦 Available Models for Prediction")
        model_info = []
        for model_key in sorted(models_data.keys()):
            model_data = models_data[model_key]
            accuracy = model_data.get("accuracy", 0)
            f1 = model_data.get("f1_score", 0)
            model_info.append({
                'Model': model_key.upper(),
                'Accuracy': f"{accuracy*100:.2f}%",
                'F1 Score': f"{f1*100:.2f}%"
            })
        
        st.dataframe(pd.DataFrame(model_info), use_container_width=True, hide_index=True)


st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px; margin-top: 30px;'>
    📊 File Type Identification - Model Comparison Dashboard | Built with Streamlit & Plotly
    </div>
    """,
    unsafe_allow_html=True
)
