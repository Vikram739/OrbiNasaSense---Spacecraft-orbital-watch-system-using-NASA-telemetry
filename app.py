"""
Streamlit UI for NASA Spacecraft Telemetry Anomaly Detection.

Run with: streamlit run app.py
"""
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import load_npy_channel, get_available_channels
from inference import load_model_and_config, run_inference


# Page configuration
st.set_page_config(
    page_title="OrbiNasaSense - Spacecraft Anomaly Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def load_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.2rem;
            color: #666;
            text-align: center;
            margin-bottom: 2rem;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'channel_name' not in st.session_state:
        st.session_state.channel_name = None
    if 'inference_results' not in st.session_state:
        st.session_state.inference_results = None
    if 'model_config' not in st.session_state:
        st.session_state.model_config = None


def sidebar_controls():
    """Render sidebar controls."""
    st.sidebar.title("⚙️ Controls")
    
    # Model selection
    st.sidebar.subheader("1. Select Model")
    
    models_dir = 'models'
    available_models = []
    
    if os.path.exists(models_dir):
        config_files = [f for f in os.listdir(models_dir) if f.startswith('threshold_') and f.endswith('.json')]
        available_models = [f.replace('threshold_', '').replace('.json', '') for f in config_files]
    
    if available_models:
        selected_channel = st.sidebar.selectbox(
            "Choose trained model:",
            available_models,
            help="Select a pre-trained model for a specific channel"
        )
    else:
        st.sidebar.warning("No trained models found. Please train a model first using train.py")
        selected_channel = None
    
    st.sidebar.markdown("---")
    
    # Data loading options
    st.sidebar.subheader("2. Load Data")
    
    data_source = st.sidebar.radio(
        "Data source:",
        ["Sample from dataset", "Upload custom .npy file"],
        help="Choose how to load telemetry data"
    )
    
    loaded_data = None
    channel_name = None
    
    if data_source == "Sample from dataset":
        test_dir = os.path.join('data', 'test')
        
        if os.path.exists(test_dir):
            test_files = [f for f in os.listdir(test_dir) if f.endswith('.npy')]
            
            if test_files:
                selected_file = st.sidebar.selectbox("Select test file:", test_files)
                
                if st.sidebar.button("Load Sample Data", type="primary"):
                    file_path = os.path.join(test_dir, selected_file)
                    loaded_data = load_npy_channel(file_path)
                    channel_name = selected_file.replace('_test.npy', '').replace('_train.npy', '').replace('.npy', '')
                    st.sidebar.success(f"Loaded {selected_file}")
            else:
                st.sidebar.info("No test files found in data/test/")
        else:
            st.sidebar.info("data/test/ directory not found")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload .npy file:", type=['npy'])
        
        if uploaded_file is not None:
            loaded_data = np.load(uploaded_file)
            channel_name = uploaded_file.name.replace('.npy', '')
            st.sidebar.success(f"Uploaded {uploaded_file.name}")
    
    if loaded_data is not None:
        if len(loaded_data.shape) > 1:
            loaded_data = loaded_data.flatten()
        st.session_state.data = loaded_data
        st.session_state.channel_name = channel_name
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Anomaly Detection Parameters")
    
    threshold_k = st.sidebar.slider(
        "Threshold Multiplier (k)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Anomaly threshold = mean_error + k * std_error"
    )
    
    return selected_channel, threshold_k


def plot_telemetry_data(data, title="Raw Telemetry Data"):
    """Plot raw telemetry data."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Telemetry', line=dict(color='#1f77b4', width=1)))
    fig.update_layout(title=title, xaxis_title="Time Step", yaxis_title="Value", hovermode='x unified', height=400)
    return fig


def plot_anomaly_detection(data, predictions, anomaly_mask, errors, threshold):
    """Plot anomaly detection results."""
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Telemetry Data with Detected Anomalies', 'Predictions vs Actual', 'Prediction Errors with Threshold'),
        vertical_spacing=0.1,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Plot 1: Raw data with anomalies
    fig.add_trace(go.Scatter(y=data, mode='lines', name='Telemetry', line=dict(color='#1f77b4', width=1.5)), row=1, col=1)
    anomaly_indices = np.where(anomaly_mask)[0]
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(x=anomaly_indices, y=data[anomaly_indices], mode='markers', name='Anomalies', marker=dict(color='red', size=8, symbol='x')), row=1, col=1)
    
    # Plot 2: Predictions vs actual
    valid_indices = ~np.isnan(predictions)
    fig.add_trace(go.Scatter(x=np.where(valid_indices)[0], y=data[valid_indices], mode='lines', name='Actual', line=dict(color='#1f77b4', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=np.where(valid_indices)[0], y=predictions[valid_indices], mode='lines', name='Predicted', line=dict(color='orange', width=1, dash='dot')), row=2, col=1)
    
    # Plot 3: Errors
    valid_errors = ~np.isnan(errors)
    fig.add_trace(go.Scatter(x=np.where(valid_errors)[0], y=errors[valid_errors], mode='lines', name='Error', line=dict(color='green', width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=[0, len(data)-1], y=[threshold, threshold], mode='lines', name='Threshold', line=dict(color='red', width=2, dash='dash')), row=3, col=1)
    
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_yaxes(title_text="Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Error", row=3, col=1)
    fig.update_layout(height=1000, hovermode='x unified', showlegend=True)
    
    return fig


def main():
    """Main application function."""
    load_custom_css()
    initialize_session_state()
    
    st.markdown('<div class="main-header">🛰️ OrbiNasaSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NASA Spacecraft Telemetry Anomaly Detection with LSTM</div>', unsafe_allow_html=True)
    
    selected_channel, threshold_k = sidebar_controls()
    
    st.markdown("---")
    
    if st.session_state.data is None:
        st.info("👈 Please load telemetry data from the sidebar to begin analysis.")
        st.markdown("""
        ### 📋 Getting Started
        1. **Select a trained model** from the dropdown
        2. **Load data** from sample dataset or upload a .npy file
        3. **Adjust parameters** for anomaly detection
        4. **View results** including visualizations and metrics
        
        ### 🔧 Training a Model
        ```bash
        python train.py --channel P-1 --window-size 50 --epochs 50
        ```
        """)
        return
    
    st.success(f"✅ Data loaded: {st.session_state.channel_name} ({len(st.session_state.data)} time steps)")
    
    st.subheader("📈 Raw Telemetry Data")
    fig_raw = plot_telemetry_data(st.session_state.data, f"Channel: {st.session_state.channel_name}")
    st.plotly_chart(fig_raw, width='stretch')
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Points", f"{len(st.session_state.data):,}")
    with col2:
        st.metric("Mean", f"{np.mean(st.session_state.data):.4f}")
    with col3:
        st.metric("Std Dev", f"{np.std(st.session_state.data):.4f}")
    with col4:
        st.metric("Range", f"{np.ptp(st.session_state.data):.4f}")
    
    st.markdown("---")
    
    if selected_channel:
        st.subheader("🔍 Anomaly Detection")
        
        if st.button("🚀 Run Anomaly Detection", type="primary"):
            with st.spinner("Running inference..."):
                try:
                    model, config = load_model_and_config(selected_channel)
                    st.session_state.model_config = config
                    results = run_inference(st.session_state.data, model, config, custom_threshold_k=threshold_k)
                    st.session_state.inference_results = results
                    st.success("✅ Inference completed!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    return
        
        if st.session_state.inference_results is not None:
            results = st.session_state.inference_results
            
            st.markdown("---")
            st.subheader("📊 Detection Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Anomalies Detected", f"{results['num_anomalies']:,}")
            with col2:
                st.metric("Anomaly Rate", f"{results['anomaly_percentage']:.2f}%")
            with col3:
                st.metric("Threshold", f"{results['threshold']:.6f}")
            with col4:
                st.metric("Window Size", f"{results['window_size']}")
            
            st.subheader("📉 Detailed Analysis")
            fig_anomaly = plot_anomaly_detection(st.session_state.data, results['predictions'], results['anomaly_mask'], results['errors'], results['threshold'])
            st.plotly_chart(fig_anomaly, width='stretch')
            
            if results['num_anomalies'] > 0:
                with st.expander("🔎 View Anomaly Details"):
                    anomaly_indices = results['anomaly_indices']
                    df_anomalies = pd.DataFrame({
                        'Time Step': anomaly_indices,
                        'Actual Value': st.session_state.data[anomaly_indices],
                        'Predicted Value': results['predictions'][anomaly_indices],
                        'Error': results['errors'][anomaly_indices]
                    })
                    st.dataframe(df_anomalies, width='stretch')
                    csv = df_anomalies.to_csv(index=False)
                    st.download_button("📥 Download CSV", csv, f"anomalies_{st.session_state.channel_name}.csv", "text/csv")
            else:
                st.info("No anomalies detected.")
    else:
        st.warning("⚠️ No trained models available. Train a model first using train.py")
    
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: #666; padding: 2rem;">
            <p>OrbiNasaSense - Spacecraft Orbital Watch System</p>
            <p>Built with Streamlit, TensorFlow, and Plotly</p>
        </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
