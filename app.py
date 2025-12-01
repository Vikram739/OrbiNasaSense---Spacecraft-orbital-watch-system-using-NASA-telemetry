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
from datetime import datetime

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from data_utils import load_npy_channel, get_available_channels
from inference import load_model_and_config, run_inference
from auto_model_selector import get_model_recommendation


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
    
    # Model selection with descriptions
    st.sidebar.subheader("1. Select Detection Model")
    
    models_dir = 'models'
    available_models = []
    
    if os.path.exists(models_dir):
        config_files = [f for f in os.listdir(models_dir) if f.startswith('threshold_') and f.endswith('.json')]
        available_models = [f.replace('threshold_', '').replace('.json', '') for f in config_files]
    
    # Filter available models
    available_options = ['AUTO']
    for model in ['P-1', 'M-1', 'E-1']:
        if model in available_models:
            available_options.append(model)
    
    if len(available_options) == 1:  # Only AUTO available
        st.sidebar.warning("No trained models found. Please train models first using train.py")
        selected_option = None
    else:
        # Simple dropdown for model selection
        selected_option = st.sidebar.selectbox(
            "Select Model:",
            available_options,
            index=0
        )
    
    st.sidebar.markdown("---")
    
    # Data loading options
    st.sidebar.subheader("2. Load Data")
    
    data_source = st.sidebar.radio(
        "Data source:",
        ["Sample from dataset", "Upload custom file"],
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
                    channel_name = selected_file.replace('.npy', '')
                    st.sidebar.success(f"Loaded {selected_file}")
            else:
                st.sidebar.info("No test files found in data/test/")
        else:
            st.sidebar.info("data/test/ directory not found")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload telemetry file:", 
            type=['npy', 'csv', 'txt'],
            help="Supports NPY, CSV, and TXT formats"
        )
        
        if uploaded_file is not None:
            file_ext = uploaded_file.name.split('.')[-1].lower()
            
            try:
                if file_ext == 'npy':
                    loaded_data = np.load(uploaded_file)
                elif file_ext == 'csv':
                    df = pd.read_csv(uploaded_file)
                    # Use first column or 'value' column if exists
                    if 'value' in df.columns:
                        loaded_data = df['value'].values
                    else:
                        loaded_data = df.iloc[:, 0].values
                elif file_ext == 'txt':
                    loaded_data = np.loadtxt(uploaded_file)
                
                channel_name = uploaded_file.name.rsplit('.', 1)[0]
                st.sidebar.success(f"✅ Uploaded {uploaded_file.name}")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading file: {str(e)}")
                st.sidebar.info("Expected format: Single column of numeric values")
    
    if loaded_data is not None:
        if len(loaded_data.shape) > 1:
            loaded_data = loaded_data.flatten()
        st.session_state.data = loaded_data
        st.session_state.channel_name = channel_name
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Detection Parameters")
    
    threshold_k = st.sidebar.slider(
        "Threshold Multiplier (k)",
        min_value=1.0,
        max_value=5.0,
        value=3.0,
        step=0.1,
        help="Anomaly threshold = mean_error + k * std_error"
    )
    
    speed_mode = st.sidebar.radio(
        "Processing Speed:",
        ["Fast (downsample)", "Normal"],
        index=1,
        help="Fast mode processes every 2nd point for 2x speed"
    )
    
    downsample_factor = 2 if speed_mode == "Fast (downsample)" else 1
    
    return selected_option, threshold_k, downsample_factor


def plot_telemetry_data(data, anomaly_indices=None, title="Telemetry Data"):
    """Plot telemetry data."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Telemetry',
        line=dict(color='#1f77b4', width=1.5)
    ))
    
    if anomaly_indices is not None and len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=data[anomaly_indices],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time Step",
        yaxis_title="Value",
        hovermode='x unified',
        height=400
    )
    
    return fig


def plot_anomaly_detection(data, predictions, anomaly_mask, errors, threshold, anomaly_indices=None):
    """Plot comprehensive anomaly detection results."""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=(
            '📊 Telemetry Data with Detected Anomalies',
            '🔄 Actual vs Predicted Values',
            '⚠️ Prediction Errors & Threshold'
        ),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.3, 0.3]
    )
    
    # Plot 1: Telemetry with anomalies
    fig.add_trace(go.Scatter(
        y=data,
        mode='lines',
        name='Telemetry Data',
        line=dict(color='#1f77b4', width=2),
        showlegend=True
    ), row=1, col=1)
    
    # Use provided anomaly_indices or extract from mask
    if anomaly_indices is None:
        # Ensure mask is same length as data
        if len(anomaly_mask) != len(data):
            # Pad mask to match data length
            padded_mask = np.zeros(len(data), dtype=bool)
            padded_mask[:len(anomaly_mask)] = anomaly_mask
            anomaly_mask = padded_mask
        anomaly_indices = np.where(anomaly_mask)[0]
    
    if len(anomaly_indices) > 0:
        fig.add_trace(go.Scatter(
            x=anomaly_indices,
            y=data[anomaly_indices],
            mode='markers',
            name='Anomalies Detected',
            marker=dict(color='red', size=10, symbol='x', line=dict(width=2)),
            showlegend=True
        ), row=1, col=1)
    
    # Plot 2: Actual vs Predicted
    valid_indices = ~np.isnan(predictions)
    fig.add_trace(go.Scatter(
        x=np.where(valid_indices)[0],
        y=data[valid_indices],
        mode='lines',
        name='Actual Values',
        line=dict(color='#1f77b4', width=2),
        showlegend=True
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=np.where(valid_indices)[0],
        y=predictions[valid_indices],
        mode='lines',
        name='Predicted Values',
        line=dict(color='orange', width=2, dash='dot'),
        showlegend=True
    ), row=2, col=1)
    
    # Plot 3: Errors with threshold
    valid_errors = ~np.isnan(errors)
    fig.add_trace(go.Scatter(
        x=np.where(valid_errors)[0],
        y=errors[valid_errors],
        mode='lines',
        name='Prediction Error',
        line=dict(color='green', width=1.5),
        showlegend=True
    ), row=3, col=1)
    
    fig.add_trace(go.Scatter(
        x=[0, len(data)-1],
        y=[threshold, threshold],
        mode='lines',
        name='Anomaly Threshold',
        line=dict(color='red', width=2, dash='dash'),
        showlegend=True
    ), row=3, col=1)
    
    # Update axes
    fig.update_xaxes(title_text="Time Step", row=3, col=1)
    fig.update_yaxes(title_text="Telemetry Value", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Error Magnitude", row=3, col=1)
    
    fig.update_layout(
        height=1000,
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig


def estimate_lifespan(data, anomaly_indices, current_time=0):
    """
    Estimate remaining lifespan based on anomaly trend.
    
    Returns:
        dict: Lifespan estimation with status and days remaining
    """
    if len(anomaly_indices) == 0:
        return {
            'status': 'HEALTHY',
            'days_remaining': '>1000',
            'confidence': 'HIGH',
            'message': 'No anomalies detected. System operating normally.',
            'trend': 'STABLE'
        }
    
    total_points = len(data)
    anomaly_count = len(anomaly_indices)
    anomaly_rate = (anomaly_count / total_points) * 100
    
    # Calculate anomaly density in recent vs early data
    midpoint = total_points // 2
    early_anomalies = np.sum(anomaly_indices < midpoint)
    late_anomalies = np.sum(anomaly_indices >= midpoint)
    
    # Determine trend
    if late_anomalies > early_anomalies * 1.5:
        trend = 'DEGRADING'
    elif late_anomalies < early_anomalies * 0.5:
        trend = 'IMPROVING'
    else:
        trend = 'STABLE'
    
    # Estimate lifespan based on anomaly rate and trend
    if anomaly_rate < 1:
        days_remaining = '>1000'
        status = 'HEALTHY'
        confidence = 'HIGH'
        message = 'System is healthy. Minimal degradation detected.'
    elif anomaly_rate < 3:
        if trend == 'DEGRADING':
            days_remaining = '180-365'
            status = 'GOOD'
            confidence = 'MEDIUM'
            message = 'System showing early signs of wear. Monitor closely.'
        else:
            days_remaining = '>365'
            status = 'HEALTHY'
            confidence = 'HIGH'
            message = 'System is stable with minor anomalies.'
    elif anomaly_rate < 5:
        if trend == 'DEGRADING':
            days_remaining = '90-180'
            status = 'FAIR'
            confidence = 'MEDIUM'
            message = 'System degradation accelerating. Plan maintenance.'
        else:
            days_remaining = '180-365'
            status = 'GOOD'
            confidence = 'MEDIUM'
            message = 'Moderate anomaly rate. Continue monitoring.'
    elif anomaly_rate < 10:
        if trend == 'DEGRADING':
            days_remaining = '30-90'
            status = 'WARNING'
            confidence = 'MEDIUM'
            message = 'Significant degradation. Immediate attention recommended.'
        else:
            days_remaining = '90-180'
            status = 'FAIR'
            confidence = 'LOW'
            message = 'High anomaly rate. Schedule maintenance soon.'
    else:
        if trend == 'DEGRADING':
            days_remaining = '<30'
            status = 'CRITICAL'
            confidence = 'HIGH'
            message = 'CRITICAL: System failure imminent. Emergency maintenance required!'
        else:
            days_remaining = '30-90'
            status = 'WARNING'
            confidence = 'MEDIUM'
            message = 'Very high anomaly rate. Urgent maintenance needed.'
    
    return {
        'status': status,
        'days_remaining': days_remaining,
        'confidence': confidence,
        'message': message,
        'trend': trend,
        'anomaly_rate': anomaly_rate
    }


def main():
    """Main application function."""
    load_custom_css()
    initialize_session_state()
    
    st.markdown('<div class="main-header">🛰️ OrbiNasaSense</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">NASA Spacecraft Telemetry Anomaly Detection System</div>', unsafe_allow_html=True)
    
    selected_option, threshold_k, downsample_factor = sidebar_controls()
    
    st.markdown("---")
    
    if st.session_state.data is None:
        st.info("👈 Please load telemetry data from the sidebar to begin analysis.")
        return
    
    # Auto model selection logic
    if selected_option == 'AUTO':
        st.info("🤖 AUTO mode - Analyzing data...")
        
        # Get recommendation
        recommendation = get_model_recommendation(st.session_state.data)
        selected_channel = recommendation['recommended_model']
        
        # Simple display
        st.success(f"Selected Model: **{selected_channel}** (Confidence: {recommendation['confidence']}%)")
    else:
        # User manually selected a model
        selected_channel = selected_option
    
    st.success(f"✅ Data loaded: **{st.session_state.channel_name}** ({len(st.session_state.data):,} time steps)")
    
    st.subheader("📈 Raw Telemetry Data")
    fig_raw = plot_telemetry_data(st.session_state.data, title=f"Channel: {st.session_state.channel_name}")
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
            with st.spinner("Analyzing spacecraft telemetry..."):
                try:
                    model, config = load_model_and_config(selected_channel)
                    st.session_state.model_config = config
                    results = run_inference(st.session_state.data, model, config, 
                                          custom_threshold_k=threshold_k,
                                          downsample_factor=downsample_factor)
                    st.session_state.inference_results = results
                    st.session_state.selected_model = selected_channel
                    st.success("✅ Analysis complete!")
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    return
        
        if st.session_state.inference_results is not None:
            results = st.session_state.inference_results
            
            # Calculate lifespan
            lifespan = estimate_lifespan(st.session_state.data, results['anomaly_indices'])
            
            st.markdown("---")
            st.subheader("📊 Detection Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Anomalies Detected", f"{results['num_anomalies']:,}", 
                         delta=f"{results['anomaly_percentage']:.2f}%")
            with col2:
                st.metric("Detection Threshold", f"{results['threshold']:.6f}")
            with col3:
                st.metric("Window Size", f"{results['window_size']}")
            with col4:
                anomaly_rate = results['anomaly_percentage']
                health = "🟢 HEALTHY" if anomaly_rate < 3 else "🟡 WARNING" if anomaly_rate < 7 else "🔴 CRITICAL"
                st.metric("System Status", health)
            
            # Lifespan prediction
            st.markdown("---")
            st.subheader("🔮 Lifespan Prediction")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Health Status", lifespan['status'])
            with col2:
                st.metric("Estimated Lifespan", f"{lifespan['days_remaining']} days")
            with col3:
                st.metric("Degradation Trend", lifespan['trend'])
            with col4:
                st.metric("Confidence Level", lifespan['confidence'])
            
            st.info(f"**Analysis:** {lifespan['message']}")
            
            # Detailed visualizations
            st.markdown("---")
            st.subheader("📉 Detailed Analysis")
            
            fig_anomaly = plot_anomaly_detection(
                st.session_state.data,
                results['predictions'],
                results['anomaly_mask'],
                results['errors'],
                results['threshold'],
                results['anomaly_indices']
            )
            st.plotly_chart(fig_anomaly, width='stretch')
            
            # Anomaly details table
            if results['num_anomalies'] > 0:
                st.markdown("---")
                st.subheader("🔎 Anomaly Details")
                
                with st.expander(f"View all {results['num_anomalies']} detected anomalies"):
                    anomaly_indices = results['anomaly_indices']
                    df_anomalies = pd.DataFrame({
                        'Time Step': anomaly_indices,
                        'Actual Value': st.session_state.data[anomaly_indices],
                        'Predicted Value': results['predictions'][anomaly_indices],
                        'Error': results['errors'][anomaly_indices],
                        'Severity': ['HIGH' if abs(e) > 2*results['threshold'] else 'MEDIUM' 
                                   for e in results['errors'][anomaly_indices]]
                    })
                    st.dataframe(df_anomalies, width='stretch')
                    
                    # Download option
                    csv = df_anomalies.to_csv(index=False)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        "📥 Download Anomaly Report (CSV)",
                        csv,
                        f"anomalies_{st.session_state.channel_name}_{timestamp}.csv",
                        "text/csv",
                        width='stretch'
                    )
            else:
                st.success("✅ No anomalies detected in the data. System operating normally.")
    else:
        st.warning("⚠️ Please select a trained model from the sidebar to proceed.")


if __name__ == '__main__':
    main()
