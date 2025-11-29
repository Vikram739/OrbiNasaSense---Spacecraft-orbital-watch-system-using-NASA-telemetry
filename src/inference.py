"""
Inference utility for LSTM anomaly detection.

This module provides functions to load trained models and run inference
on spacecraft telemetry data.
"""
import os
import json
import numpy as np
from typing import Tuple, Dict, Optional
import pickle

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import create_sliding_windows, normalize_data

# Try to import TensorFlow, fall back to sklearn
try:
    import tensorflow as tf
    USE_TENSORFLOW = True
except ImportError:
    USE_TENSORFLOW = False
    from model_sklearn import SimpleLSTMAlternative


def load_model_and_config(channel: str, 
                          models_dir: str = 'models') -> Tuple:
    """
    Load a trained model and its configuration.
    
    Args:
        channel: Channel name (e.g., 'P-1', 'M-1')
        models_dir: Directory containing saved models
        
    Returns:
        Tuple of (model, config_dict)
    """
    # Load config first to determine model type
    config_path = os.path.join(models_dir, f'threshold_{channel}.json')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Try to load TensorFlow model first
    if USE_TENSORFLOW:
        model_path = os.path.join(models_dir, f'lstm_model_{channel}.keras')
        if not os.path.exists(model_path):
            model_path = os.path.join(models_dir, f'lstm_model_{channel}.h5')
        
        if os.path.exists(model_path):
            print(f"Loading TensorFlow model from {model_path}...")
            model = tf.keras.models.load_model(model_path)
            return model, config
    
    # Load sklearn model
    model_path = os.path.join(models_dir, f'model_{channel}.pkl')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found for channel {channel} in {models_dir}")
    
    print(f"Loading model from {model_path}...")
    model = SimpleLSTMAlternative.load(model_path)
    
    return model, config


def run_inference(data: np.ndarray,
                 model,
                 config: Dict,
                 custom_threshold_k: Optional[float] = None) -> Dict:
    """
    Run inference on telemetry data and detect anomalies.
    
    Args:
        data: 1D array of telemetry data
        model: Trained Keras model
        config: Configuration dictionary with threshold and normalization params
        custom_threshold_k: Optional custom threshold multiplier (overrides config)
        
    Returns:
        Dictionary containing:
            - 'predictions': Predicted values (denormalized)
            - 'errors': Prediction errors
            - 'anomaly_mask': Boolean array indicating anomalies
            - 'anomaly_indices': Indices where anomalies occur
            - 'threshold': Threshold used
            - 'num_anomalies': Count of anomalous points
            - 'anomaly_percentage': Percentage of anomalous points
    """
    # Extract config parameters
    window_size = config['window_size']
    data_mean = config['data_mean']
    data_std = config['data_std']
    mean_error = config['mean_error']
    std_error = config['std_error']
    
    # Determine threshold
    if custom_threshold_k is not None:
        threshold = mean_error + custom_threshold_k * std_error
    else:
        threshold = config['threshold']
    
    # Normalize data
    normalized_data = (data - data_mean) / data_std
    
    # Create sliding windows
    X, y = create_sliding_windows(normalized_data, window_size)
    
    # Reshape for LSTM
    X = X.reshape(-1, window_size, 1)
    
    # Make predictions
    print(f"Running inference on {len(X)} windows...")
    predictions_normalized = model.predict(X, verbose=0).flatten()
    
    # Denormalize predictions
    predictions = predictions_normalized * data_std + data_mean
    actual_values = y * data_std + data_mean
    
    # Compute errors
    errors = np.abs(predictions_normalized - y)
    
    # Detect anomalies
    anomaly_mask = errors > threshold
    anomaly_indices = np.where(anomaly_mask)[0]
    
    # Calculate statistics
    num_anomalies = int(np.sum(anomaly_mask))
    anomaly_percentage = (num_anomalies / len(anomaly_mask)) * 100 if len(anomaly_mask) > 0 else 0
    
    # Prepare full-length arrays (accounting for window offset)
    # The first `window_size` points don't have predictions
    full_length = len(data)
    full_predictions = np.full(full_length, np.nan)
    full_errors = np.full(full_length, np.nan)
    full_anomaly_mask = np.zeros(full_length, dtype=bool)
    
    # Fill in predictions starting from window_size
    full_predictions[window_size:window_size+len(predictions)] = predictions
    full_errors[window_size:window_size+len(errors)] = errors
    full_anomaly_mask[window_size:window_size+len(anomaly_mask)] = anomaly_mask
    
    result = {
        'predictions': full_predictions,
        'actual_values': actual_values,
        'errors': full_errors,
        'anomaly_mask': full_anomaly_mask,
        'anomaly_indices': anomaly_indices + window_size,  # Adjust for offset
        'threshold': threshold,
        'num_anomalies': num_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'window_size': window_size,
        'data_mean': data_mean,
        'data_std': data_std
    }
    
    return result


def quick_inference(channel: str, 
                   data: np.ndarray,
                   models_dir: str = 'models',
                   threshold_k: Optional[float] = None) -> Dict:
    """
    Convenience function to load model and run inference in one call.
    
    Args:
        channel: Channel name
        data: Telemetry data array
        models_dir: Directory containing saved models
        threshold_k: Optional custom threshold multiplier
        
    Returns:
        Inference results dictionary
    """
    model, config = load_model_and_config(channel, models_dir)
    return run_inference(data, model, config, threshold_k)


def print_inference_summary(result: Dict):
    """
    Print a summary of inference results.
    
    Args:
        result: Result dictionary from run_inference
    """
    print("\n" + "=" * 60)
    print("Inference Summary")
    print("=" * 60)
    print(f"Total data points: {len(result['anomaly_mask'])}")
    print(f"Window size: {result['window_size']}")
    print(f"Anomaly threshold: {result['threshold']:.6f}")
    print(f"Detected anomalies: {result['num_anomalies']}")
    print(f"Anomaly percentage: {result['anomaly_percentage']:.2f}%")
    
    if result['num_anomalies'] > 0:
        print(f"\nFirst 10 anomaly locations: {result['anomaly_indices'][:10]}")
    
    print("=" * 60)


if __name__ == '__main__':
    """
    Example usage of the inference module.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Run inference on telemetry data')
    parser.add_argument('--channel', type=str, required=True, help='Channel name')
    parser.add_argument('--data-file', type=str, required=True, help='Path to .npy data file')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--threshold-k', type=float, default=None, help='Custom threshold multiplier')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data_file}...")
    data = np.load(args.data_file)
    
    if len(data.shape) > 1:
        data = data.flatten()
    
    print(f"Data shape: {data.shape}")
    
    # Run inference
    result = quick_inference(
        args.channel,
        data,
        args.models_dir,
        args.threshold_k
    )
    
    # Print summary
    print_inference_summary(result)
