"""
Training script for LSTM anomaly detection model.

Usage:
    python train.py --channel P-1 --window-size 50 --epochs 50
"""
import os
import sys
import argparse
import json
import numpy as np
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_utils import load_npy_channel, create_sliding_windows, normalize_data

# Try to import TensorFlow version, fall back to sklearn version
try:
    import tensorflow as tf
    from model import build_lstm_model, create_callbacks, AnomalyDetector
    USE_TENSORFLOW = True
    print("Using TensorFlow implementation")
except ImportError:
    from model_sklearn import build_lstm_model, create_callbacks, AnomalyDetector
    USE_TENSORFLOW = False
    print("Using scikit-learn implementation (TensorFlow not available)")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train LSTM model for spacecraft telemetry anomaly detection'
    )
    
    parser.add_argument(
        '--channel',
        type=str,
        required=True,
        help='Channel name (e.g., P-1, M-1, etc.)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing train/test folders with .npy files'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        default=50,
        help='Window size for sliding windows'
    )
    
    parser.add_argument(
        '--lstm-units',
        type=int,
        default=64,
        help='Number of units in LSTM layers'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--threshold-k',
        type=float,
        default=3.0,
        help='Threshold multiplier (mean + k*std)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help='Directory to save trained models'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("NASA Spacecraft Telemetry Anomaly Detection - Training")
    print("=" * 60)
    print(f"Channel: {args.channel}")
    print(f"Window size: {args.window_size}")
    print(f"LSTM units: {args.lstm_units}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)
    
    # Construct file paths
    train_file = os.path.join(args.data_dir, 'train', f'{args.channel}_train.npy')
    test_file = os.path.join(args.data_dir, 'test', f'{args.channel}_test.npy')
    
    # Load training data
    print(f"\nLoading training data from {train_file}...")
    try:
        train_data = load_npy_channel(train_file)
        print(f"Training data shape: {train_data.shape}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nPlease ensure the data file exists at: {train_file}")
        print("You may need to download the SMAP/MSL dataset first.")
        sys.exit(1)
    
    # Load test data (optional, for validation)
    test_data = None
    if os.path.exists(test_file):
        print(f"Loading test data from {test_file}...")
        test_data = load_npy_channel(test_file)
        print(f"Test data shape: {test_data.shape}")
    
    # Normalize data
    print("\nNormalizing data...")
    if test_data is not None:
        train_normalized, test_normalized, mean, std = normalize_data(train_data, test_data)
    else:
        train_normalized, mean, std = normalize_data(train_data)
        test_normalized = None
    
    print(f"Mean: {mean:.4f}, Std: {std:.4f}")
    
    # Create sliding windows
    print(f"\nCreating sliding windows (window_size={args.window_size})...")
    X_train, y_train = create_sliding_windows(train_normalized, args.window_size)
    print(f"Training windows: {X_train.shape}, Targets: {y_train.shape}")
    
    # Reshape for LSTM (add feature dimension)
    X_train = X_train.reshape(-1, args.window_size, 1)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Build model
    print("\nBuilding LSTM model...")
    model = build_lstm_model(
        window_size=args.window_size,
        lstm_units=args.lstm_units
    )
    
    if USE_TENSORFLOW:
        model.summary()
    else:
        print(f"Model: RandomForest with {args.lstm_units} estimators")
    
    # Define model path
    if USE_TENSORFLOW:
        model_path = os.path.join(args.output_dir, f'lstm_model_{args.channel}.keras')
    else:
        model_path = os.path.join(args.output_dir, f'model_{args.channel}.pkl')
    
    # Train model
    print("\nTraining model...")
    if USE_TENSORFLOW:
        # Create callbacks
        callbacks = create_callbacks(model_path, patience=10)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        print("\nLoading best model...")
        model = tf.keras.models.load_model(model_path)
    else:
        # Train sklearn model
        model.fit(X_train, y_train)
        
        # Save model
        model.save(model_path)
        print(f"Model saved to {model_path}")
    
    # Create anomaly detector
    detector = AnomalyDetector(model)
    
    # Compute threshold from training data
    print(f"\nComputing anomaly threshold (k={args.threshold_k})...")
    threshold = detector.compute_threshold(X_train, y_train, k=args.threshold_k)
    
    print(f"Mean training error: {detector.mean_error:.6f}")
    print(f"Std training error: {detector.std_error:.6f}")
    print(f"Anomaly threshold: {threshold:.6f}")
    
    # Save threshold and normalization parameters
    config = {
        'channel': args.channel,
        'window_size': args.window_size,
        'lstm_units': args.lstm_units,
        'threshold': float(threshold),
        'mean_error': float(detector.mean_error),
        'std_error': float(detector.std_error),
        'threshold_k': args.threshold_k,
        'data_mean': float(mean),
        'data_std': float(std),
        'train_data_shape': train_data.shape,
        'model_path': model_path
    }
    
    config_path = os.path.join(args.output_dir, f'threshold_{args.channel}.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Configuration saved to: {config_path}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions, val_errors, val_anomalies = detector.detect_anomalies(X_val, y_val)
    
    num_anomalies = np.sum(val_anomalies)
    anomaly_rate = num_anomalies / len(val_anomalies) * 100
    
    print(f"Validation set anomalies: {num_anomalies} / {len(val_anomalies)} ({anomaly_rate:.2f}%)")
    
    # Training history summary
    if USE_TENSORFLOW:
        print("\nTraining Summary:")
        print(f"Final training loss: {history.history['loss'][-1]:.6f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    else:
        print("\nTraining Summary:")
        print("Model trained successfully with RandomForest")
        # Evaluate on validation set
        val_predictions = model.predict(X_val)
        val_mse = np.mean((val_predictions.flatten() - y_val) ** 2)
        print(f"Validation MSE: {val_mse:.6f}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"\nTo run inference, use:")
    print(f"  streamlit run app.py")
    

if __name__ == '__main__':
    main()
