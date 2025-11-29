"""
LSTM model for spacecraft telemetry anomaly detection.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple


def build_lstm_model(window_size: int, 
                     lstm_units: int = 64, 
                     dropout_rate: float = 0.2) -> keras.Model:
    """
    Build a simple LSTM model for time series prediction.
    
    The model takes a sequence of `window_size` timesteps and predicts the next value.
    
    Args:
        window_size: Number of timesteps in input sequence
        lstm_units: Number of units in LSTM layers
        dropout_rate: Dropout rate for regularization
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(window_size, 1)),
        
        # First LSTM layer
        layers.LSTM(lstm_units, return_sequences=True),
        layers.Dropout(dropout_rate),
        
        # Second LSTM layer
        layers.LSTM(lstm_units // 2, return_sequences=False),
        layers.Dropout(dropout_rate),
        
        # Dense output layer
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    
    # Compile with Adam optimizer and MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model


def create_callbacks(model_path: str, patience: int = 10) -> list:
    """
    Create training callbacks for model checkpointing and early stopping.
    
    Args:
        model_path: Path to save the best model
        patience: Number of epochs to wait for improvement before stopping
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    return callbacks


class AnomalyDetector:
    """
    Wrapper class for the LSTM anomaly detection model.
    """
    
    def __init__(self, model: keras.Model):
        """
        Initialize the anomaly detector.
        
        Args:
            model: Trained Keras LSTM model
        """
        self.model = model
        self.threshold = None
        self.mean_error = None
        self.std_error = None
    
    def compute_threshold(self, 
                         train_data: np.ndarray, 
                         train_targets: np.ndarray,
                         k: float = 3.0) -> float:
        """
        Compute anomaly detection threshold from training errors.
        
        Args:
            train_data: Training input sequences
            train_targets: Training target values
            k: Number of standard deviations above mean for threshold
            
        Returns:
            Computed threshold value
        """
        # Predict on training data
        predictions = self.model.predict(train_data, verbose=0)
        
        # Compute errors
        errors = np.abs(predictions.flatten() - train_targets)
        
        # Calculate statistics
        self.mean_error = np.mean(errors)
        self.std_error = np.std(errors)
        
        # Set threshold
        self.threshold = self.mean_error + k * self.std_error
        
        return self.threshold
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Make predictions using the model.
        
        Args:
            data: Input sequences
            
        Returns:
            Predicted values
        """
        return self.model.predict(data, verbose=0)
    
    def detect_anomalies(self, 
                        data: np.ndarray, 
                        targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Detect anomalies in the data.
        
        Args:
            data: Input sequences
            targets: True target values
            
        Returns:
            Tuple of (predictions, errors, anomaly_mask)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call compute_threshold first.")
        
        # Make predictions
        predictions = self.predict(data)
        
        # Compute errors
        errors = np.abs(predictions.flatten() - targets)
        
        # Create anomaly mask
        anomaly_mask = errors > self.threshold
        
        return predictions.flatten(), errors, anomaly_mask
