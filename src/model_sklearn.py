"""
Lightweight model using pure NumPy for anomaly detection.
Compatible with Python 3.14+ (avoids sklearn threading issues)
"""
import numpy as np
import pickle
from typing import Tuple


class SimpleLSTMAlternative:
    """
    Simple regression-based alternative to LSTM for time series prediction.
    Uses only NumPy - fully compatible with Python 3.14+.
    """
    
    def __init__(self, window_size: int, n_estimators: int = 100):
        """
        Initialize the model.
        
        Args:
            window_size: Number of timesteps in input sequence
            n_estimators: Not used (kept for compatibility)
        """
        self.window_size = window_size
        self.weights = None
        self.bias = None
        self.mean = None
        self.std = None
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """
        Train the model using linear regression.
        
        Args:
            X: Input sequences of shape (n_samples, window_size, 1)
            y: Target values of shape (n_samples,)
        """
        # Reshape X to 2D
        X_2d = X.reshape(X.shape[0], -1)
        
        # Normalize
        self.mean = np.mean(X_2d, axis=0)
        self.std = np.std(X_2d, axis=0) + 1e-8  # Avoid division by zero
        X_norm = (X_2d - self.mean) / self.std
        
        # Add bias term
        X_bias = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])
        
        # Solve using normal equations: w = (X^T X)^{-1} X^T y
        try:
            self.weights = np.linalg.lstsq(X_bias, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Fallback to simple mean prediction if lstsq fails
            self.weights = np.zeros(X_bias.shape[1])
            self.weights[-1] = np.mean(y)
        
        self.is_fitted = True
        return self
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray, verbose=0) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Input sequences of shape (n_samples, window_size, 1)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")
        
        # Reshape and normalize
        X_2d = X.reshape(X.shape[0], -1)
        X_norm = (X_2d - self.mean) / self.std
        
        # Add bias term
        X_bias = np.hstack([X_norm, np.ones((X_norm.shape[0], 1))])
        
        # Predict
        predictions = X_bias @ self.weights
        
        return predictions.reshape(-1, 1)
    
    def save(self, filepath: str):
        """Save model to file."""
        model_data = {
            'weights': self.weights,
            'mean': self.mean,
            'std': self.std,
            'window_size': self.window_size,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    @classmethod
    def load(cls, filepath: str):
        """Load model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        instance = cls(model_data['window_size'])
        instance.weights = model_data['weights']
        instance.mean = model_data['mean']
        instance.std = model_data['std']
        instance.is_fitted = model_data['is_fitted']
        
        return instance


def build_lstm_model(window_size: int, 
                     lstm_units: int = 64, 
                     dropout_rate: float = 0.2):
    """
    Build a model for time series prediction.
    Uses pure NumPy linear regression for Python 3.14 compatibility.
    
    Args:
        window_size: Number of timesteps in input sequence
        lstm_units: Number of estimators (trees) to use
        dropout_rate: Not used in this implementation
        
    Returns:
        Model instance
    """
    return SimpleLSTMAlternative(window_size, n_estimators=lstm_units)


def create_callbacks(model_path: str, patience: int = 10) -> list:
    """
    Create training callbacks (compatibility function).
    Not used with sklearn but kept for API compatibility.
    
    Args:
        model_path: Path to save the best model
        patience: Not used
        
    Returns:
        Empty list (callbacks not needed for sklearn)
    """
    return []


class AnomalyDetector:
    """
    Wrapper class for the anomaly detection model.
    """
    
    def __init__(self, model):
        """
        Initialize the anomaly detector.
        
        Args:
            model: Trained model
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
        predictions = self.model.predict(train_data)
        
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
        return self.model.predict(data)
    
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
