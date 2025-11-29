"""
Utility functions for loading and preprocessing NASA telemetry data.
"""
import numpy as np
import os
from typing import Tuple, Optional


def load_npy_channel(channel_path: str) -> np.ndarray:
    """
    Load a single telemetry channel from a .npy file.
    
    Args:
        channel_path: Path to the .npy file
        
    Returns:
        numpy array of shape (n_timesteps,) or (n_timesteps, n_features)
    """
    if not os.path.exists(channel_path):
        raise FileNotFoundError(f"Channel file not found: {channel_path}")
    
    data = np.load(channel_path)
    
    # If data is 2D, flatten to 1D if it's a single feature
    if len(data.shape) == 2 and data.shape[1] == 1:
        data = data.flatten()
    
    return data


def create_sliding_windows(data: np.ndarray, 
                          window_size: int, 
                          stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for time series data.
    
    Args:
        data: 1D array of time series data
        window_size: Number of time steps in each window
        stride: Step size for sliding the window
        
    Returns:
        X: Input sequences of shape (n_windows, window_size)
        y: Target values of shape (n_windows,) - next value after each window
    """
    if len(data.shape) > 1:
        data = data.flatten()
    
    X, y = [], []
    
    for i in range(0, len(data) - window_size, stride):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    
    return np.array(X), np.array(y)


def normalize_data(train_data: np.ndarray, 
                   test_data: Optional[np.ndarray] = None) -> Tuple:
    """
    Normalize data using mean and std from training data.
    
    Args:
        train_data: Training data
        test_data: Optional test data
        
    Returns:
        If test_data is None: (normalized_train, mean, std)
        Otherwise: (normalized_train, normalized_test, mean, std)
    """
    mean = np.mean(train_data)
    std = np.std(train_data)
    
    if std == 0:
        std = 1.0  # Avoid division by zero
    
    normalized_train = (train_data - mean) / std
    
    if test_data is not None:
        normalized_test = (test_data - mean) / std
        return normalized_train, normalized_test, mean, std
    
    return normalized_train, mean, std


def denormalize_data(normalized_data: np.ndarray, 
                     mean: float, 
                     std: float) -> np.ndarray:
    """
    Denormalize data back to original scale.
    
    Args:
        normalized_data: Normalized data
        mean: Mean used for normalization
        std: Standard deviation used for normalization
        
    Returns:
        Denormalized data
    """
    return normalized_data * std + mean


def get_available_channels(data_dir: str, subset: str = 'train') -> list:
    """
    Get list of available channel files in a directory.
    
    Args:
        data_dir: Base data directory
        subset: 'train' or 'test'
        
    Returns:
        List of channel filenames
    """
    path = os.path.join(data_dir, subset)
    
    if not os.path.exists(path):
        return []
    
    channels = [f for f in os.listdir(path) if f.endswith('.npy')]
    return sorted(channels)
