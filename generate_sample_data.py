"""
Generate synthetic telemetry data for demonstration purposes.
This creates sample data similar to NASA SMAP/MSL telemetry.
"""
import numpy as np
import os

# Set random seed for reproducibility
np.random.seed(42)

def generate_telemetry_channel(n_samples=8000, anomaly_rate=0.05):
    """
    Generate synthetic spacecraft telemetry data with anomalies.
    
    Args:
        n_samples: Number of time steps
        anomaly_rate: Fraction of data that should be anomalous
    
    Returns:
        data: Telemetry time series with injected anomalies
    """
    # Base signal: combination of trends and periodic components
    t = np.linspace(0, 100, n_samples)
    
    # Normal behavior: slow drift + multiple periodic components
    trend = 0.3 * np.sin(t / 20)
    seasonal1 = 0.2 * np.sin(2 * np.pi * t / 50)
    seasonal2 = 0.1 * np.cos(2 * np.pi * t / 30)
    noise = np.random.normal(0, 0.05, n_samples)
    
    data = trend + seasonal1 + seasonal2 + noise + 0.5
    
    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        # Different types of anomalies
        anomaly_type = np.random.choice(['spike', 'drop', 'shift'])
        
        if anomaly_type == 'spike':
            data[idx] += np.random.uniform(0.5, 1.5)
        elif anomaly_type == 'drop':
            data[idx] -= np.random.uniform(0.5, 1.5)
        else:  # shift
            # Create a temporary shift
            shift_length = min(50, n_samples - idx)
            data[idx:idx+shift_length] += np.random.uniform(0.3, 0.7)
    
    return data


def main():
    """Generate sample data for multiple channels."""
    
    # Create directories
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)
    
    print("Generating synthetic telemetry data...")
    print("=" * 60)
    
    # Generate data for SMAP channels (P-1, P-2)
    channels = ['P-1', 'P-2', 'M-1']
    
    for channel in channels:
        print(f"\nGenerating channel {channel}...")
        
        # Training data (normal data with few anomalies)
        train_data = generate_telemetry_channel(n_samples=8000, anomaly_rate=0.02)
        train_file = f'data/train/{channel}_train.npy'
        np.save(train_file, train_data)
        print(f"  ✓ Saved training data: {train_file} (shape: {train_data.shape})")
        
        # Test data (with more anomalies to detect)
        test_data = generate_telemetry_channel(n_samples=3000, anomaly_rate=0.08)
        test_file = f'data/test/{channel}_test.npy'
        np.save(test_file, test_data)
        print(f"  ✓ Saved test data: {test_file} (shape: {test_data.shape})")
    
    print("\n" + "=" * 60)
    print("✅ Sample data generation complete!")
    print("\nYou can now:")
    print("  1. Train a model: python train.py --channel P-1 --epochs 30")
    print("  2. Run the UI: streamlit run app.py")
    print("\nNote: This is synthetic data for demonstration.")
    print("For real NASA data, download from:")
    print("  https://software.nasa.gov/software/NPO-50838-1")
    print("=" * 60)


if __name__ == '__main__':
    main()
