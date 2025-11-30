"""
Automatic Model Selection based on data characteristics.
Analyzes uploaded data and selects the best model automatically.
"""
import numpy as np


def analyze_data_characteristics(data):
    """
    Analyze statistical characteristics of the data.
    
    Returns:
        dict: Statistical features of the data
    """
    # Basic statistics
    mean = np.mean(data)
    std = np.std(data)
    variance = np.var(data)
    
    # Range and scale
    data_range = np.ptp(data)  # peak to peak
    min_val = np.min(data)
    max_val = np.max(data)
    
    # Variability measures
    cv = std / mean if mean != 0 else 0  # coefficient of variation
    
    # Trend analysis
    # Check if data has increasing/decreasing trend
    indices = np.arange(len(data))
    trend_coef = np.polyfit(indices, data, 1)[0]
    
    # Periodicity check (simple autocorrelation)
    if len(data) > 100:
        autocorr = np.correlate(data - mean, data - mean, mode='same')
        autocorr = autocorr[len(autocorr)//2:]
        # Find peaks in autocorrelation
        has_periodicity = np.max(autocorr[10:min(100, len(autocorr))]) > 0.3 * autocorr[0]
    else:
        has_periodicity = False
    
    # Count significant changes (potential anomalies)
    diff = np.abs(np.diff(data))
    threshold = np.mean(diff) + 2 * np.std(diff)
    num_spikes = np.sum(diff > threshold)
    
    return {
        'mean': mean,
        'std': std,
        'variance': variance,
        'range': data_range,
        'min': min_val,
        'max': max_val,
        'cv': cv,
        'trend': trend_coef,
        'has_periodicity': has_periodicity,
        'num_spikes': num_spikes,
        'data_length': len(data)
    }


def select_best_model(data):
    """
    Automatically select the best model based on data characteristics.
    
    Args:
        data: numpy array of telemetry data
        
    Returns:
        tuple: (model_name, confidence, reason)
    """
    features = analyze_data_characteristics(data)
    
    # Decision rules based on data characteristics
    
    # Power System (P-1) characteristics:
    # - Stable with small variations (low CV)
    # - Range typically around 4-6 for normalized data
    # - Few sudden spikes
    # - Centered around positive values
    
    # Mechanical System (M-1) characteristics:
    # - Periodic patterns (cyclical)
    # - Higher variability
    # - Regular oscillations
    
    # Environmental (E-1) characteristics:
    # - Slow gradual changes
    # - Can have trend (heating/cooling)
    # - Lower frequency variations
    # - Wide range of values
    
    scores = {
        'P-1': 0,
        'M-1': 0,
        'E-1': 0
    }
    
    # Score for P-1 (Power System)
    if 0.4 <= features['mean'] <= 0.8:  # Typical normalized power voltage range
        scores['P-1'] += 30
    if features['cv'] < 0.2:  # Low coefficient of variation = stable
        scores['P-1'] += 25
    if features['num_spikes'] < len(data) * 0.02:  # Less than 2% spikes
        scores['P-1'] += 20
    if abs(features['trend']) < 0.0001:  # Relatively flat (no strong trend)
        scores['P-1'] += 15
    if not features['has_periodicity']:  # Power is usually non-periodic
        scores['P-1'] += 10
    
    # Score for M-1 (Mechanical System)
    if features['has_periodicity']:  # Cyclical patterns
        scores['M-1'] += 40
    if features['cv'] > 0.15:  # Higher variability
        scores['M-1'] += 20
    if features['num_spikes'] > len(data) * 0.01:  # Some spikes expected
        scores['M-1'] += 15
    if 0.3 <= features['mean'] <= 0.9:  # Typical range
        scores['M-1'] += 15
    if features['range'] > 0.4:  # Wider range of values
        scores['M-1'] += 10
    
    # Score for E-1 (Environmental)
    if abs(features['trend']) > 0.0001:  # Has trend (warming/cooling)
        scores['E-1'] += 30
    if features['cv'] > 0.3:  # Can have high variability
        scores['E-1'] += 20
    if -0.2 <= features['mean'] <= 0.3:  # Can be centered around 0
        scores['E-1'] += 20
    if features['range'] > 0.5:  # Wide temperature ranges
        scores['E-1'] += 15
    if features['num_spikes'] < len(data) * 0.01:  # Gradual changes
        scores['E-1'] += 15
    
    # Select model with highest score
    best_model = max(scores, key=scores.get)
    max_score = scores[best_model]
    total_possible = 100
    confidence = min(100, int((max_score / total_possible) * 100))
    
    # Generate reason
    reasons = {
        'P-1': f"Data shows stable patterns (CV: {features['cv']:.3f}) with mean around {features['mean']:.3f}, typical of power systems",
        'M-1': f"Data exhibits {'periodic behavior' if features['has_periodicity'] else 'cyclical patterns'} with variability (CV: {features['cv']:.3f}), characteristic of mechanical systems",
        'E-1': f"Data shows {'trending behavior' if abs(features['trend']) > 0.0001 else 'gradual changes'} with range {features['range']:.3f}, typical of environmental sensors"
    }
    
    # If confidence is too low, provide a warning
    if confidence < 50:
        reason = f"{reasons[best_model]} (Low confidence - data may be unusual or from unknown source)"
    else:
        reason = reasons[best_model]
    
    return best_model, confidence, reason


def get_model_recommendation(data):
    """
    Get model recommendation with detailed analysis.
    
    Args:
        data: numpy array of telemetry data
        
    Returns:
        dict: Recommendation details
    """
    model, confidence, reason = select_best_model(data)
    features = analyze_data_characteristics(data)
    
    return {
        'recommended_model': model,
        'confidence': confidence,
        'reason': reason,
        'data_summary': {
            'length': features['data_length'],
            'mean': f"{features['mean']:.4f}",
            'std': f"{features['std']:.4f}",
            'range': f"{features['range']:.4f}",
            'has_periodicity': features['has_periodicity'],
            'trend': 'Increasing' if features['trend'] > 0.0001 else 'Decreasing' if features['trend'] < -0.0001 else 'Stable'
        }
    }
