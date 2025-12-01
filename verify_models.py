import pickle
import json
import os
import numpy as np

print('=' * 60)
print('VERIFYING MODELS - NASA DATA CHECK')
print('=' * 60)

models_dir = 'models'

for channel in ['P-1', 'M-1', 'E-1']:
    print(f'\n{"=" * 60}')
    print(f'Channel: {channel}')
    print(f'{"=" * 60}')
    
    config_file = os.path.join(models_dir, f'threshold_{channel}.json')
    model_file = os.path.join(models_dir, f'model_{channel}.pkl')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f'\nConfig Info:')
        print(f'  Data mean: {config.get("data_mean", "NOT FOUND"):.4f}')
        print(f'  Data std: {config.get("data_std", "NOT FOUND"):.4f}')
        print(f'  Mean error: {config.get("mean_error", "NOT FOUND"):.6f}')
        print(f'  Threshold: {config.get("threshold", "NOT FOUND"):.6f}')
        
        # Check NASA files
        nasa_train = f'data/train/{channel}.npy'
        
        if os.path.exists(nasa_train):
            train_data = np.load(nasa_train)
            print(f'\nNASA training file: {nasa_train}')
            print(f'   Shape: {train_data.shape}')
            print(f'   Mean: {np.mean(train_data):.4f}')
            print(f'   Std: {np.std(train_data):.4f}')
            
            # Compare with config
            mean_diff = abs(config.get('data_mean', 0) - np.mean(train_data))
            print(f'   Mean difference: {mean_diff:.6f}')
            
            if mean_diff < 0.01:
                print(f'\n*** MODEL TRAINED ON NASA DATA ***')
            else:
                print(f'\n*** MODEL NOT TRAINED ON NASA DATA ***')
                print(f'   Expected mean: {np.mean(train_data):.4f}')
                print(f'   Model has mean: {config.get("data_mean", 0):.4f}')
        else:
            print(f'WARNING: NASA file not found: {nasa_train}')
    
    if os.path.exists(model_file):
        size = os.path.getsize(model_file) / 1024
        print(f'Model file: {model_file} ({size:.2f} KB)')

print('\n' + '=' * 60)
print('VERIFICATION COMPLETE')
print('=' * 60)
