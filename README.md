# 🛰️ OrbiNasaSense - Spacecraft Orbital Watch System

**Intelligent anomaly detection system for NASA spacecraft telemetry using machine learning.** This project automatically monitors spacecraft sensor data and detects unusual patterns that could indicate equipment failures, solar flare damage, or other critical issues - helping engineers identify problems before they become catastrophic.

Built with Python, NumPy, and Streamlit, OrbiNasaSense provides real-time visualization and analysis of spacecraft telemetry data with an intuitive web-based dashboard.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ (3.12 or 3.14 recommended)
- pip (Python package manager)

### Installation & Setup

1. **Clone the repository:**
```bash
git clone https://github.com/Vikram739/OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry.git
cd OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Generate sample data:**
```bash
python generate_sample_data.py
```
This creates synthetic NASA-like telemetry data in `data/train/` and `data/test/` folders.

4. **Train a model:**
```bash
python train.py --channel P-1 --window-size 50 --epochs 50
```
This trains an anomaly detection model on the P-1 channel data (takes 2-5 minutes).

5. **Launch the dashboard:**
```bash
streamlit run app.py
```
Your browser will open automatically to `http://localhost:8501`.

---

## 📖 How to Use

### Step 1: Select a Model
In the left sidebar, choose a trained model from the dropdown (e.g., "P-1").

### Step 2: Load Data
- **Option A:** Click "Load Sample Data" and select a test file (e.g., `P-1_test.npy`)
- **Option B:** Upload your own `.npy` file with telemetry data

### Step 3: Run Detection
Click the **"🚀 Run Anomaly Detection"** button in the main panel.

### Step 4: Analyze Results
The dashboard displays:
- **Telemetry Plot:** Blue line shows raw data, red X's mark detected anomalies
- **Predictions Plot:** Compares actual values vs. model predictions
- **Error Analysis:** Shows prediction errors and threshold line
- **Metrics:** Number of anomalies, detection rate, threshold value
- **Details Table:** Expandable list of all detected anomalies with timestamps
- **Export:** Download anomaly data as CSV for further analysis

### Adjusting Sensitivity
Use the **Threshold Multiplier (k)** slider in the sidebar:
- **Lower (1.0-2.0):** More sensitive, catches smaller anomalies
- **Default (3.0):** Balanced detection
- **Higher (4.0-5.0):** Less sensitive, only major anomalies

---

## 🔧 What This Project Does

### Core Functionality

**1. Time Series Prediction**
- Uses sliding window approach to learn patterns in spacecraft sensor data
- Predicts next sensor value based on previous readings (default: 50 time steps)
- Supports any numerical telemetry: temperature, pressure, voltage, etc.

**2. Anomaly Detection**
- Compares predictions with actual values
- Flags data points where error exceeds learned threshold
- Adaptive threshold based on training data statistics: `threshold = mean_error + k × std_error`

**3. Interactive Visualization**
- Real-time Plotly charts with zoom, pan, and hover capabilities
- Multi-subplot analysis: raw data, predictions, and error analysis
- Customizable parameters for different use cases

**4. Data Export**
- Download detected anomalies as CSV
- Includes timestamps, actual values, predicted values, and error magnitudes

### Technical Architecture

```
Data → Preprocessing → Model Training → Anomaly Detection → Visualization
  ↓         ↓              ↓                  ↓                 ↓
.npy    Normalize    Linear Regression   Error > Threshold   Streamlit UI
files   Windows      (or LSTM*)           Flag anomaly       Interactive
```

*Linear Regression used for Python 3.14 compatibility; LSTM available for Python 3.8-3.12*

### Key Features

✅ **Python 3.14 Compatible** - Uses pure NumPy implementation  
✅ **Flexible Data Input** - Works with any numerical time series  
✅ **Adjustable Sensitivity** - Tune threshold for your use case  
✅ **Visual Feedback** - Interactive charts and real-time analysis  
✅ **Production Ready** - Complete data pipeline from raw data to insights  
✅ **Extensible** - Easy to add new channels or data sources  

---

## 📁 Project Structure

```
OrbiNasaSense/
├── app.py                      # Streamlit web dashboard
├── train.py                    # Model training script
├── generate_sample_data.py     # Synthetic data generator
├── requirements.txt            # Python dependencies
├── LAUNCH_UI.bat              # Quick launcher (Windows)
├── RUN_PROJECT.bat            # Full setup script (Windows)
├── RUN_PROJECT.ps1            # PowerShell launcher
├── src/
│   ├── data_utils.py          # Data loading utilities
│   ├── inference.py           # Anomaly detection logic
│   ├── model_sklearn.py       # Pure NumPy model (Python 3.14)
│   └── model.py               # TensorFlow LSTM model (Python 3.8-3.12)
├── data/
│   ├── train/                 # Training data (.npy files)
│   └── test/                  # Test data (.npy files)
└── models/
    ├── model_*.pkl            # Trained models
    └── threshold_*.json       # Detection thresholds
```

---

## 🎯 Use Cases

- **NASA/ESA Missions:** Monitor satellite and spacecraft sensors
- **CubeSat Operations:** Detect anomalies in small satellite telemetry
- **Ground Testing:** Analyze pre-launch sensor data
- **Research:** Study spacecraft behavior patterns
- **Education:** Learn anomaly detection and time series analysis

---

## 🔬 Training Your Own Models

### Basic Training
```bash
python train.py --channel CHANNEL_NAME --window-size 50 --epochs 50
```

### Advanced Options
```bash
python train.py \
  --channel P-2 \
  --window-size 100 \          # Longer memory (default: 50)
  --epochs 100 \               # More training (default: 50)
  --batch-size 64 \            # Larger batches (default: 32)
  --threshold-k 2.5            # More sensitive (default: 3.0)
```

### Parameters Explained

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `--channel` | Name of the data channel | Any string (e.g., P-1, temperature) |
| `--window-size` | How many past values to consider | 50-100 for high-freq, 10-30 for slow |
| `--epochs` | Training iterations | 50-100 (more = better, slower) |
| `--batch-size` | Samples per training step | 32-128 |
| `--threshold-k` | Anomaly sensitivity multiplier | 2.0 (sensitive) to 4.0 (robust) |

---

## 📊 Working with Your Own Data

### Data Format Requirements

Your data must be:
1. **Time series** - Sequential measurements over time
2. **Numerical** - Sensor readings (floats or integers)
3. **Single channel** - One sensor per file
4. **NumPy format** - Saved as `.npy` file

### Converting Your Data

**From CSV:**
```python
import numpy as np
import pandas as pd

# Read your CSV
df = pd.read_csv('satellite_data.csv')

# Extract sensor column
data = df['temperature'].values  # or 'pressure', 'voltage', etc.

# Save as .npy
np.save('data/train/my_sensor_train.npy', data[:8000])
np.save('data/test/my_sensor_test.npy', data[8000:])
```

**From Text File:**
```python
import numpy as np

# One value per line
data = np.loadtxt('sensor_readings.txt')
np.save('data/train/my_data_train.npy', data)
```

**From Excel:**
```python
import pandas as pd
import numpy as np

df = pd.read_excel('telemetry.xlsx')
data = df['sensor_1'].values
np.save('data/train/sensor1_train.npy', data)
```

### Real NASA Data

Download official NASA SMAP/MSL telemetry:
```bash
# From GitHub
git clone https://github.com/khundman/telemanom.git

# Copy to project
cp telemanom/data/train/*.npy data/train/
cp telemanom/data/test/*.npy data/test/

# Train on real data
python train.py --channel P-1 --epochs 100
```

---

## 🛠️ Troubleshooting

### "No trained models found"
Run training first:
```bash
python train.py --channel P-1 --epochs 50
```

### "No data files found"
Generate sample data:
```bash
python generate_sample_data.py
```

### Streamlit won't start
Use the full Python path:
```bash
# Windows
python -m streamlit run app.py

# Mac/Linux
python3 -m streamlit run app.py
```

### Port already in use
Change the port:
```bash
streamlit run app.py --server.port 8502
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 📄 License

MIT License - feel free to use this project for educational or commercial purposes.

---

## 🙏 Acknowledgments

- **NASA SMAP/MSL** - Inspiration and data format
- **Telemanom Project** - Reference implementation
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations

---

## 📧 Contact

**Developer:** Vikram  
**GitHub:** [@Vikram739](https://github.com/Vikram739)  
**Project Link:** [OrbiNasaSense](https://github.com/Vikram739/OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry)

---

## ⭐ Star This Project

If you found this useful, please consider giving it a star! It helps others discover the project.

---

**Built with 💙 for spacecraft safety and anomaly detection research.**
