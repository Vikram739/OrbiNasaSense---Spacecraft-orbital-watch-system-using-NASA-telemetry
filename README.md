# 🛰️ OrbiNasaSense - Spacecraft Anomaly Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://orbinaasense.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Real-time anomaly detection system for NASA spacecraft telemetry using machine learning.**

Automatically monitors spacecraft sensor data and detects unusual patterns that could indicate equipment failures or critical issues. Features AUTO model selection that intelligently analyzes your data and picks the optimal detection algorithm.

🌐 **[Live Demo](https://orbinaasense.streamlit.app)** | 📊 **[GitHub](https://github.com/Vikram739/OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry)**

---

## ✨ Features

- 🤖 **AUTO Model Selection** - Automatically detects data type and selects best model
- ⚡ **Fast Processing** - Downsample mode for 2x speed boost
- 📊 **Interactive Dashboard** - Real-time visualization with Plotly charts
- 🔮 **Lifespan Prediction** - Estimates remaining operational days
- 📈 **3-Panel Analysis** - Telemetry, predictions, and error visualization
- 💾 **Export Reports** - Download anomaly data as CSV
- 🎯 **Multi-Model Support** - P-1 (Power), M-1 (Mechanical), E-1 (Environmental)

---

## 🚀 Quick Start

### Option 1: Use Live Demo (Easiest)

Visit **[orbinaasense.streamlit.app](https://orbinaasense.streamlit.app)** - No installation needed!

### Option 2: Run Locally

```bash
# 1. Clone repository
git clone https://github.com/Vikram739/OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry.git
cd OrbiNasaSense---Spacecraft-orbital-watch-system-using-NASA-telemetry

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch app
streamlit run app.py
```

Open browser at `http://localhost:8501`

---

## 📖 How to Use

### 1. **Select Model**
- **AUTO** (Recommended) - Analyzes your data automatically
- **P-1** - Power/Electrical systems
- **M-1** - Mechanical systems
- **E-1** - Environmental sensors

### 2. **Load Data**
- Choose sample from dataset, or
- Upload your own `.npy`, `.csv`, or `.txt` file

### 3. **Adjust Settings** (Optional)
- **Threshold Multiplier** - Detection sensitivity (1.0-5.0)
- **Processing Speed** - Normal or Fast (2x speed)

### 4. **Run Detection**
Click "🚀 Run Anomaly Detection" button

### 5. **Analyze Results**
- View detected anomalies in interactive charts
- Check lifespan prediction
- Download detailed report

---

## 🏗️ Project Structure

```
OrbiNasaSense/
├── app.py                          # Streamlit UI
├── train.py                        # Model training
├── requirements.txt                # Dependencies
├── .github/workflows/
│   └── keep-alive.yml             # Auto-ping to prevent sleep
├── .streamlit/
│   └── config.toml                # Streamlit config
├── src/
│   ├── data_utils.py              # Data utilities
│   ├── inference.py               # Detection logic
│   ├── model_sklearn.py           # Pure NumPy model
│   └── auto_model_selector.py     # AUTO detection
├── data/
│   ├── train/                     # Training data
│   └── test/                      # Test data
└── models/
    ├── model_P-1.pkl              # Trained models
    ├── model_M-1.pkl
    ├── model_E-1.pkl
    └── threshold_*.json           # Detection thresholds
```

---

## 🎓 Training Custom Models

```bash
# Train on your data
python train.py --channel MY_SENSOR --window-size 50 --epochs 50

# Parameters:
#   --channel: Sensor name (e.g., P-1, temperature)
#   --window-size: Past values to consider (default: 50)
#   --epochs: Training iterations (default: 50)
#   --threshold-k: Sensitivity (default: 3.0)
```

---

## 📊 Data Format

Your data should be:
- **Time series** numerical values
- **Saved as** `.npy`, `.csv`, or `.txt`
- **Single column** per file

**Example: Convert CSV to NPY**
```python
import numpy as np
import pandas as pd

df = pd.read_csv('sensor_data.csv')
data = df['temperature'].values
np.save('data/train/my_sensor_train.npy', data[:8000])
np.save('data/test/my_sensor_test.npy', data[8000:])
```

---

## 🔧 Technology Stack

- **Frontend:** Streamlit
- **Visualization:** Plotly
- **ML Framework:** Pure NumPy (Python 3.14 compatible)
- **Data Processing:** Pandas, NumPy
- **Deployment:** Streamlit Cloud + GitHub Actions

---

## 🌐 Deployment

### Streamlit Cloud (Free)

1. **Fork this repository**
2. **Go to** [share.streamlit.io](https://share.streamlit.io)
3. **New app** → Select your repo
4. **Main file:** `app.py`
5. **Deploy!**

**GitHub Actions** automatically pings your app every 10 minutes to prevent sleep.

### Update App URL

After deployment, edit `.github/workflows/keep-alive.yml`:
```yaml
APP_URL="https://your-actual-app-url.streamlit.app"
```

---

## 🎯 Use Cases

- 🛰️ **Spacecraft Monitoring** - Detect sensor anomalies in satellites
- 🚀 **Launch Systems** - Monitor pre-launch telemetry
- 🔬 **Research** - Study spacecraft behavior patterns
- 📚 **Education** - Learn anomaly detection techniques
- 🏢 **Industrial IoT** - Monitor equipment sensors

---

## 📈 Performance

- **Processing Speed:** 
  - Normal: ~100k points/sec
  - Fast mode: ~200k points/sec (2x)
- **Model Size:** <1 MB per channel
- **Memory Usage:** <100 MB
- **Startup Time:** <5 seconds

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📄 License

MIT License - Free for educational and commercial use.

---

## 🙏 Acknowledgments

- **NASA SMAP/MSL** - Telemetry data format
- **Streamlit** - Web framework
- **Plotly** - Interactive visualizations

---

## 📧 Contact

**Developer:** Vikram  
**GitHub:** [@Vikram739](https://github.com/Vikram739)  
**Live Demo:** [orbinaasense.streamlit.app](https://orbinaasense.streamlit.app)

---

## ⭐ Star This Project

If you find this useful, give it a star! ⭐

---

**Built with 💙 for spacecraft safety**

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
