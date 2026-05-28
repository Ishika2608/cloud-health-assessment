# ☁ CloudPulse — Health Assessment & Performance Prediction
## Project Structure
```
cloud_health/
├── backend/
│   ├── app.py          ← Flask REST API
│   ├── model.py        ← ML models (Module 1 + Module 2)
│   └── requirements.txt
├── frontend/
│   └── index.html      ← Dashboard (open in browser)
├── start.bat           ← One-click Windows launcher
└── README.md
```
## Quick Start (Windows)

### Option A — Double-click
Just run `start.bat` — it installs dependencies and starts the server.

### Option B — Manual
```cmd
# Step 1: Install dependencies
pip install flask flask-cors scikit-learn numpy pandas

# Step 2: Start backend
cd backend
python app.py

# Step 3: Open frontend
# Open frontend\index.html in your browser
```

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | `/api/health` | Server health check |
| POST | `/api/predict` | Run prediction |
| GET | `/api/metrics` | Model evaluation metrics |
| GET | `/api/simulate` | Generate 20 random predictions |

### POST /api/predict — Example

**Request:**
```json
{
  "cpu":     75,
  "memory":  80,
  "disk_io": 45,
  "network": 60
}
```

**Response:**
```json
{
  "predicted_cpu":    82.4,
  "predicted_memory": 83.1,
  "health_status":   "Critical",
  "risk_level":      "High",
  "votes": {
    "decision_tree":      "Critical",
    "random_forest":      "Critical",
    "logistic_regression":"Warning"
  }
}
```
## Architecture

```
Input Metrics (CPU, Memory, Disk I/O, Network)
        │
        ▼
┌──────────────────────────────────────┐
│  MODULE 1 — REGRESSION ENSEMBLE      │
│  RF Regressor + GB Regressor         │
│  Final = (RF_pred + GB_pred) / 2     │
│  Output: Predicted CPU, Memory       │
└──────────────────────────────────────┘
        │  Predicted values fed as features
        ▼
┌──────────────────────────────────────┐
│  MODULE 2 — CLASSIFICATION ENSEMBLE  │
│  Decision Tree + Random Forest       │
│              + Logistic Regression   │
│  Final = Majority Vote               │
│  Output: Normal / Warning / Critical │
└──────────────────────────────────────┘
        │
        ▼
   Dashboard Output
```
## Label Rules (as per spec)
- CPU > 85% → **Critical**
- CPU 70–85% → **Warning**  
- CPU < 70% → **Normal**

## Evaluation Metrics
- Regression: MAE, RMSE
- Classification: Accuracy, Confusion Matrix