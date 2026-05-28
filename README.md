# ☁️ Health Assessment and Performance Prediction in Cloud Environments using Ensemble Modeling

A mini project by **Anupriya Bhoyar**, **Dhanashree Khamitkar**, **Ishika Singh**
T.Y. B.Tech CSE (Data Science) | G H Raisoni College of Engineering and Management, Pune | 2025-26

---

## 📌 About the Project

This system is a stateless, full-stack cloud monitoring solution that combines **performance prediction** and **health assessment** using ensemble machine learning models.

It collects cloud metrics (CPU, Memory, Disk I/O, Network), predicts future resource usage, and classifies system health as **Normal**, **Warning**, or **Critical** — all in real time without relying on historical data storage.

---

## 🗂️ Project Structure

```
cloud-health-assessment/
├── backend/
│   ├── app.py              ← Flask REST API
│   ├── model.py            ← ML models (Module 1 + Module 2)
│   └── requirements.txt
├── frontend/
│   └── index.html          ← Dashboard (open in browser)
├── start.bat               ← One-click Windows launcher
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask 2.x |
| Machine Learning | scikit-learn, pandas, numpy |
| Frontend | HTML, CSS, JavaScript |
| API Testing | Postman |
| Version Control | Git & GitHub |

---

## 🧠 ML Architecture

### Module 1 — Performance Prediction (Regression)
- Random Forest Regressor + Gradient Boosting Regressor
- Ensemble averaging for final CPU & Memory prediction

### Module 2 — Health Classification
- Decision Tree + Random Forest + Logistic Regression
- Hard voting classifier → outputs Normal / Warning / Critical

---

## 🚀 Quick Start (Windows)

### Option A — Double-click
Just run `start.bat` — it installs dependencies and starts the server.

### Option B — Manual

```bash
# Step 1: Install dependencies
pip install flask flask-cors scikit-learn numpy pandas

# Step 2: Start backend
cd backend
python app.py

# Step 3: Open frontend
# Open frontend/index.html in your browser
```

---

## 🔗 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Check if API is running |
| POST | `/api/predict` | Predict CPU, Memory & Health Status |
| GET | `/api/metrics` | Get model evaluation metrics |
| GET | `/api/simulate` | Generate synthetic prediction data |

### Sample POST `/api/predict`
```json
{
  "cpu": 75,
  "memory": 80,
  "disk_io": 50,
  "network": 60
}
```

### Sample Response
```json
{
  "predicted_cpu": 78.4,
  "predicted_memory": 83.1,
  "health_status": "Warning",
  "risk_level": "Medium",
  "votes": {
    "decision_tree": "Warning",
    "random_forest": "Warning",
    "logistic_regression": "Critical"
  }
}
```

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| Classification Accuracy | ~83.5% |
| Regression MAE | ~3.21 |
| Regression RMSE | ~5.4 |
| Test Cases Passed | 8/8 ✅ |

---

## 👩‍💻 Contributors

| Name | Roll No | Contribution |
|---|---|---|
| Anupriya Bhoyar | 10 | Frontend, Reporting & Documentation |
| Dhanashree Khamitkar | 14 | Backend & API |
| Ishika Singh | 21 | Machine Learning Model |

---

## 🏫 Institution

**G H Raisoni College of Engineering and Management**
Wagholi, Pune 412207
Department of CSE (Data Science)
