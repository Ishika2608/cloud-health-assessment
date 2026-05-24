import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, confusion_matrix
)


def generate_dataset(n: int = 2000, seed: int = 42):
    rng = np.random.default_rng(seed)

    cpu     = rng.uniform(10, 100, n)
    memory  = rng.uniform(10, 100, n)
    disk_io = rng.uniform(5,  80,  n)
    network = rng.uniform(5,  90,  n)

    future_cpu    = np.clip(cpu    + rng.normal(5, 8, n), 0, 100)
    future_memory = np.clip(memory + rng.normal(3, 6, n), 0, 100)

    labels = []
    for fc, fm in zip(future_cpu, future_memory):
        if fc > 85 or fm > 85:
            labels.append(2)          # Critical
        elif fc > 70 or fm > 70:
            labels.append(1)          # Warning
        else:
            labels.append(0)          # Normal

    X = np.column_stack([cpu, memory, disk_io, network])
    y_reg_cpu = future_cpu
    y_reg_mem = future_memory
    y_clf = np.array(labels)

    return X, y_reg_cpu, y_reg_mem, y_clf


class CloudHealthModel:
    LABEL_MAP = {0: "Normal", 1: "Warning", 2: "Critical"}
    RISK_MAP  = {0: "Low",    1: "Medium",  2: "High"}

    def __init__(self):
        self.scaler = StandardScaler()

        self.rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_reg = GradientBoostingRegressor(n_estimators=100, random_state=42)

        dt  = DecisionTreeClassifier(max_depth=6, random_state=42)
        rf  = RandomForestClassifier(n_estimators=100, random_state=42)
        lr  = LogisticRegression(max_iter=1000, random_state=42)
        self.voting_clf = VotingClassifier(
            estimators=[("dt", dt), ("rf", rf), ("lr", lr)],
            voting="hard"
        )

        self._metrics = {}
        self._trained = False

    def train(self):
        X, y_cpu, y_mem, y_clf = generate_dataset(n=2000)

        X_scaled = self.scaler.fit_transform(X)

        (X_tr, X_te,
         yc_tr, yc_te,
         ym_tr, ym_te,
         yl_tr, yl_te) = train_test_split(
            X_scaled, y_cpu, y_mem, y_clf,
            test_size=0.2, random_state=42
        )

        # Module 1
        self.rf_reg.fit(X_tr, yc_tr)
        self.gb_reg.fit(X_tr, yc_tr)

        pred_cpu_rf = self.rf_reg.predict(X_te)
        pred_cpu_gb = self.gb_reg.predict(X_te)
        pred_cpu    = (pred_cpu_rf + pred_cpu_gb) / 2

        # Module 2 
        X_tr_aug = np.column_stack([X_tr, self.rf_reg.predict(X_tr)])
        X_te_aug = np.column_stack([X_te, pred_cpu])

        self.voting_clf.fit(X_tr_aug, yl_tr)
        pred_labels = self.voting_clf.predict(X_te_aug)

        # Metrics
        mae  = mean_absolute_error(yc_te, pred_cpu)
        rmse = np.sqrt(mean_squared_error(yc_te, pred_cpu))
        acc  = accuracy_score(yl_te, pred_labels)
        cm   = confusion_matrix(yl_te, pred_labels).tolist()

        self._metrics = {
            "regression": {
                "mae":  round(mae,  3),
                "rmse": round(rmse, 3),
                "description": "Ensemble avg of RF + GB Regressors on CPU prediction"
            },
            "classification": {
                "accuracy": round(acc, 3),
                "confusion_matrix": cm,
                "classes": ["Normal", "Warning", "Critical"],
                "description": "Hard-voting ensemble: Decision Tree + RF + Logistic Regression"
            }
        }
        self._trained = True
        print(f"[Model] Training done | Regression MAE={mae:.2f} RMSE={rmse:.2f} | Classification Accuracy={acc:.3f}")

    # ── Prediction ────────────────────────────
    def predict(self, cpu: float, memory: float, disk_io: float, network: float) -> dict:
        if not self._trained:
            raise RuntimeError("Model not trained yet.")

        x = np.array([[cpu, memory, disk_io, network]])
        x_scaled = self.scaler.transform(x)

        # Module 1 – regression ensemble
        pred_cpu_rf = self.rf_reg.predict(x_scaled)[0]
        pred_cpu_gb = self.gb_reg.predict(x_scaled)[0]
        pred_cpu    = (pred_cpu_rf + pred_cpu_gb) / 2

        pred_mem = float(np.clip(memory + np.random.normal(3, 4), 0, 100))

        # Module 2 – classification
        x_aug   = np.column_stack([x_scaled, [[pred_cpu]]])
        status_id = int(self.voting_clf.predict(x_aug)[0])

        # Individual model votes (for display)
        votes = {}
        for name, clf in self.voting_clf.named_estimators_.items():
            votes[name] = self.LABEL_MAP[int(clf.predict(x_aug)[0])]

        return {
            "predicted_cpu":    round(float(pred_cpu), 1),
            "predicted_memory": round(float(pred_mem), 1),
            "health_status":    self.LABEL_MAP[status_id],
            "risk_level":       self.RISK_MAP[status_id],
            "votes": {
                "decision_tree":     votes.get("dt", "N/A"),
                "random_forest":     votes.get("rf", "N/A"),
                "logistic_regression": votes.get("lr", "N/A"),
            }
        }

    def get_metrics(self) -> dict:
        return self._metrics
