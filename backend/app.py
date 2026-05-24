from flask import Flask, jsonify, request
from flask_cors import CORS
from model import CloudHealthModel
import numpy as np

app = Flask(__name__)
CORS(app)

# Initialize and train the model on startup
model = CloudHealthModel()
model.train()

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "message": "Cloud Health API is running"})

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    cpu       = float(data.get("cpu", 50))
    memory    = float(data.get("memory", 50))
    disk_io   = float(data.get("disk_io", 50))
    network   = float(data.get("network", 50))

    result = model.predict(cpu, memory, disk_io, network)
    return jsonify(result)

@app.route("/api/metrics", methods=["GET"])
def metrics():
    """Return model evaluation metrics."""
    return jsonify(model.get_metrics())

@app.route("/api/simulate", methods=["GET"])
def simulate():
    """Generate a batch of random predictions for the live chart."""
    points = []
    for _ in range(20):
        cpu     = float(np.random.uniform(30, 95))
        memory  = float(np.random.uniform(30, 95))
        disk_io = float(np.random.uniform(10, 80))
        network = float(np.random.uniform(10, 90))
        res = model.predict(cpu, memory, disk_io, network)
        res["cpu"]     = cpu
        res["memory"]  = memory
        res["disk_io"] = disk_io
        res["network"] = network
        points.append(res)
    return jsonify(points)

if __name__ == "__main__":
    print("Starting Cloud Health API on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
