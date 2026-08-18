"""
app.py - Real-Time Log Anomaly Detection Web Application Dashboard.
"""

import os
import sys
import json
import time
import threading
import logging
import joblib
from flask import Flask, render_template, request, jsonify, Response

from utils import load_config, parse_log_line, send_to_slack, generate_synthetic_logs
from train import train_anomaly_model
from monitor import tail_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = Flask(__name__)

# Global application state
CONFIG = load_config("config.yaml")
MODEL_PATH = CONFIG.get("model", {}).get("path", "model.joblib")
LIVE_LOG_PATH = CONFIG.get("logging", {}).get("monitor_log_path", "live.log")

MODEL_PIPELINE = None
SIMULATOR_RUNNING = True
SIMULATOR_THREAD = None


def ensure_model_loaded():
    """Ensure trained model is loaded into memory, training a baseline model if missing."""
    global MODEL_PIPELINE
    if not os.path.exists(MODEL_PATH):
        logging.info(f"Model artifact '{MODEL_PATH}' not found. Training baseline model...")
        train_log = CONFIG.get("logging", {}).get("train_log_path", "training_logs.log")
        generate_synthetic_logs(train_log, num_lines=1000, anomaly_ratio=0.05)
        MODEL_PIPELINE = train_anomaly_model(train_log, MODEL_PATH, CONFIG)
    else:
        try:
            MODEL_PIPELINE = joblib.load(MODEL_PATH)
            logging.info(f"Loaded existing anomaly detection model from '{MODEL_PATH}'.")
        except Exception as e:
            logging.error(f"Failed to load model file '{MODEL_PATH}': {e}")


def background_log_simulator():
    """Background thread generating live synthetic log stream."""
    import random
    import datetime

    normal_templates = [
        '192.168.1.{ip} - - [{ts}] "GET /index.html HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /api/v1/users HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "POST /api/v1/orders HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /static/bundle.js HTTP/1.1" 304 0 "-" "Mozilla/5.0"',
        '{ts_sys} [INFO] DB Connection pool health check OK (active: {ms}ms)',
    ]

    anomaly_templates = [
        '10.0.0.99 - - [{ts}] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 4096 "-" "Python-urllib/3.8"',
        '10.0.0.99 - - [{ts}] "GET /../../etc/passwd HTTP/1.1" 403 512 "-" "CustomScanner/1.0"',
        '10.0.0.99 - - [{ts}] "POST /admin/upload HTTP/1.1" 500 99999999 "-" "ExploitBot"',
        '{ts_sys} [CRITICAL] FATAL: OutOfMemoryError in worker thread pool! Heap dump generated.',
        '{ts_sys} [ERROR] Unauthorized access attempt detected from IP 192.168.99.99 on port 22',
    ]

    os.makedirs(os.path.dirname(os.path.abspath(LIVE_LOG_PATH)) or ".", exist_ok=True)

    while True:
        if SIMULATOR_RUNNING:
            now = datetime.datetime.now()
            ts = now.strftime("%d/%b/%Y:%H:%M:%S +0000")
            ts_sys = now.strftime("%Y-%m-%d %H:%M:%S")

            ip_val = random.randint(1, 250)
            bytes_val = random.randint(200, 4500)
            ms_val = random.randint(5, 150)

            is_anomaly = random.random() < 0.08
            template = random.choice(anomaly_templates) if is_anomaly else random.choice(normal_templates)
            line = template.format(ts=ts, ts_sys=ts_sys, ip=ip_val, bytes=bytes_val, ms=ms_val)

            with open(LIVE_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()

        time.sleep(0.3)


@app.route("/")
def index():
    """Main dashboard interface."""
    slack_url = CONFIG.get("slack", {}).get("webhook_url", "")
    slack_cooldown = CONFIG.get("slack", {}).get("cooldown_seconds", 10)
    return render_template("index.html", slack_webhook_url=slack_url, slack_cooldown=slack_cooldown)


@app.route("/api/stream")
def sse_stream():
    """
    Server-Sent Events (SSE) streaming endpoint.
    Continuously yields real-time log lines and ML predictions to the web dashboard.
    """
    def event_generator():
        global MODEL_PIPELINE
        poll_interval = CONFIG.get("logging", {}).get("tail_poll_interval", 0.1)

        for line in tail_file(LIVE_LOG_PATH, seek_end=True, poll_interval=poll_interval):
            line_clean = line.strip()
            if not line_clean:
                continue

            try:
                parsed = parse_log_line(line_clean)
                prediction = int(MODEL_PIPELINE.predict([line_clean])[0]) if MODEL_PIPELINE else 1
                score = float(MODEL_PIPELINE.decision_function([line_clean])[0]) if MODEL_PIPELINE else 0.0

                if prediction == -1:
                    webhook_url = CONFIG.get("slack", {}).get("webhook_url", "")
                    send_to_slack(webhook_url, line_clean, parsed["timestamp"], score, CONFIG)

                payload = {
                    "raw_line": line_clean,
                    "parsed": parsed,
                    "prediction": prediction,
                    "score": score,
                }

                yield f"data: {json.dumps(payload)}\n\n"
            except Exception as e:
                logging.error(f"Error in SSE log processing: {e}")

    return Response(event_generator(), mimetype="text/event-stream")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Interactive Sandbox endpoint to evaluate any raw log string."""
    data = request.json or {}
    log_line = data.get("log_line", "").strip()

    if not log_line:
        return jsonify({"error": "No log_line provided"}), 400

    try:
        parsed = parse_log_line(log_line)
        prediction = int(MODEL_PIPELINE.predict([log_line])[0]) if MODEL_PIPELINE else 1
        score = float(MODEL_PIPELINE.decision_function([log_line])[0]) if MODEL_PIPELINE else 0.0

        return jsonify({
            "log_line": log_line,
            "parsed": parsed,
            "prediction": prediction,
            "score": score,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    """Trigger ML model retraining with custom hyperparameters."""
    global MODEL_PIPELINE
    data = request.json or {}

    contamination = float(data.get("contamination", 0.05))
    n_estimators = int(data.get("n_estimators", 100))
    num_samples = int(data.get("num_samples", 1000))

    CONFIG["model"]["contamination"] = contamination
    CONFIG["model"]["n_estimators"] = n_estimators

    train_log = CONFIG.get("logging", {}).get("train_log_path", "training_logs.log")
    generate_synthetic_logs(train_log, num_lines=num_samples, anomaly_ratio=contamination)

    try:
        MODEL_PIPELINE = train_anomaly_model(train_log, MODEL_PATH, CONFIG)
        return jsonify({
            "status": "success",
            "message": "Model retrained successfully",
            "summary": {
                "num_samples": num_samples,
                "normal_entries": int(num_samples * (1 - contamination)),
                "detected_anomalies": int(num_samples * contamination),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/simulator/toggle", methods=["POST"])
def toggle_simulator():
    """Start or stop the background log stream generator."""
    global SIMULATOR_RUNNING
    SIMULATOR_RUNNING = not SIMULATOR_RUNNING
    return jsonify({"running": SIMULATOR_RUNNING})


@app.route("/api/config", methods=["GET", "POST"])
def handle_config():
    """Get or update application settings."""
    global CONFIG
    if request.method == "POST":
        data = request.json or {}
        if "slack" in data:
            CONFIG.setdefault("slack", {}).update(data["slack"])
        return jsonify({"status": "updated", "config": CONFIG})
    return jsonify(CONFIG)


@app.route("/api/slack/test", methods=["POST"])
def api_slack_test():
    """Trigger a test alert to Slack Webhook."""
    webhook_url = CONFIG.get("slack", {}).get("webhook_url", "")
    sample_line = '10.0.0.99 - - [19/Aug/2026] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 4096'
    success = send_to_slack(webhook_url, sample_line, score=-0.085, config=CONFIG)
    return jsonify({"success": success, "message": "Test alert triggered." if success else "Slack notification failed."})


def start_server(port=5050):
    """Start Flask web server and background stream simulator."""
    ensure_model_loaded()

    global SIMULATOR_THREAD
    SIMULATOR_THREAD = threading.Thread(target=background_log_simulator, daemon=True)
    SIMULATOR_THREAD.start()

    logging.info(f"Log Sentinel Web Dashboard running at http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    start_server(5050)

