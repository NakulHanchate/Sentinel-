"""
utils.py - Helper functions for parsing, feature extraction, alerting, and simulation.
"""

import os
import re
import time
import logging
import datetime
import yaml
import requests
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global state for Slack alert cooldown tracking
_last_slack_alert_time = 0.0


def load_config(config_path="config.yaml"):
    """
    Load YAML configuration settings.
    Falls back to default settings if file doesn't exist or is invalid.
    """
    default_config = {
        "slack": {
            "webhook_url": "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK",
            "enabled": True,
            "cooldown_seconds": 10,
            "channel": "#log-alerts",
            "bot_name": "Log Sentinel",
        },
        "model": {
            "path": "model.joblib",
            "contamination": 0.05,
            "n_estimators": 100,
            "random_state": 42,
        },
        "features": {
            "tfidf_max_features": 100,
            "scale_numeric": True,
        },
        "logging": {
            "train_log_path": "training_logs.log",
            "monitor_log_path": "live.log",
            "tail_poll_interval": 0.1,
        },
    }

    if not os.path.exists(config_path):
        logging.warning(f"Config file '{config_path}' not found. Using default configuration.")
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}

        # Merge with default config
        for key, value in user_config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value

        return default_config
    except Exception as e:
        logging.error(f"Error reading config file '{config_path}': {e}. Using defaults.")
        return default_config


def parse_log_line(line):
    """
    Parse raw log line into a structured dictionary.
    Supports standard HTTP access logs, Syslog, and custom/unformatted log lines.
    """
    line_str = line.strip() if isinstance(line, str) else str(line).strip()

    # Default values
    parsed = {
        "raw_line": line_str,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "ip": "127.0.0.1",
        "method": "GET",
        "path": "/",
        "status_code": 200,
        "bytes_sent": 0,
        "log_level": "INFO",
        "log_level_code": 2,
        "message": line_str,
    }

    if not line_str:
        return parsed

    # 1. Standard Nginx / Apache Access Log Format:
    # 192.168.1.1 - - [19/Aug/2026:10:00:00 +0000] "GET /api/v1/resource HTTP/1.1" 200 1024 "ref" "agent"
    combined_log_regex = r'^(\S+) \S+ \S+ \[(.*?)\] "(\S+) (\S+)\s*.*?" (\d{3}) (\d+|-)'
    match = re.match(combined_log_regex, line_str)
    if match:
        parsed["ip"] = match.group(1)
        parsed["timestamp"] = match.group(2)
        parsed["method"] = match.group(3)
        parsed["path"] = match.group(4)
        parsed["status_code"] = int(match.group(5))
        parsed["bytes_sent"] = int(match.group(6)) if match.group(6) != "-" else 0
        parsed["message"] = f"{parsed['method']} {parsed['path']} {parsed['status_code']}"

        # Assign log level based on HTTP status
        if parsed["status_code"] >= 500:
            parsed["log_level"] = "CRITICAL"
            parsed["log_level_code"] = 5
        elif parsed["status_code"] >= 400:
            parsed["log_level"] = "ERROR"
            parsed["log_level_code"] = 4
        elif parsed["status_code"] >= 300:
            parsed["log_level"] = "WARN"
            parsed["log_level_code"] = 3
        else:
            parsed["log_level"] = "INFO"
            parsed["log_level_code"] = 2
        return parsed

    # 2. Syslog / Application Log Format:
    # 2026-08-19 10:00:00 [ERROR] Database connection failed after 30000ms
    syslog_regex = r'^(\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2})\s+\[?(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]?\s+(.*)'
    match_sys = re.match(syslog_regex, line_str, re.IGNORECASE)
    if match_sys:
        parsed["timestamp"] = match_sys.group(1)
        lvl = match_sys.group(2).upper()
        parsed["log_level"] = lvl
        parsed["message"] = match_sys.group(3)

        level_map = {"DEBUG": 1, "INFO": 2, "WARN": 3, "WARNING": 3, "ERROR": 4, "CRITICAL": 5, "FATAL": 5}
        parsed["log_level_code"] = level_map.get(lvl, 2)
        if parsed["log_level_code"] >= 4:
            parsed["status_code"] = 500
        return parsed

    # 3. Fallback extraction for keywords (e.g. status codes, errors)
    status_match = re.search(r'\b(2\d\d|3\d\d|4\d\d|5\d\d)\b', line_str)
    if status_match:
        parsed["status_code"] = int(status_match.group(1))

    if any(kw in line_str.upper() for kw in ["ERROR", "EXCEPTION", "FAIL", "FATAL", "CRITICAL"]):
        parsed["log_level"] = "ERROR"
        parsed["log_level_code"] = 4
    elif any(kw in line_str.upper() for kw in ["WARN", "WARNING"]):
        parsed["log_level"] = "WARN"
        parsed["log_level_code"] = 3

    return parsed


class LogFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible feature extraction pipeline for log lines.
    Combines TF-IDF NLP text vectors with scaled numerical metrics.
    """

    def __init__(self, tfidf_max_features=100, scale_numeric=True):
        self.tfidf_max_features = tfidf_max_features
        self.scale_numeric = scale_numeric
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.tfidf_max_features,
            token_pattern=r'(?u)\b\w+\b',
            ngram_range=(1, 2),
            lowercase=True,
        )
        self.scaler = StandardScaler()

    def _extract_raw_features(self, logs):
        """Helper to extract text and numerical columns from raw logs or parsed dicts."""
        parsed_logs = []
        for item in logs:
            if isinstance(item, dict):
                parsed_logs.append(item)
            else:
                parsed_logs.append(parse_log_line(str(item)))

        texts = [p["raw_line"] for p in parsed_logs]
        numerics = []
        for p in parsed_logs:
            is_error = 1.0 if (p["status_code"] >= 400 or p["log_level_code"] >= 4) else 0.0
            is_critical = 1.0 if (p["status_code"] >= 500 or p["log_level_code"] >= 5) else 0.0
            line_len = float(len(p["raw_line"]))
            bytes_sent = float(p["bytes_sent"])
            status_code = float(p["status_code"])
            log_level_code = float(p["log_level_code"])

            numerics.append([
                status_code,
                bytes_sent,
                log_level_code,
                is_error,
                is_critical,
                line_len,
            ])

        return texts, np.array(numerics, dtype=np.float64)

    def fit(self, X, y=None):
        texts, numerics = self._extract_raw_features(X)
        self.tfidf_vectorizer.fit(texts)
        if self.scale_numeric:
            self.scaler.fit(numerics)
        return self

    def transform(self, X):
        texts, numerics = self._extract_raw_features(X)
        text_features = self.tfidf_vectorizer.transform(texts).toarray()

        if self.scale_numeric:
            num_features = self.scaler.transform(numerics)
        else:
            num_features = numerics

        return np.hstack([num_features, text_features])


def send_to_slack(webhook_url, log_line, timestamp=None, score=None, config=None):
    """
    Send real-time alert to a Slack Webhook.
    Includes alert cooldown enforcement to prevent Slack throttling.
    """
    global _last_slack_alert_time

    if config and not config.get("slack", {}).get("enabled", True):
        logging.info("[SLACK ALERT SKIPPED] Slack notifications disabled in config.")
        return False

    cooldown = config.get("slack", {}).get("cooldown_seconds", 10) if config else 10
    now = time.time()
    if (now - _last_slack_alert_time) < cooldown:
        logging.info(f"[SLACK ALERT COOLDOWN] Skipped sending to Slack (cooldown active: {cooldown}s).")
        return False

    ts_str = timestamp or datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    score_str = f"{score:.4f}" if score is not None else "N/A"

    payload = {
        "text": f"🚨 *Log Anomaly Detected!*",
        "attachments": [
            {
                "color": "#FF0000",
                "fields": [
                    {"title": "Timestamp", "value": ts_str, "short": True},
                    {"title": "Anomaly Score", "value": score_str, "short": True},
                    {"title": "Log Entry", "value": f"```{log_line}```", "short": False},
                ],
            }
        ],
    }

    is_dummy_url = not webhook_url or "YOUR/SLACK/WEBHOOK" in webhook_url or "dummy" in webhook_url.lower()

    if is_dummy_url:
        logging.warning(f"[SLACK ALERT (DRY-RUN)] Webhook URL not configured. Payload:\n{payload}")
        _last_slack_alert_time = now
        return True

    try:
        response = requests.post(webhook_url, json=payload, headers={"Content-Type": "application/json"}, timeout=5)
        if response.status_code == 200:
            logging.info("Successfully sent alert to Slack.")
            _last_slack_alert_time = now
            return True
        else:
            logging.error(f"Failed to send Slack alert. HTTP {response.status_code}: {response.text}")
            return False
    except Exception as e:
        logging.error(f"Exception encountered while sending Slack alert: {e}")
        return False


def generate_synthetic_logs(file_path, num_lines=500, anomaly_ratio=0.05):
    """
    Generate synthetic log entries (normal web traffic mixed with anomalies) for testing.
    """
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

    normal_templates = [
        '192.168.1.{ip} - - [{ts}] "GET /index.html HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /api/v1/products HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "POST /api/v1/login HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /static/style.css HTTP/1.1" 304 0 "-" "Mozilla/5.0"',
        '{ts_sys} [INFO] Background job completed successfully in {ms}ms',
    ]

    anomaly_templates = [
        '10.0.0.99 - - [{ts}] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 4096 "-" "Python-urllib/3.8"',
        '10.0.0.99 - - [{ts}] "GET /../../etc/passwd HTTP/1.1" 403 512 "-" "CustomScanner/1.0"',
        '10.0.0.99 - - [{ts}] "POST /admin/upload HTTP/1.1" 500 99999999 "-" "ExploitBot"',
        '{ts_sys} [CRITICAL] FATAL: OutOfMemoryError in worker thread pool! Heap dump generated.',
        '{ts_sys} [ERROR] Unauthorized access attempt detected from IP 192.168.99.99 on port 22',
    ]

    num_anomalies = int(num_lines * anomaly_ratio)
    num_normals = num_lines - num_anomalies

    lines = []
    base_time = datetime.datetime.now() - datetime.timedelta(hours=2)

    for i in range(num_lines):
        curr_time = base_time + datetime.timedelta(seconds=i * 2)
        ts = curr_time.strftime("%d/%b/%Y:%H:%M:%S +0000")
        ts_sys = curr_time.strftime("%Y-%m-%d %H:%M:%S")
        ip_last = (i % 200) + 1
        bytes_val = (i * 37) % 4000 + 200
        ms_val = (i * 13) % 250 + 10

        # Decide normal vs anomaly
        if i > 0 and (i % max(1, (num_lines // max(1, num_anomalies)))) == 0:
            template = np.random.choice(anomaly_templates)
        else:
            template = np.random.choice(normal_templates)

        line = template.format(ts=ts, ts_sys=ts_sys, ip=ip_last, bytes=bytes_val, ms=ms_val)
        lines.append(line + "\n")

    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    logging.info(f"Generated {num_lines} synthetic log lines in '{file_path}' (approx {num_anomalies} anomalies).")
