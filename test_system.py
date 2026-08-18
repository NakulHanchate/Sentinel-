"""
test_system.py - Unit test suite for log anomaly detection pipeline & Flask web server.
"""

import os
import json
import tempfile
import unittest
import numpy as np
import joblib

from utils import load_config, parse_log_line, LogFeatureExtractor, send_to_slack, generate_synthetic_logs
from train import train_anomaly_model
from app import app, ensure_model_loaded


class TestLogAnomalySystem(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        self.train_log = os.path.join(self.temp_dir.name, "train.log")
        self.model_path = os.path.join(self.temp_dir.name, "model.joblib")

        self.config = load_config(self.config_path)

        app.config['TESTING'] = True
        self.client = app.test_client()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_parse_log_line_http(self):
        line = '192.168.1.50 - - [19/Aug/2026:10:00:00 +0000] "GET /api/v1/users HTTP/1.1" 200 1024 "-" "Mozilla"'
        parsed = parse_log_line(line)

        self.assertEqual(parsed["ip"], "192.168.1.50")
        self.assertEqual(parsed["method"], "GET")
        self.assertEqual(parsed["path"], "/api/v1/users")
        self.assertEqual(parsed["status_code"], 200)
        self.assertEqual(parsed["bytes_sent"], 1024)
        self.assertEqual(parsed["log_level"], "INFO")

    def test_parse_log_line_syslog(self):
        line = "2026-08-19 10:00:00 [ERROR] OutOfMemoryError in worker thread pool"
        parsed = parse_log_line(line)

        self.assertEqual(parsed["log_level"], "ERROR")
        self.assertEqual(parsed["log_level_code"], 4)
        self.assertIn("OutOfMemoryError", parsed["message"])

    def test_feature_extractor(self):
        logs = [
            '192.168.1.1 - - [19/Aug/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 500',
            '10.0.0.99 - - [19/Aug/2026:10:00:00 +0000] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 4096',
        ]
        extractor = LogFeatureExtractor(tfidf_max_features=20, scale_numeric=True)
        extractor.fit(logs)
        features = extractor.transform(logs)

        self.assertIsInstance(features, np.ndarray)
        self.assertEqual(features.shape[0], 2)
        self.assertGreater(features.shape[1], 6)

    def test_end_to_end_training_and_inference(self):
        generate_synthetic_logs(self.train_log, num_lines=200, anomaly_ratio=0.05)
        pipeline = train_anomaly_model(self.train_log, self.model_path, self.config)

        self.assertTrue(os.path.exists(self.model_path))

        loaded_pipeline = joblib.load(self.model_path)

        normal_line = '192.168.1.1 - - [19/Aug/2026:10:00:00 +0000] "GET /index.html HTTP/1.1" 200 500'
        anomaly_line = '10.0.0.99 - - [19/Aug/2026:10:00:00 +0000] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 99999999'

        normal_pred = loaded_pipeline.predict([normal_line])[0]
        anomaly_pred = loaded_pipeline.predict([anomaly_line])[0]

        self.assertEqual(normal_pred, 1)
        self.assertEqual(anomaly_pred, -1)

    def test_flask_api_predict(self):
        ensure_model_loaded()
        response = self.client.post(
            "/api/predict",
            data=json.dumps({"log_line": '192.168.1.1 - - [19/Aug/2026] "GET /index.html HTTP/1.1" 200 500'}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("prediction", data)
        self.assertIn("score", data)

    def test_flask_simulator_toggle(self):
        response = self.client.post("/api/simulator/toggle")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("running", data)


if __name__ == "__main__":
    unittest.main()
