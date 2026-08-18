"""
train.py - Processes historical logs, trains IsolationForest model, and serializes the pipeline.
"""

import os
import sys
import argparse
import logging
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline

from utils import load_config, LogFeatureExtractor, generate_synthetic_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def train_anomaly_model(log_file, model_output_path, config):
    """
    Train IsolationForest anomaly detection model on historical log file.
    """
    if not os.path.exists(log_file):
        logging.warning(f"Training log file '{log_file}' not found. Generating sample data...")
        generate_synthetic_logs(log_file, num_lines=1000, anomaly_ratio=0.05)

    logging.info(f"Reading training logs from '{log_file}'...")
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        log_lines = [line.strip() for line in f if line.strip()]

    if not log_lines:
        raise ValueError(f"No valid log lines found in '{log_file}'. Cannot train model.")

    logging.info(f"Loaded {len(log_lines)} log entries for training.")

    # Model parameters from config
    model_cfg = config.get("model", {})
    feat_cfg = config.get("features", {})

    contamination = model_cfg.get("contamination", 0.05)
    n_estimators = model_cfg.get("n_estimators", 100)
    random_state = model_cfg.get("random_state", 42)
    tfidf_max_features = feat_cfg.get("tfidf_max_features", 100)
    scale_numeric = feat_cfg.get("scale_numeric", True)

    logging.info(f"Configuring feature extractor (tfidf_max_features={tfidf_max_features}, scale_numeric={scale_numeric})...")
    feature_extractor = LogFeatureExtractor(
        tfidf_max_features=tfidf_max_features,
        scale_numeric=scale_numeric
    )

    logging.info(f"Configuring IsolationForest (contamination={contamination}, n_estimators={n_estimators}, random_state={random_state})...")
    isolation_forest = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1
    )

    # Construct Scikit-Learn Pipeline
    pipeline = Pipeline([
        ("feature_extractor", feature_extractor),
        ("model", isolation_forest)
    ])

    logging.info("Fitting anomaly detection pipeline...")
    pipeline.fit(log_lines)

    # Evaluate predictions on training set
    predictions = pipeline.predict(log_lines)
    scores = pipeline.decision_function(log_lines)
    anomalies_count = np.sum(predictions == -1)
    normal_count = np.sum(predictions == 1)

    logging.info(f"Training completed successfully.")
    logging.info(f"Summary: Normal entries = {normal_count}, Detected Anomalies = {anomalies_count} ({anomalies_count/len(log_lines):.2%}).")
    logging.info(f"Score range: Min={scores.min():.4f}, Max={scores.max():.4f}, Mean={scores.mean():.4f}")

    # Serialize model pipeline to disk
    os.makedirs(os.path.dirname(os.path.abspath(model_output_path)), exist_ok=True)
    joblib.dump(pipeline, model_output_path)
    logging.info(f"Serialized model pipeline saved to '{model_output_path}'.")

    return pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Log Anomaly Detection Model")
    parser.add_argument("--log-file", type=str, help="Path to historical log file for training")
    parser.add_argument("--model-path", type=str, help="Path where trained model (.joblib) will be saved")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--generate-sample-data", action="store_true", help="Force generation of synthetic training logs")

    args = parser.parse_args()
    config = load_config(args.config)

    log_file = args.log_file or config.get("logging", {}).get("train_log_path", "training_logs.log")
    model_path = args.model_path or config.get("model", {}).get("path", "model.joblib")

    if args.generate_sample_data:
        logging.info(f"Generating synthetic training log file at '{log_file}'...")
        generate_synthetic_logs(log_file, num_lines=1000, anomaly_ratio=0.05)

    try:
        train_anomaly_model(log_file, model_path, config)
    except Exception as e:
        logging.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
