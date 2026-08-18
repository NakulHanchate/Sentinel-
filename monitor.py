"""
monitor.py - Continuously tails live log files, runs real-time ML anomaly inference, and sends Slack alerts.
"""

import os
import sys
import time
import argparse
import logging
import signal
import joblib

from utils import load_config, parse_log_line, send_to_slack, generate_synthetic_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Global flag for graceful shutdown
RUNNING = True


def signal_handler(sig, frame):
    global RUNNING
    logging.info("Shutdown signal received. Stopping log monitor...")
    RUNNING = False


def tail_file(filepath, seek_end=True, poll_interval=0.1):
    """
    Efficient generator that continuously yields new lines from a target log file (like tail -f).
    Handles file rotation and file truncation automatically without lagging.
    """
    while RUNNING and not os.path.exists(filepath):
        logging.info(f"Waiting for target log file '{filepath}' to be created...")
        time.sleep(1.0)

    while RUNNING:
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                if seek_end:
                    f.seek(0, os.SEEK_END)

                while RUNNING:
                    line = f.readline()
                    if line:
                        yield line
                    else:
                        # Handle file rotation / truncation check
                        if os.path.exists(filepath):
                            try:
                                if os.path.getsize(filepath) < f.tell():
                                    logging.warning(f"Log file '{filepath}' truncated or rotated. Resetting position...")
                                    break
                            except OSError:
                                pass
                        time.sleep(poll_interval)
                seek_end = False  # After reopening, read from start of new file
        except (OSError, IOError) as e:
            if RUNNING:
                logging.error(f"Error accessing file '{filepath}': {e}. Retrying in 1s...")
                time.sleep(1.0)


def start_monitoring(log_file, model_path, config, dry_run=False, seek_end=True):
    """
    Main monitoring loop: streams lines, executes inference, triggers alerts.
    """
    if not os.path.exists(model_path):
        logging.error(f"Model file '{model_path}' not found! Please run 'python train.py' first.")
        sys.exit(1)

    logging.info(f"Loading anomaly detection model from '{model_path}'...")
    try:
        model_pipeline = joblib.load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model file '{model_path}': {e}")
        sys.exit(1)

    webhook_url = config.get("slack", {}).get("webhook_url", "")
    poll_interval = config.get("logging", {}).get("tail_poll_interval", 0.1)

    logging.info(f"Started real-time log monitoring on '{log_file}'. Press Ctrl+C to stop.")
    if dry_run:
        logging.info("Running in DRY-RUN mode. Slack alerts will be logged to console only.")

    processed_count = 0
    anomaly_count = 0
    start_time = time.time()

    for line in tail_file(log_file, seek_end=seek_end, poll_interval=poll_interval):
        if not RUNNING:
            break

        line_clean = line.strip()
        if not line_clean:
            continue

        processed_count += 1

        try:
            # 1. Parse line
            parsed = parse_log_line(line_clean)

            # 2. Run inference
            # Predict returns 1 for normal, -1 for anomaly
            pred = model_pipeline.predict([line_clean])[0]
            # decision_function returns anomaly score (lower / negative values indicate anomalies)
            score = model_pipeline.decision_function([line_clean])[0]

            # 3. Handle anomaly detection
            if pred == -1:
                anomaly_count += 1
                logging.warning(
                    f"⚠️ ANOMALY DETECTED [Score: {score:.4f}] | "
                    f"Status: {parsed['status_code']} | "
                    f"Level: {parsed['log_level']} | "
                    f"Raw: {line_clean[:120]}..."
                )

                # Send Slack alert
                if not dry_run:
                    send_to_slack(
                        webhook_url=webhook_url,
                        log_line=line_clean,
                        timestamp=parsed["timestamp"],
                        score=score,
                        config=config,
                    )
                else:
                    logging.info(f"[DRY-RUN ALERT] Anomaly payload prepared for line: '{line_clean[:80]}'")

        except Exception as e:
            # Robust exception handling: don't crash loop on single malformed line
            logging.error(f"Error processing log line '{line_clean[:80]}': {e}")

        # Periodic status heartbeat
        if processed_count % 1000 == 0:
            elapsed = time.time() - start_time
            rate = processed_count / elapsed if elapsed > 0 else 0
            logging.info(f"[HEARTBEAT] Processed {processed_count} lines ({rate:.1f} lines/sec). Total anomalies detected: {anomaly_count}.")

    logging.info(f"Monitoring stopped. Summary: {processed_count} lines monitored, {anomaly_count} anomalies flagged.")


def main():
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(description="Real-Time Log Anomaly Detection Monitor")
    parser.add_argument("--log-file", type=str, help="Path to live log file to monitor")
    parser.add_argument("--model-path", type=str, help="Path to trained model (.joblib)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--dry-run", action="store_true", help="Log alerts locally without posting to Slack")
    parser.add_argument("--from-beginning", action="store_true", help="Read existing log file contents from beginning rather than tailing end")

    args = parser.parse_args()
    config = load_config(args.config)

    log_file = args.log_file or config.get("logging", {}).get("monitor_log_path", "live.log")
    model_path = args.model_path or config.get("model", {}).get("path", "model.joblib")

    seek_end = not args.from_beginning

    start_monitoring(
        log_file=log_file,
        model_path=model_path,
        config=config,
        dry_run=args.dry_run,
        seek_end=seek_end,
    )


if __name__ == "__main__":
    main()
