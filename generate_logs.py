"""
generate_logs.py - Continuous synthetic log generator for live testing of monitor.py.
"""

import time
import argparse
import logging
import random
import datetime
from utils import generate_synthetic_logs

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def stream_logs(output_file, interval=0.2, anomaly_prob=0.08):
    """
    Stream continuous synthetic logs to output_file.
    """
    logging.info(f"Streaming continuous logs into '{output_file}' (interval={interval}s, anomaly_prob={anomaly_prob}). Press Ctrl+C to stop.")

    normal_templates = [
        '192.168.1.{ip} - - [{ts}] "GET /index.html HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /api/v1/products HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "POST /api/v1/checkout HTTP/1.1" 200 {bytes} "-" "Mozilla/5.0"',
        '192.168.1.{ip} - - [{ts}] "GET /static/app.js HTTP/1.1" 304 0 "-" "Mozilla/5.0"',
        '{ts_sys} [INFO] Metric collection completed successfully in {ms}ms',
    ]

    anomaly_templates = [
        '10.0.0.99 - - [{ts}] "GET /api/search?q=\' OR 1=1 -- HTTP/1.1" 500 4096 "-" "Python-urllib/3.8"',
        '10.0.0.99 - - [{ts}] "GET /../../etc/shadow HTTP/1.1" 403 512 "-" "ExploitScanner/2.0"',
        '10.0.0.99 - - [{ts}] "POST /admin/upload HTTP/1.1" 500 99999999 "-" "DataExfiltrationBot"',
        '{ts_sys} [CRITICAL] OutOfMemoryError: Java heap space crash in payment-service',
        '{ts_sys} [ERROR] Database connection pool exhausted! 500 requests queued.',
    ]

    count = 0
    with open(output_file, "a", encoding="utf-8") as f:
        while True:
            now = datetime.datetime.now()
            ts = now.strftime("%d/%b/%Y:%H:%M:%S +0000")
            ts_sys = now.strftime("%Y-%m-%d %H:%M:%S")

            ip_val = random.randint(1, 250)
            bytes_val = random.randint(200, 4000)
            ms_val = random.randint(5, 200)

            is_anomaly = random.random() < anomaly_prob
            if is_anomaly:
                template = random.choice(anomaly_templates)
                tag = "[ANOMALY]"
            else:
                template = random.choice(normal_templates)
                tag = "[NORMAL]"

            line = template.format(ts=ts, ts_sys=ts_sys, ip=ip_val, bytes=bytes_val, ms=ms_val)
            f.write(line + "\n")
            f.flush()

            count += 1
            if count % 10 == 0:
                logging.info(f"Wrote {count} lines to {output_file} (last line: {tag} {line[:70]}...)")

            time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Live Synthetic Log Stream Generator")
    parser.add_argument("--output", type=str, default="live.log", help="Target log file to stream into")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay in seconds between log entries")
    parser.add_argument("--anomaly-prob", type=float, default=0.08, help="Probability of generating an anomalous log line")

    args = parser.parse_args()
    try:
        stream_logs(args.output, args.interval, args.anomaly_prob)
    except KeyboardInterrupt:
        logging.info("Stream generator stopped.")


if __name__ == "__main__":
    main()
