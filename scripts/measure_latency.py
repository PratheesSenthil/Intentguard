"""
scripts/measure_latency.py
==========================
Benchmark latency of the FastAPI POST /validate endpoint on realistic payloads.

Performs:
1. Model loading & 20 warm-up requests.
2. 200+ timed requests using distinct realistic payloads extracted from tests/*.json.
3. Statistical analysis: median, 95th percentile, mean, min, max (in milliseconds).
4. Payload size reporting.
"""

import sys
import os
import json
import time
import numpy as np
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure intentguard is in python path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from intentguard.api import app
from intentguard.embedder import _get_model

def load_payloads():
    test_dir = REPO_ROOT / "tests"
    json_files = [
        test_dir / "labeled_ecommerce_set.json",
        test_dir / "heldout_ecommerce_set.json",
        test_dir / "fresh_ecommerce_set.json",
        test_dir / "fresh2_ecommerce_set.json",
    ]
    payloads = []
    for jf in json_files:
        if not jf.exists():
            continue
        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)
            for case in data.get("cases", []):
                payload = {
                    "user_message": case["user_message"],
                    "api_calls": case["api_calls"],
                    "trusted_values": case.get("trusted_values", []),
                    "audit": False,  # Isolate pipeline compute from file disk logging
                    "grounding": True,
                }
                payloads.append(payload)
    return payloads

def run_benchmark():
    print("Loading test payloads from tests/*.json...")
    payloads = load_payloads()
    print(f"Loaded {len(payloads)} unique test cases.")

    # 1. Warm-up & Model verification
    print("Pre-loading sentence-transformers model...")
    model = _get_model()
    print(f"Model loaded: {model.__class__.__name__}")

    client = TestClient(app)

    print("Running 20 warm-up requests...")
    for i in range(20):
        p = payloads[i % len(payloads)]
        res = client.post("/validate", json=p)
        assert res.status_code == 200, f"Warm-up failed: {res.text}"

    print("Warm-up complete. Model is fully warm.\n")

    # 2. Timed benchmark run (at least 200 requests)
    target_count = max(200, len(payloads) * 2)
    latencies_ms = []
    payload_byte_sizes = []
    call_counts = []

    print(f"Executing {target_count} timed requests via in-process TestClient...")
    for i in range(target_count):
        p = payloads[i % len(payloads)]
        payload_bytes = len(json.dumps(p).encode("utf-8"))
        payload_byte_sizes.append(payload_bytes)
        call_counts.append(len(p["api_calls"]))

        t0 = time.perf_counter()
        resp = client.post("/validate", json=p)
        t1 = time.perf_counter()

        assert resp.status_code == 200
        latencies_ms.append((t1 - t0) * 1000.0)

    # 3. Compute statistics
    latencies = np.array(latencies_ms)
    median_val = float(np.median(latencies))
    p95_val = float(np.percentile(latencies, 95))
    mean_val = float(np.mean(latencies))
    min_val = float(np.min(latencies))
    max_val = float(np.max(latencies))

    min_bytes = int(np.min(payload_byte_sizes))
    max_bytes = int(np.max(payload_byte_sizes))
    mean_bytes = float(np.mean(payload_byte_sizes))

    min_calls = int(np.min(call_counts))
    max_calls = int(np.max(call_counts))
    mean_calls = float(np.mean(call_counts))

    print("=" * 60)
    print("LATENCY MEASUREMENT RESULTS (POST /validate)")
    print("=" * 60)
    print(f"Embedding model pre-loaded (warm): YES")
    print(f"Warm-up requests: 20")
    print(f"Timed requests:   {target_count}")
    print(f"Payload size (bytes): min={min_bytes}, max={max_bytes}, mean={mean_bytes:.1f} B")
    print(f"API calls per req:    min={min_calls}, max={max_calls}, mean={mean_calls:.2f} calls")
    print("-" * 60)
    print(f"Min:    {min_val:.2f} ms")
    print(f"Median: {median_val:.2f} ms")
    print(f"Mean:   {mean_val:.2f} ms")
    print(f"P95:    {p95_val:.2f} ms")
    print(f"Max:    {max_val:.2f} ms")
    print("=" * 60)

    # Save results as json for record
    out_dir = REPO_ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    stats_data = {
        "model_warm": True,
        "warmup_requests": 20,
        "timed_requests": target_count,
        "min_ms": round(min_val, 2),
        "median_ms": round(median_val, 2),
        "mean_ms": round(mean_val, 2),
        "p95_ms": round(p95_val, 2),
        "max_ms": round(max_val, 2),
        "payload_bytes": {"min": min_bytes, "max": max_bytes, "mean": round(mean_bytes, 1)},
        "calls_per_req": {"min": min_calls, "max": max_calls, "mean": round(mean_calls, 2)},
    }
    with open(out_dir / "latency_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(stats_data, f, indent=2)

    return stats_data

if __name__ == "__main__":
    run_benchmark()
