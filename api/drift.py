"""Bounded input-distribution monitoring; KS drift does not measure model accuracy."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from prometheus_client.core import GaugeMetricFamily

MAX_REFERENCE_ROWS = 512


def _fingerprint(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def create_baseline(features, feature_names, source=None, max_rows=MAX_REFERENCE_ROWS):
    """Sample only supplied raw training rows; callers own the train/holdout split."""
    names = list(feature_names)
    values = np.asarray(features, dtype=float)
    if not names or len(names) != len(set(names)) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Baseline feature names must be unique nonempty strings")
    if values.ndim != 2 or values.shape[1] != len(names) or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Baseline requires finite raw training rows in the declared feature order")
    if not 2 <= max_rows <= MAX_REFERENCE_ROWS:
        raise ValueError("Baseline max_rows must be between 2 and 512")
    count = min(len(values), max_rows)
    indices = np.sort(np.random.default_rng(42).choice(len(values), size=count, replace=False))
    payload = {"version": 1, "feature_names": names, "training_rows": len(values),
               "reference_rows": count, "reference_samples": values[indices].tolist(),
               "source": dict(source or {})}
    payload["source"].setdefault("split", "training_only")
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def validate_baseline(payload, feature_names=None):
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported drift baseline schema")
    names = payload.get("feature_names", [])
    samples = np.asarray(payload.get("reference_samples", []), dtype=float)
    if not names or len(names) != len(set(names)) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Invalid drift baseline feature names")
    if feature_names is not None and list(feature_names) != names:
        raise ValueError("Drift baseline feature order does not match raw model inputs")
    if (samples.ndim != 2 or samples.shape[1] != len(names) or not 2 <= len(samples) <= MAX_REFERENCE_ROWS
            or not np.isfinite(samples).all() or payload.get("reference_rows") != len(samples)
            or payload.get("training_rows", 0) < len(samples)):
        raise ValueError("Invalid or unbounded drift baseline samples")
    content = {key: value for key, value in payload.items() if key != "fingerprint"}
    if payload.get("fingerprint") != _fingerprint(content):
        raise ValueError("Drift baseline fingerprint does not match its content")
    if not isinstance(payload.get("source"), dict) or payload["source"].get("split") != "training_only":
        raise ValueError("Drift reference must identify training-only source rows")
    return payload


def load_baseline(path, feature_names):
    path = Path(path)
    if not path.is_file():
        return None
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("Drift baseline artifact is too large")
    return validate_baseline(json.loads(path.read_text()), feature_names)


class DriftMonitor:
    def __init__(self, baseline, client=None, max_samples=512, min_samples=20, threshold=0.05,
                 unavailable_reason=None):
        if not 20 <= min_samples <= max_samples <= 512:
            raise ValueError("Drift window requires 20 <= min_samples <= max_samples <= 512")
        if not 0 < threshold < 1:
            raise ValueError("Drift KS threshold must be between zero and one")
        self.baseline = validate_baseline(baseline) if baseline is not None else None
        self.client = client
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.threshold = threshold
        self.unavailable_reason = unavailable_reason
        self.key = f"fraud:drift:{self.baseline['fingerprint']}" if self.baseline is not None else None

    def _client(self):
        if self.client is None:
            import redis
            self.client = redis.Redis.from_url(
                os.getenv("DRIFT_REDIS_URL", os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")),
                socket_connect_timeout=2, socket_timeout=2, decode_responses=True,
            )
        return self.client

    def observe(self, features):
        if self.baseline is None:
            return False
        values = np.asarray(features, dtype=float)
        if values.shape != (len(self.baseline["feature_names"]),) or not np.isfinite(values).all():
            raise ValueError("Live drift observation does not match raw model features")
        try:
            pipeline = self._client().pipeline(transaction=True)
            pipeline.rpush(self.key, json.dumps(values.tolist(), allow_nan=False))
            pipeline.ltrim(self.key, -self.max_samples, -1)
            pipeline.expire(self.key, 3600)
            pipeline.execute()
            return True
        except Exception:
            return False

    def report(self):
        report = {"status": "UNAVAILABLE", "samples": None, "min_samples": self.min_samples,
                  "max_samples": self.max_samples, "threshold": self.threshold, "features": {},
                  "meaning": "Raw input distribution shift; labelled outcomes are required to measure AUC"}
        if self.baseline is None:
            report["reason"] = self.unavailable_reason or "No raw training baseline artifact is available"
            return report
        report["baseline"] = {"fingerprint": self.baseline["fingerprint"],
                              "reference_rows": self.baseline["reference_rows"],
                              "source": {key: value for key, value in self.baseline["source"].items()
                                         if key in {"data_kind", "dataset_sha256", "split"}}}
        if self.baseline["reference_rows"] < self.min_samples:
            report["reason"] = "Training baseline has fewer than the required minimum samples"
            return report
        try:
            # Limit the response at Redis, including when an older window is larger.
            observations = self._client().lrange(self.key, -self.max_samples, -1)
            report["samples"] = len(observations)
            values = np.asarray([json.loads(row) for row in observations], dtype=float)
            if observations and (values.shape != (len(observations), len(self.baseline["feature_names"]))
                                 or not np.isfinite(values).all()):
                raise ValueError("Stored live observations are invalid")
            if len(observations) < self.min_samples:
                report["status"] = "WARMING_UP"
                return report
            from scipy.stats import ks_2samp
            reference = np.asarray(self.baseline["reference_samples"], dtype=float)
            for index, name in enumerate(self.baseline["feature_names"]):
                result = ks_2samp(reference[:, index], values[:, index], alternative="two-sided", method="auto")
                report["features"][name] = {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}
            report["status"] = "DRIFT" if any(f["statistic"] > self.threshold for f in report["features"].values()) else "OK"
            return report
        except Exception:
            report.update(status="ERROR", reason="Shared live drift window is unavailable or invalid", features={})
            return report


class DriftCollector:
    """Read shared Redis state at scrape time, consistently across Gunicorn workers."""
    def __init__(self, monitor_provider):
        self.monitor_provider = monitor_provider

    def describe(self):
        yield GaugeMetricFamily("fraud_drift_ks", "Two-sample KS statistic against raw training reference", labels=["feature"])
        yield GaugeMetricFamily("fraud_drift_available", "One when drift has enough valid reference and live samples")
        yield GaugeMetricFamily("fraud_drift_window_samples", "Number of observations in the shared bounded live window")

    def collect(self):
        report = self.monitor_provider().report()
        available = report["status"] in {"OK", "DRIFT"}
        yield GaugeMetricFamily("fraud_drift_available", "One when drift has enough valid reference and live samples", value=int(available))
        if report["samples"] is not None:
            yield GaugeMetricFamily("fraud_drift_window_samples", "Number of observations in the shared bounded live window", value=report["samples"])
        if available:
            metric = GaugeMetricFamily("fraud_drift_ks", "Two-sample KS statistic against raw training reference", labels=["feature"])
            for name, result in report["features"].items():
                metric.add_metric([name], result["statistic"])
            yield metric
