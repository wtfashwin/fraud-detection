"""Synthetic distribution fixtures test drift mechanics, never fraud-model quality."""
import json
from copy import deepcopy

import numpy as np
import pytest
from prometheus_client import CollectorRegistry, generate_latest


class MemoryRedis:
    """Transport boundary fixture; shared storage models Redis list semantics only."""
    def __init__(self, shared=None):
        self.shared = shared if shared is not None else {}
        self.operations = []

    def pipeline(self, transaction=True):
        return MemoryPipeline(self)

    def lrange(self, key, start, stop):
        values = self.shared.get(key, [])
        return values[start:] if stop == -1 else values[start:stop + 1]


class MemoryPipeline:
    def __init__(self, client):
        self.client = client
        self.commands = []

    def rpush(self, key, value):
        self.commands.append(("rpush", key, value)); return self

    def ltrim(self, key, start, stop):
        self.commands.append(("ltrim", key, start, stop)); return self

    def expire(self, key, ttl):
        self.commands.append(("expire", key, ttl)); return self

    def execute(self):
        for command in self.commands:
            op, key, *args = command
            if op == "rpush": self.client.shared.setdefault(key, []).append(args[0])
            elif op == "ltrim": self.client.shared[key] = self.client.shared[key][args[0]:] if args[1] == -1 else self.client.shared[key][args[0]:args[1] + 1]
        self.client.operations.extend(self.commands)
        return [True] * len(self.commands)


def baseline(rows=20):
    from api.drift import create_baseline
    data = np.column_stack([np.arange(rows, dtype=float), np.arange(rows, dtype=float) * 2])
    return create_baseline(data, ["V1", "Amount"], source={"data_kind": "synthetic", "split": "training_only"})


def test_baseline_is_bounded_reproducible_and_preserves_raw_order():
    from api.drift import create_baseline
    data = np.arange(1200, dtype=float).reshape(600, 2)
    first = create_baseline(data, ["V1", "Amount"], source={"data_kind": "synthetic"})
    second = create_baseline(data, ["V1", "Amount"], source={"data_kind": "synthetic"})
    assert first == second
    assert first["training_rows"] == 600
    assert first["reference_rows"] == 512
    assert len(first["reference_samples"]) == 512
    assert first["feature_names"] == ["V1", "Amount"]
    assert first["source"]["data_kind"] == "synthetic"


def test_missing_baseline_has_no_fake_zero_ks_series():
    from api.drift import DriftCollector, DriftMonitor
    monitor = DriftMonitor(None)
    assert monitor.report()["status"] == "UNAVAILABLE"
    registry = CollectorRegistry()
    registry.register(DriftCollector(lambda: monitor))
    metrics = generate_latest(registry).decode()
    assert 'fraud_drift_available 0.0' in metrics
    assert 'fraud_drift_ks{' not in metrics


def test_two_monitors_share_bounded_window_instead_of_process_local_state():
    from api.drift import DriftMonitor
    shared = {}
    one = DriftMonitor(baseline(), client=MemoryRedis(shared), max_samples=20, min_samples=20)
    two = DriftMonitor(baseline(), client=MemoryRedis(shared), max_samples=20, min_samples=20)
    for i in range(20):
        (one if i % 2 else two).observe([i, i * 2])
    assert one.report()["samples"] == two.report()["samples"] == 20
    assert one.report()["status"] == "OK"
    assert one.report()["features"]["V1"] == {"statistic": 0.0, "pvalue": 1.0}
    for i in range(20): two.observe([i + 100, (i + 100) * 2])
    drift = one.report()
    assert drift["samples"] == 20
    assert drift["status"] == "DRIFT"
    assert drift["features"]["V1"]["statistic"] == 1.0
    assert drift["features"]["V1"]["pvalue"] < 0.001
    assert "auc" not in drift


def test_insufficient_samples_are_warming_up():
    from api.drift import DriftMonitor
    monitor = DriftMonitor(baseline(), client=MemoryRedis(), min_samples=20)
    for i in range(19): monitor.observe([100 + i, 100 + i])
    report = monitor.report()
    assert report["status"] == "WARMING_UP"
    assert report["samples"] == 19
    assert report["features"] == {}


def test_redis_failure_is_explicit_and_does_not_fabricate_statistics():
    from api.drift import DriftMonitor
    class Unavailable:
        def lrange(self, *args): raise ConnectionError("synthetic transport failure")
        def pipeline(self, *args, **kwargs): raise ConnectionError("synthetic transport failure")
    monitor = DriftMonitor(baseline(), client=Unavailable())
    assert monitor.observe([1, 2]) is False
    assert monitor.report()["status"] == "ERROR"
    assert monitor.report()["features"] == {}


def test_invalid_or_tampered_baseline_is_rejected():
    from api.drift import validate_baseline
    corrupted = deepcopy(baseline())
    corrupted["reference_samples"][0][0] = float("nan")
    with pytest.raises(ValueError): validate_baseline(corrupted)
    corrupted = deepcopy(baseline())
    corrupted["reference_samples"][0][0] += 1
    with pytest.raises(ValueError, match="fingerprint"): validate_baseline(corrupted)


def test_metrics_read_current_shared_window_at_scrape_time():
    from api.drift import DriftCollector, DriftMonitor
    monitor = DriftMonitor(baseline(), client=MemoryRedis(), min_samples=20)
    registry = CollectorRegistry()
    registry.register(DriftCollector(lambda: monitor))
    assert 'fraud_drift_ks{' not in generate_latest(registry).decode()
    for i in range(20): monitor.observe([100 + i, 200 + i])
    metrics = generate_latest(registry).decode()
    assert 'fraud_drift_ks{feature="V1"} 1.0' in metrics
    assert 'fraud_drift_available 1.0' in metrics
