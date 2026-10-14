"""Backward-compatible entrypoint; api.worker is the sole task implementation."""
from api.worker import app as celery_app, compute_shap

__all__ = ["celery_app", "compute_shap"]
