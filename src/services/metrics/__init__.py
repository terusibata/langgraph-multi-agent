"""Metrics service module."""

from src.services.metrics.collector import MetricsCollector, get_metrics_collector

__all__ = ["MetricsCollector", "get_metrics_collector"]
