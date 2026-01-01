"""Metrics collection service."""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import structlog

logger = structlog.get_logger()


class MetricsCollector:
    """
    Collector for Prometheus metrics.

    Tracks request counts, latencies, token usage, and errors.
    """

    def __init__(self):
        """Initialize metrics."""
        # Request metrics
        self.requests_total = Counter(
            "agent_requests_total",
            "Total number of agent requests",
            ["tenant_id", "status"],
        )

        self.request_duration = Histogram(
            "agent_request_duration_seconds",
            "Agent request duration in seconds",
            ["tenant_id"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0],
        )

        # Token metrics
        self.tokens_used = Counter(
            "agent_tokens_total",
            "Total tokens used",
            ["tenant_id", "model_id", "type"],
        )

        self.cost_usd = Counter(
            "agent_cost_usd_total",
            "Total cost in USD",
            ["tenant_id", "model_id"],
        )

        # Agent metrics
        self.agent_executions = Counter(
            "subagent_executions_total",
            "SubAgent execution count",
            ["agent_name", "status"],
        )

        self.agent_duration = Histogram(
            "subagent_duration_seconds",
            "SubAgent execution duration",
            ["agent_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        self.agent_retries = Counter(
            "subagent_retries_total",
            "SubAgent retry count",
            ["agent_name"],
        )

        # Tool metrics
        self.tool_calls = Counter(
            "tool_calls_total",
            "Tool call count",
            ["tool_name", "success"],
        )

        self.tool_duration = Histogram(
            "tool_duration_seconds",
            "Tool execution duration",
            ["tool_name"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )

        # Thread metrics
        self.active_threads = Gauge(
            "active_threads",
            "Number of active threads",
            ["tenant_id"],
        )

        self.thread_context_usage = Histogram(
            "thread_context_usage_percent",
            "Thread context usage percentage",
            ["tenant_id"],
            buckets=[10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100],
        )

        # Error metrics
        self.errors_total = Counter(
            "agent_errors_total",
            "Total errors",
            ["error_code", "category"],
        )

    def record_request(
        self,
        tenant_id: str,
        status: str,
        duration_seconds: float,
    ) -> None:
        """Record a request completion."""
        self.requests_total.labels(tenant_id=tenant_id, status=status).inc()
        self.request_duration.labels(tenant_id=tenant_id).observe(duration_seconds)

    def record_tokens(
        self,
        tenant_id: str,
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record token usage."""
        self.tokens_used.labels(
            tenant_id=tenant_id,
            model_id=model_id,
            type="input",
        ).inc(input_tokens)
        self.tokens_used.labels(
            tenant_id=tenant_id,
            model_id=model_id,
            type="output",
        ).inc(output_tokens)
        self.cost_usd.labels(
            tenant_id=tenant_id,
            model_id=model_id,
        ).inc(cost_usd)

    def record_agent_execution(
        self,
        agent_name: str,
        status: str,
        duration_seconds: float,
        retries: int = 0,
    ) -> None:
        """Record SubAgent execution."""
        self.agent_executions.labels(agent_name=agent_name, status=status).inc()
        self.agent_duration.labels(agent_name=agent_name).observe(duration_seconds)
        if retries > 0:
            self.agent_retries.labels(agent_name=agent_name).inc(retries)

    def record_tool_call(
        self,
        tool_name: str,
        success: bool,
        duration_seconds: float,
    ) -> None:
        """Record tool call."""
        self.tool_calls.labels(tool_name=tool_name, success=str(success)).inc()
        self.tool_duration.labels(tool_name=tool_name).observe(duration_seconds)

    def record_thread_context(
        self,
        tenant_id: str,
        usage_percent: float,
    ) -> None:
        """Record thread context usage."""
        self.thread_context_usage.labels(tenant_id=tenant_id).observe(usage_percent)

    def record_error(self, error_code: str, category: str) -> None:
        """Record an error."""
        self.errors_total.labels(error_code=error_code, category=category).inc()

    def set_active_threads(self, tenant_id: str, count: int) -> None:
        """Set active thread count."""
        self.active_threads.labels(tenant_id=tenant_id).set(count)

    def get_metrics(self) -> bytes:
        """Get Prometheus metrics output."""
        return generate_latest()

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# Global metrics collector
_metrics_collector: MetricsCollector | None = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector
