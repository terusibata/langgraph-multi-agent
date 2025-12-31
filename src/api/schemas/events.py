"""SSE event schemas."""

from datetime import datetime
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


EventType = Literal[
    "session_start",
    "plan_created",
    "agent_start",
    "agent_retry",
    "agent_end",
    "tool_call",
    "tool_result",
    "evaluation",
    "token",
    "llm_metrics",
    "session_complete",
    "error",
]


class BaseEvent(BaseModel):
    """Base class for all SSE events."""

    event_type: EventType = Field(..., description="Type of the event")
    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid4().hex[:12]}",
        description="Unique event identifier",
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Event timestamp",
    )

    def to_sse(self) -> str:
        """Convert event to SSE format string."""
        return f"event: {self.event_type}\ndata: {self.model_dump_json()}\n\n"

    def to_sse_dict(self) -> dict:
        """Convert event to dict format for sse-starlette."""
        return {
            "event": self.event_type,
            "data": self.model_dump_json(),
            "id": self.event_id,
        }


class SessionStartEvent(BaseEvent):
    """Event sent when session starts."""

    event_type: Literal["session_start"] = "session_start"
    session_id: str = Field(..., description="Session identifier")
    thread_id: str = Field(..., description="Thread identifier")


class ExecutionPlanSummary(BaseModel):
    """Summary of execution plan."""

    initial_agents: list[str] = Field(..., description="Agents to be executed")
    parallel_groups: list[list[str]] = Field(
        default_factory=list,
        description="Parallel execution groups",
    )
    estimated_steps: int = Field(..., description="Estimated number of steps")


class PlanCreatedEvent(BaseEvent):
    """Event sent when execution plan is created."""

    event_type: Literal["plan_created"] = "plan_created"
    session_id: str = Field(..., description="Session identifier")
    plan: ExecutionPlanSummary = Field(..., description="Execution plan summary")


class AgentStartEvent(BaseEvent):
    """Event sent when a SubAgent starts execution."""

    event_type: Literal["agent_start"] = "agent_start"
    session_id: str = Field(..., description="Session identifier")
    agent_name: str = Field(..., description="Name of the agent")
    task_description: str | None = Field(
        default=None,
        description="Description of the task",
    )


class AgentRetryEvent(BaseEvent):
    """Event sent when a SubAgent retries."""

    event_type: Literal["agent_retry"] = "agent_retry"
    session_id: str = Field(..., description="Session identifier")
    agent_name: str = Field(..., description="Name of the agent")
    attempt: int = Field(..., description="Retry attempt number")
    reason: str = Field(..., description="Reason for retry")
    modified_query: str | None = Field(
        default=None,
        description="Modified query for retry",
    )


class AgentEndEvent(BaseEvent):
    """Event sent when a SubAgent completes execution."""

    event_type: Literal["agent_end"] = "agent_end"
    session_id: str = Field(..., description="Session identifier")
    agent_name: str = Field(..., description="Name of the agent")
    status: Literal["success", "partial", "failed"] = Field(
        ...,
        description="Execution status",
    )
    duration_ms: int = Field(..., description="Execution duration in milliseconds")
    result_summary: str | None = Field(
        default=None,
        description="Brief summary of results",
    )


class ToolCallEvent(BaseEvent):
    """Event sent when a tool is called."""

    event_type: Literal["tool_call"] = "tool_call"
    session_id: str = Field(..., description="Session identifier")
    tool_name: str = Field(..., description="Name of the tool")
    agent_name: str = Field(..., description="Agent that called the tool")
    parameters_summary: dict[str, Any] | None = Field(
        default=None,
        description="Summary of tool parameters",
    )


class ToolResultEvent(BaseEvent):
    """Event sent when a tool returns results."""

    event_type: Literal["tool_result"] = "tool_result"
    session_id: str = Field(..., description="Session identifier")
    tool_name: str = Field(..., description="Name of the tool")
    agent_name: str = Field(..., description="Agent that called the tool")
    success: bool = Field(..., description="Whether tool call succeeded")
    result_summary: str | None = Field(
        default=None,
        description="Brief summary of results",
    )


class EvaluationResult(BaseModel):
    """Result of intermediate evaluation."""

    has_sufficient_info: bool = Field(..., description="Whether sufficient info exists")
    missing_info: list[str] = Field(
        default_factory=list,
        description="List of missing information",
    )
    next_action: str = Field(..., description="Next action to take")
    reasoning: str | None = Field(
        default=None,
        description="Reasoning for the decision",
    )


class EvaluationEvent(BaseEvent):
    """Event sent when intermediate evaluation is performed."""

    event_type: Literal["evaluation"] = "evaluation"
    session_id: str = Field(..., description="Session identifier")
    evaluation: EvaluationResult = Field(..., description="Evaluation result")


class TokenEvent(BaseEvent):
    """Event sent for each generated token (streaming)."""

    event_type: Literal["token"] = "token"
    session_id: str = Field(..., description="Session identifier")
    content: str = Field(..., description="Token content")
    finish_reason: str | None = Field(
        default=None,
        description="Reason for finishing (if last token)",
    )


class LLMCallMetrics(BaseModel):
    """Metrics for a single LLM call."""

    call_id: str = Field(..., description="LLM call identifier")
    model_id: str = Field(..., description="Model used")
    agent: str = Field(..., description="Agent that made the call")
    phase: str | None = Field(default=None, description="Processing phase")
    input_tokens: int = Field(..., description="Input tokens used")
    output_tokens: int = Field(..., description="Output tokens generated")
    cost_usd: float = Field(..., description="Cost in USD")


class LLMMetricsEvent(BaseEvent):
    """Event sent when LLM call completes with metrics."""

    event_type: Literal["llm_metrics"] = "llm_metrics"
    session_id: str = Field(..., description="Session identifier")
    metrics: LLMCallMetrics = Field(..., description="LLM call metrics")


class AgentExecutionSummary(BaseModel):
    """Summary of a SubAgent execution."""

    name: str = Field(..., description="Agent name")
    status: Literal["success", "partial", "failed"] = Field(
        ...,
        description="Execution status",
    )
    retries: int = Field(default=0, description="Number of retries")
    search_variations: list[str] = Field(
        default_factory=list,
        description="Search query variations tried",
    )
    duration_ms: int = Field(..., description="Execution duration")


class ToolExecutionSummary(BaseModel):
    """Summary of a tool execution."""

    tool: str = Field(..., description="Tool name")
    agent: str = Field(..., description="Agent that used the tool")
    success: bool = Field(..., description="Whether execution succeeded")


class ExecutionSummary(BaseModel):
    """Summary of the entire execution."""

    plan: dict = Field(..., description="Execution plan summary")
    agents_executed: list[AgentExecutionSummary] = Field(
        ...,
        description="Summary of executed agents",
    )
    tools_executed: list[ToolExecutionSummary] = Field(
        ...,
        description="Summary of executed tools",
    )


class TotalMetrics(BaseModel):
    """Total metrics for the session."""

    input_tokens: int = Field(..., description="Total input tokens")
    output_tokens: int = Field(..., description="Total output tokens")
    total_tokens: int = Field(..., description="Total tokens")
    total_cost_usd: float = Field(..., description="Total cost in USD")
    llm_call_count: int = Field(..., description="Number of LLM calls")
    tool_call_count: int = Field(..., description="Number of tool calls")


class SessionMetrics(BaseModel):
    """Complete session metrics."""

    duration_ms: int = Field(..., description="Total session duration")
    llm_calls: list[LLMCallMetrics] = Field(
        default_factory=list,
        description="Individual LLM call metrics",
    )
    totals: TotalMetrics = Field(..., description="Aggregated metrics")


class ResponseData(BaseModel):
    """Final response data."""

    content: str = Field(..., description="Full response content")
    finish_reason: str = Field(..., description="Reason for finishing")


class ThreadStateData(BaseModel):
    """Thread state after session."""

    status: Literal["active", "warning", "locked"] = Field(
        ...,
        description="Thread status",
    )
    context_tokens_used: int = Field(..., description="Context tokens used")
    context_max_tokens: int = Field(..., description="Maximum context tokens")
    context_usage_percent: float = Field(..., description="Usage percentage")
    message_count: int = Field(..., description="Number of messages")
    thread_total_tokens: int = Field(..., description="Total tokens in thread")
    thread_total_cost_usd: float = Field(..., description="Total cost in USD")


class SessionCompleteData(BaseModel):
    """Data for session complete event."""

    session_id: str = Field(..., description="Session identifier")
    thread_id: str = Field(..., description="Thread identifier")
    response: ResponseData = Field(..., description="Final response")
    execution_summary: ExecutionSummary = Field(
        ...,
        description="Execution summary",
    )
    metrics: SessionMetrics = Field(..., description="Session metrics")
    thread_state: ThreadStateData = Field(..., description="Thread state")


class SessionCompleteEvent(BaseEvent):
    """Event sent when session completes successfully."""

    event_type: Literal["session_complete"] = "session_complete"
    data: SessionCompleteData = Field(..., description="Complete session data")


class ErrorData(BaseModel):
    """Error data."""

    code: str = Field(..., description="Error code")
    category: str = Field(..., description="Error category")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(default=None, description="Detailed error info")
    service: str | None = Field(
        default=None,
        description="Service that caused the error",
    )
    recoverable: bool = Field(default=False, description="Whether recoverable")
    user_action: str | None = Field(
        default=None,
        description="Suggested user action",
    )


class PartialResults(BaseModel):
    """Partial results when error occurs."""

    agents_completed: list[str] = Field(
        default_factory=list,
        description="Successfully completed agents",
    )
    agents_failed: list[str] = Field(
        default_factory=list,
        description="Failed agents",
    )


class PartialMetrics(BaseModel):
    """Partial metrics when error occurs."""

    duration_ms: int = Field(..., description="Duration until error")
    totals: TotalMetrics = Field(..., description="Partial totals")


class ErrorEventData(BaseModel):
    """Data for error event."""

    session_id: str = Field(..., description="Session identifier")
    thread_id: str | None = Field(default=None, description="Thread identifier")
    error: ErrorData = Field(..., description="Error information")
    partial_results: PartialResults | None = Field(
        default=None,
        description="Partial results if available",
    )
    partial_metrics: PartialMetrics | None = Field(
        default=None,
        description="Partial metrics if available",
    )
    thread_state: ThreadStateData | None = Field(
        default=None,
        description="Thread state if available",
    )


class ErrorEvent(BaseEvent):
    """Event sent when an error occurs."""

    event_type: Literal["error"] = "error"
    data: ErrorEventData = Field(..., description="Error event data")
