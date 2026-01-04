"""Agent state definitions for LangGraph."""

from datetime import datetime, timezone
from typing import Annotated, Any, Literal, TypedDict
from uuid import uuid4

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.api.middleware.service_tokens import ServiceTokens


class CompanyContext(BaseModel):
    """Context information for a company (tenant)."""

    company_id: str = Field(..., description="Company identifier (same as tenant_id)")
    company_name: str | None = Field(default=None, description="Company name")
    vision: str | None = Field(
        default=None,
        description="Company vision or mission statement",
    )
    terminology: dict[str, str] = Field(
        default_factory=dict,
        description="Company-specific terminology mapping (term -> definition)",
    )
    reference_info: dict = Field(
        default_factory=dict,
        description="Additional reference information (guidelines, policies, etc.)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional company metadata",
    )


class ServiceToken(BaseModel):
    """Service token information."""

    token: str
    instance: str | None = None
    expires_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)


class RequestContext(BaseModel):
    """Context for the current request."""

    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier (for logging)")
    permissions: list[str] = Field(
        default_factory=list,
        description="Allowed operations",
    )
    service_tokens: ServiceTokens = Field(
        default_factory=ServiceTokens,
        description="Service tokens for external APIs",
    )
    company_context: CompanyContext | None = Field(
        default=None,
        description="Company-specific context (vision, terminology, etc.)",
    )
    request_metadata: dict = Field(
        default_factory=dict,
        description="Request metadata",
    )

    @property
    def request_id(self) -> str:
        """Get request ID from metadata."""
        return self.request_metadata.get("request_id", "unknown")


class FileInput(BaseModel):
    """File input attached to a request."""

    filename: str
    content_type: str
    content: str  # Base64 encoded
    size_bytes: int


class AdHocAgentSpec(BaseModel):
    """
    Specification for an ad-hoc agent generated at runtime.

    Ad-hoc agents are created dynamically by the Planner based on
    the available tools and the task requirements.
    """

    id: str = Field(default_factory=lambda: f"adhoc_{uuid4().hex[:8]}")
    name: str = Field(..., description="Generated name for this ad-hoc agent")
    purpose: str = Field(..., description="The purpose/goal of this agent")
    tools: list[str] = Field(default_factory=list, description="Tool names to use")
    system_prompt: str = Field(default="", description="Auto-generated system prompt")
    expected_output: str = Field(
        default="",
        description="Description of expected output format",
    )
    reasoning: str = Field(
        default="",
        description="Why this combination of tools was chosen",
    )


class Task(BaseModel):
    """A task in the execution plan."""

    id: str = Field(default_factory=lambda: f"task_{uuid4().hex[:8]}")
    # For pre-defined agents (static or template)
    agent_name: str | None = Field(default=None, description="Name of pre-defined agent")
    # For ad-hoc agents generated at runtime
    adhoc_spec: AdHocAgentSpec | None = Field(
        default=None,
        description="Specification for ad-hoc agent",
    )
    priority: int = Field(default=0)
    depends_on: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)
    status: Literal["pending", "running", "completed", "failed"] = "pending"

    @property
    def effective_agent_name(self) -> str:
        """Get the effective agent name (pre-defined or ad-hoc)."""
        if self.agent_name:
            return self.agent_name
        if self.adhoc_spec:
            return self.adhoc_spec.name
        return "unknown"

    @property
    def is_adhoc(self) -> bool:
        """Check if this task uses an ad-hoc agent."""
        return self.adhoc_spec is not None


class ParallelGroup(BaseModel):
    """A group of tasks to execute in parallel."""

    group_id: str = Field(default_factory=lambda: f"group_{uuid4().hex[:8]}")
    task_ids: list[str] = Field(default_factory=list)
    timeout_seconds: int = Field(default=30)


class ExecutionPlan(BaseModel):
    """Execution plan for the session."""

    tasks: list[Task] = Field(default_factory=list)
    parallel_groups: list[ParallelGroup] = Field(default_factory=list)
    execution_order: list[str] = Field(default_factory=list)
    current_phase: str = Field(default="planning")


class SubAgentResult(BaseModel):
    """Result from a SubAgent execution."""

    agent_name: str
    status: Literal["success", "partial", "failed"]
    data: Any = None
    error: str | None = None
    retry_count: int = Field(default=0)
    search_variations: list[str] = Field(default_factory=list)
    duration_ms: int = Field(default=0)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ToolResult(BaseModel):
    """Result from a tool execution."""

    tool_name: str
    agent_name: str
    success: bool
    data: Any = None
    error: str | None = None
    duration_ms: int = Field(default=0)


class Evaluation(BaseModel):
    """Intermediate evaluation result."""

    has_sufficient_info: bool = Field(default=False)
    missing_info: list[str] = Field(default_factory=list)
    next_action: str = Field(default="continue")
    reasoning: str | None = None


class LLMCallMetric(BaseModel):
    """Metrics for a single LLM call."""

    call_id: str = Field(default_factory=lambda: f"llm_{uuid4().hex[:8]}")
    model_id: str
    agent: str
    phase: str | None = None
    input_tokens: int = Field(default=0)
    output_tokens: int = Field(default=0)
    cost_usd: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SessionMetrics(BaseModel):
    """Metrics for the entire session."""

    session_id: str = Field(default_factory=lambda: f"sess_{uuid4().hex[:12]}")
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    duration_ms: int = Field(default=0)
    llm_calls: list[LLMCallMetric] = Field(default_factory=list)
    total_input_tokens: int = Field(default=0)
    total_output_tokens: int = Field(default=0)
    total_cost_usd: float = Field(default=0.0)
    tool_call_count: int = Field(default=0)

    @property
    def llm_call_count(self) -> int:
        """Get total LLM call count."""
        return len(self.llm_calls)

    def add_llm_call(self, metric: LLMCallMetric) -> None:
        """Add an LLM call metric."""
        self.llm_calls.append(metric)
        self.total_input_tokens += metric.input_tokens
        self.total_output_tokens += metric.output_tokens
        self.total_cost_usd += metric.cost_usd

    def finalize(self) -> None:
        """Finalize metrics calculation."""
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            delta = self.completed_at - self.started_at
            self.duration_ms = int(delta.total_seconds() * 1000)


class ThreadState(BaseModel):
    """State of the conversation thread."""

    status: Literal["active", "warning", "locked"] = "active"
    context_tokens_used: int = Field(default=0)
    context_max_tokens: int = Field(default=200000)
    message_count: int = Field(default=0)
    thread_total_tokens: int = Field(default=0)
    thread_total_cost_usd: float = Field(default=0.0)

    @property
    def context_usage_percent(self) -> float:
        """Calculate context usage percentage."""
        if self.context_max_tokens == 0:
            return 0.0
        return (self.context_tokens_used / self.context_max_tokens) * 100

    def update_status(self, warning_threshold: int = 80, lock_threshold: int = 95) -> None:
        """Update status based on context usage."""
        usage = self.context_usage_percent
        if usage >= lock_threshold:
            self.status = "locked"
        elif usage >= warning_threshold:
            self.status = "warning"
        else:
            self.status = "active"


class AgentState(TypedDict):
    """
    Shared state for the multi-agent system.

    This TypedDict defines the structure of state that flows through
    the LangGraph workflow.
    """

    # Conversation history (uses add_messages reducer for proper merging)
    messages: Annotated[list[BaseMessage], add_messages]

    # Request context (authentication, tokens)
    request_context: RequestContext

    # Session identifiers
    session_id: str
    thread_id: str

    # Current input
    user_input: str
    file_inputs: list[FileInput]

    # Execution planning
    execution_plan: ExecutionPlan

    # SubAgent results
    sub_agent_results: dict[str, SubAgentResult]

    # Intermediate evaluation
    intermediate_evaluation: Evaluation

    # Tool results
    tool_results: list[ToolResult]

    # Metrics
    metrics: SessionMetrics

    # Thread state
    thread_state: ThreadState

    # Control flags
    should_continue: bool
    current_agent: str | None

    # Execution mode flags
    fast_response: bool
    direct_tool_mode: bool

    # Response format options
    response_format: str | None
    response_schema: dict | None

    # Thread title (for new threads)
    thread_title: str | None

    # Final response
    final_response: str | None


def create_initial_state(
    user_input: str,
    request_context: RequestContext,
    thread_id: str | None = None,
    session_id: str | None = None,
    file_inputs: list[FileInput] | None = None,
    fast_response: bool = False,
    direct_tool_mode: bool = False,
    response_format: str | None = None,
    response_schema: dict | None = None,
) -> AgentState:
    """
    Create an initial agent state.

    Args:
        user_input: The user's input message
        request_context: Request context with auth info
        thread_id: Optional thread ID for continuation
        session_id: Optional session ID
        file_inputs: Optional file attachments
        fast_response: Enable fast response mode (no sub-agents or tools)
        direct_tool_mode: Enable direct tool mode (MainAgent uses tools directly)
        response_format: Response format ('text' or 'json')
        response_schema: JSON schema for structured response

    Returns:
        Initial AgentState
    """
    return AgentState(
        messages=[],
        request_context=request_context,
        session_id=session_id or f"sess_{uuid4().hex[:12]}",
        thread_id=thread_id or f"thread_{uuid4().hex[:12]}",
        user_input=user_input,
        file_inputs=file_inputs or [],
        execution_plan=ExecutionPlan(),
        sub_agent_results={},
        intermediate_evaluation=Evaluation(),
        tool_results=[],
        metrics=SessionMetrics(),
        thread_state=ThreadState(),
        should_continue=True,
        current_agent=None,
        fast_response=fast_response,
        direct_tool_mode=direct_tool_mode,
        response_format=response_format,
        response_schema=response_schema,
        thread_title=None,
        final_response=None,
    )
