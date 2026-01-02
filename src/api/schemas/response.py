"""Response schemas for API endpoints."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    """Standard error response."""

    code: str = Field(..., description="Error code")
    category: str = Field(..., description="Error category")
    message: str = Field(..., description="Human-readable error message")
    detail: str | None = Field(default=None, description="Detailed error information")
    recoverable: bool = Field(default=False, description="Whether the error is recoverable")
    user_action: str | None = Field(
        default=None,
        description="Suggested action for the user",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "code": "AUTH_001",
                    "category": "authentication",
                    "message": "アクセスキーが無効です",
                    "detail": "Invalid signature",
                    "recoverable": False,
                    "user_action": "新しいアクセスキーを取得してください",
                }
            ]
        }
    }


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        ...,
        description="Overall health status",
    )
    version: str = Field(..., description="Application version")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Health check timestamp",
    )
    components: dict[str, dict] = Field(
        default_factory=dict,
        description="Component health status",
    )


class ThreadMetrics(BaseModel):
    """Metrics for a thread."""

    total_tokens_used: int = Field(default=0, description="Total tokens used")
    total_cost_usd: float = Field(default=0.0, description="Total cost in USD")
    message_count: int = Field(default=0, description="Number of messages")
    context_tokens_used: int = Field(default=0, description="Current context tokens")
    context_usage_percent: float = Field(default=0.0, description="Context usage percentage")


class ThreadState(BaseModel):
    """Thread state information."""

    status: Literal["active", "warning", "locked"] = Field(
        ...,
        description="Thread status",
    )
    context_tokens_used: int = Field(..., description="Current context tokens")
    context_max_tokens: int = Field(..., description="Maximum context tokens")
    context_usage_percent: float = Field(..., description="Context usage percentage")
    message_count: int = Field(..., description="Number of messages")
    thread_total_tokens: int = Field(..., description="Total tokens used in thread")
    thread_total_cost_usd: float = Field(..., description="Total cost in USD")


class ThreadResponse(BaseModel):
    """Thread information response."""

    thread_id: str = Field(..., description="Thread identifier")
    tenant_id: str = Field(..., description="Tenant identifier")
    status: Literal["active", "warning", "locked"] = Field(
        ...,
        description="Thread status",
    )
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metrics: ThreadMetrics = Field(..., description="Thread metrics")
    metadata: dict = Field(default_factory=dict, description="Additional metadata")


class ThreadStatusResponse(BaseModel):
    """Minimal thread status response."""

    thread_id: str = Field(..., description="Thread identifier")
    status: Literal["active", "warning", "locked"] = Field(
        ...,
        description="Thread status",
    )
    context_usage_percent: float = Field(..., description="Context usage percentage")
    can_send_message: bool = Field(
        ...,
        description="Whether new messages can be sent",
    )


class ModelInfo(BaseModel):
    """Information about an available model."""

    model_id: str = Field(..., description="Model identifier")
    provider: str = Field(..., description="Model provider")
    max_tokens: int = Field(..., description="Maximum output tokens")
    context_window: int = Field(..., description="Context window size")
    is_default: bool = Field(default=False, description="Whether this is the default model")


class ModelListResponse(BaseModel):
    """List of available models."""

    models: list[ModelInfo] = Field(..., description="Available models")
    default_model_id: str = Field(..., description="Default model ID")


class AgentInfo(BaseModel):
    """Information about an available SubAgent."""

    name: str = Field(..., description="Agent name")
    description: str = Field(..., description="Agent description")
    capabilities: list[str] = Field(..., description="Agent capabilities")
    enabled: bool = Field(..., description="Whether agent is enabled")
    tools: list[str] = Field(..., description="Tools used by this agent")


class AgentListResponse(BaseModel):
    """List of available SubAgents."""

    agents: list[AgentInfo] = Field(..., description="Available agents")


class ToolInfo(BaseModel):
    """Information about an available tool."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    required_service_token: str | None = Field(
        default=None,
        description="Required service token",
    )
    enabled: bool = Field(..., description="Whether tool is enabled")


class ToolListResponse(BaseModel):
    """List of available tools."""

    tools: list[ToolInfo] = Field(..., description="Available tools")
