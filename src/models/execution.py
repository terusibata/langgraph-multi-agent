"""Execution session and result database models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text, JSON, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from src.models.base import Base, TimestampMixin


class ExecutionSessionModel(Base, TimestampMixin):
    """Database model for execution sessions."""

    __tablename__ = "execution_sessions"

    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    thread_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("threads.thread_id", ondelete="CASCADE"), nullable=False, index=True
    )
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    user_input: Mapped[str] = mapped_column(Text, nullable=False)
    final_response: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True
    )  # "pending", "running", "completed", "failed"

    # Execution plan
    execution_plan: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Metrics
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    llm_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tool_call_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # LLM call details
    llm_calls: Mapped[list[dict[str, Any]]] = mapped_column(JSON, nullable=False, default=list)

    # Error info if failed
    error_code: Mapped[str | None] = mapped_column(String(32), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Request context (for audit)
    request_context: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    # Relationships
    results: Mapped[list["ExecutionResultModel"]] = relationship(
        "ExecutionResultModel", back_populates="session", cascade="all, delete-orphan"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "thread_id": self.thread_id,
            "tenant_id": self.tenant_id,
            "user_input": self.user_input,
            "final_response": self.final_response,
            "status": self.status,
            "execution_plan": self.execution_plan,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": self.total_cost_usd,
            "llm_call_count": self.llm_call_count,
            "tool_call_count": self.tool_call_count,
            "llm_calls": self.llm_calls,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "request_context": self.request_context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "results": [r.to_dict() for r in self.results] if self.results else [],
        }


class ExecutionResultModel(Base, TimestampMixin):
    """Database model for sub-agent/tool execution results."""

    __tablename__ = "execution_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("execution_sessions.session_id", ondelete="CASCADE"),
        nullable=False, index=True
    )

    # Agent or tool info
    result_type: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True
    )  # "agent" or "tool"
    agent_name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    tool_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Execution details
    is_adhoc: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    adhoc_spec: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    task_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Result
    status: Mapped[str] = mapped_column(
        String(16), nullable=False
    )  # "success", "partial", "failed"
    data: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Retry info
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    search_variations: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    # Timing
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    # Relationships
    session: Mapped["ExecutionSessionModel"] = relationship(
        "ExecutionSessionModel", back_populates="results"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "session_id": self.session_id,
            "result_type": self.result_type,
            "agent_name": self.agent_name,
            "tool_name": self.tool_name,
            "is_adhoc": self.is_adhoc,
            "adhoc_spec": self.adhoc_spec,
            "task_id": self.task_id,
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "retry_count": self.retry_count,
            "search_variations": self.search_variations,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
