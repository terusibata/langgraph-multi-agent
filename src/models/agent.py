"""Agent database models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, Integer, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class AgentModel(Base, TimestampMixin):
    """Database model for dynamic agent definitions."""

    __tablename__ = "agents"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    capabilities: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    tools: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    executor: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=lambda: {"type": "llm"}
    )
    retry_strategy: Mapped[dict[str, Any]] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: {
            "max_attempts": 3,
            "retry_conditions": ["no_results"],
            "query_modification": "synonym",
            "backoff_seconds": 0.5,
        },
    )
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    created_by: Mapped[str | None] = mapped_column(String(255), nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "executor": self.executor,
            "retry_strategy": self.retry_strategy,
            "priority": self.priority,
            "enabled": self.enabled,
            "metadata": self.metadata_,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentModel":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            executor=data.get("executor", {"type": "llm"}),
            retry_strategy=data.get("retry_strategy", {
                "max_attempts": 3,
                "retry_conditions": ["no_results"],
                "query_modification": "synonym",
                "backoff_seconds": 0.5,
            }),
            priority=data.get("priority", 0),
            enabled=data.get("enabled", True),
            metadata_=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            updated_at=data.get("updated_at", datetime.now(timezone.utc)),
            created_by=data.get("created_by"),
        )


class TemplateAgentModel(Base, TimestampMixin):
    """Database model for template agent definitions."""

    __tablename__ = "template_agents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    description: Mapped[str] = mapped_column(Text, nullable=False)
    purpose: Mapped[str] = mapped_column(Text, nullable=False, default="")
    capabilities: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    tools: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    parallel_execution: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    expected_output: Mapped[str] = mapped_column(Text, nullable=False, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True, index=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "parallel_execution": self.parallel_execution,
            "expected_output": self.expected_output,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TemplateAgentModel":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            purpose=data.get("purpose", ""),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            parallel_execution=data.get("parallel_execution", False),
            expected_output=data.get("expected_output", ""),
            enabled=data.get("enabled", True),
        )
