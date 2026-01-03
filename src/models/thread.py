"""Thread database models."""

from datetime import datetime, timezone
from typing import Any, Literal

from sqlalchemy import Boolean, Float, Integer, String, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class ThreadModel(Base, TimestampMixin):
    """Database model for conversation threads."""

    __tablename__ = "threads"

    thread_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="active", index=True
    )  # "active", "warning", "locked"
    total_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    message_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_tokens_used: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "thread_id": self.thread_id,
            "tenant_id": self.tenant_id,
            "status": self.status,
            "total_tokens_used": self.total_tokens_used,
            "total_cost_usd": self.total_cost_usd,
            "message_count": self.message_count,
            "context_tokens_used": self.context_tokens_used,
            "metadata": self.metadata_,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ThreadModel":
        """Create from dictionary."""
        return cls(
            thread_id=data["thread_id"],
            tenant_id=data["tenant_id"],
            status=data.get("status", "active"),
            total_tokens_used=data.get("total_tokens_used", 0),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            message_count=data.get("message_count", 0),
            context_tokens_used=data.get("context_tokens_used", 0),
            metadata_=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.now(timezone.utc)),
            updated_at=data.get("updated_at", datetime.now(timezone.utc)),
        )
