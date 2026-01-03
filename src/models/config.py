"""Configuration database models."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, String, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column

from src.models.base import Base, TimestampMixin


class SystemConfigModel(Base, TimestampMixin):
    """Database model for system configuration."""

    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "key": self.key,
            "value": self.value,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# Default system configuration keys
CONFIG_KEYS = {
    "planning": {
        "dynamic_mode": True,
        "prefer_templates": True,
    },
    "execution": {
        "parallel_timeout_seconds": 30,
        "sub_agent_timeout_seconds": 60,
        "tool_timeout_seconds": 30,
        "max_adhoc_agents": 5,
        "max_tools_per_adhoc_agent": 4,
    },
}
