"""Configuration repository for database operations."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.models.config import SystemConfigModel, CONFIG_KEYS

logger = structlog.get_logger()


class ConfigRepository:
    """Repository for system configuration CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def get_config(self, key: str) -> dict[str, Any] | None:
        """Get a configuration value by key."""
        result = await self.session.execute(
            select(SystemConfigModel).where(SystemConfigModel.key == key)
        )
        config = result.scalar_one_or_none()
        if config:
            return config.value
        # Return default if exists
        return CONFIG_KEYS.get(key)

    async def set_config(
        self,
        key: str,
        value: dict[str, Any],
        description: str | None = None,
    ) -> SystemConfigModel:
        """Set a configuration value."""
        result = await self.session.execute(
            select(SystemConfigModel).where(SystemConfigModel.key == key)
        )
        config = result.scalar_one_or_none()

        if config:
            config.value = value
            if description is not None:
                config.description = description
            config.updated_at = datetime.now(timezone.utc)
        else:
            config = SystemConfigModel(
                key=key,
                value=value,
                description=description,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
            self.session.add(config)

        await self.session.commit()
        await self.session.refresh(config)
        logger.info("config_set", key=key)
        return config

    async def get_all_configs(self) -> dict[str, dict[str, Any]]:
        """Get all configuration values."""
        result = await self.session.execute(select(SystemConfigModel))
        configs = result.scalars().all()
        return {c.key: c.value for c in configs}

    async def delete_config(self, key: str) -> bool:
        """Delete a configuration value."""
        result = await self.session.execute(
            select(SystemConfigModel).where(SystemConfigModel.key == key)
        )
        config = result.scalar_one_or_none()
        if not config:
            return False
        await self.session.delete(config)
        await self.session.commit()
        logger.info("config_deleted", key=key)
        return True

    async def get_planning_config(self) -> dict[str, Any]:
        """Get planning configuration."""
        config = await self.get_config("planning")
        return config or CONFIG_KEYS.get("planning", {})

    async def get_execution_config(self) -> dict[str, Any]:
        """Get execution configuration."""
        config = await self.get_config("execution")
        return config or CONFIG_KEYS.get("execution", {})

    async def is_dynamic_mode(self) -> bool:
        """Check if dynamic mode is enabled."""
        config = await self.get_planning_config()
        return config.get("dynamic_mode", True)

    async def should_prefer_templates(self) -> bool:
        """Check if templates should be preferred."""
        config = await self.get_planning_config()
        return config.get("prefer_templates", True)

    async def initialize_defaults(self) -> None:
        """Initialize default configuration values if not present."""
        for key, value in CONFIG_KEYS.items():
            existing = await self.session.execute(
                select(SystemConfigModel.key).where(SystemConfigModel.key == key)
            )
            if existing.scalar_one_or_none() is None:
                await self.set_config(key, value, f"Default {key} configuration")
        logger.info("default_configs_initialized")
