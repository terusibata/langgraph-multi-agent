"""Tool repository for database operations."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.models.tool import ToolModel

logger = structlog.get_logger()


class ToolRepository:
    """Repository for tool CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    async def create_tool(self, tool_data: dict[str, Any]) -> ToolModel:
        """Create a new tool definition."""
        tool = ToolModel.from_dict(tool_data)
        self.session.add(tool)
        await self.session.commit()
        await self.session.refresh(tool)
        logger.info("tool_created", tool_id=tool.id, name=tool.name)
        return tool

    async def get_tool_by_id(self, tool_id: str) -> ToolModel | None:
        """Get a tool by ID."""
        result = await self.session.execute(
            select(ToolModel).where(ToolModel.id == tool_id)
        )
        return result.scalar_one_or_none()

    async def get_tool_by_name(self, name: str) -> ToolModel | None:
        """Get a tool by name."""
        result = await self.session.execute(
            select(ToolModel).where(ToolModel.name == name)
        )
        return result.scalar_one_or_none()

    async def list_tools(
        self,
        enabled_only: bool = False,
        category: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ToolModel]:
        """List all tools."""
        query = select(ToolModel)
        if enabled_only:
            query = query.where(ToolModel.enabled == True)
        if category:
            query = query.where(ToolModel.category == category)
        query = query.order_by(ToolModel.category, ToolModel.name)
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_tool(
        self,
        tool_id: str,
        updates: dict[str, Any],
    ) -> ToolModel | None:
        """Update a tool definition."""
        tool = await self.get_tool_by_id(tool_id)
        if not tool:
            return None

        for key, value in updates.items():
            if hasattr(tool, key) and value is not None:
                if key == "metadata":
                    setattr(tool, "metadata_", value)
                else:
                    setattr(tool, key, value)
        tool.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(tool)
        logger.info("tool_updated", tool_id=tool_id)
        return tool

    async def delete_tool(self, tool_id: str) -> bool:
        """Delete a tool definition."""
        result = await self.session.execute(
            delete(ToolModel).where(ToolModel.id == tool_id)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("tool_deleted", tool_id=tool_id)
        return deleted

    async def delete_tool_by_name(self, name: str) -> bool:
        """Delete a tool by name."""
        result = await self.session.execute(
            delete(ToolModel).where(ToolModel.name == name)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("tool_deleted", name=name)
        return deleted

    async def tool_exists(self, name: str) -> bool:
        """Check if a tool with the given name exists."""
        result = await self.session.execute(
            select(ToolModel.id).where(ToolModel.name == name)
        )
        return result.scalar_one_or_none() is not None

    async def list_tools_by_service(
        self,
        service_token: str,
        enabled_only: bool = True,
    ) -> list[ToolModel]:
        """List tools that require a specific service token."""
        query = select(ToolModel).where(
            ToolModel.required_service_token == service_token
        )
        if enabled_only:
            query = query.where(ToolModel.enabled == True)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_all_tool_names(self) -> list[str]:
        """Get all tool names."""
        result = await self.session.execute(select(ToolModel.name))
        return [row[0] for row in result.fetchall()]

    async def list_tools_by_names(
        self,
        names: list[str],
    ) -> list[ToolModel]:
        """List tools by names."""
        if not names:
            return []
        result = await self.session.execute(
            select(ToolModel).where(ToolModel.name.in_(names))
        )
        return list(result.scalars().all())
