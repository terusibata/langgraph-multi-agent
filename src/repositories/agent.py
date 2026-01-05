"""Agent repository for database operations."""

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from src.models.agent import AgentModel, TemplateAgentModel

logger = structlog.get_logger()


class AgentRepository:
    """Repository for agent CRUD operations."""

    def __init__(self, session: AsyncSession):
        """Initialize with database session."""
        self.session = session

    # ==========================================================================
    # Dynamic Agent operations
    # ==========================================================================

    async def create_agent(self, agent_data: dict[str, Any]) -> AgentModel:
        """Create a new agent definition."""
        agent = AgentModel.from_dict(agent_data)
        self.session.add(agent)
        await self.session.commit()
        await self.session.refresh(agent)
        logger.info("agent_created", agent_id=agent.id, name=agent.name)
        return agent

    async def get_agent_by_id(self, agent_id: str) -> AgentModel | None:
        """Get an agent by ID."""
        result = await self.session.execute(
            select(AgentModel).where(AgentModel.id == agent_id)
        )
        return result.scalar_one_or_none()

    async def get_agent_by_name(self, name: str) -> AgentModel | None:
        """Get an agent by name."""
        result = await self.session.execute(
            select(AgentModel).where(AgentModel.name == name)
        )
        return result.scalar_one_or_none()

    async def list_agents(
        self,
        enabled_only: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AgentModel]:
        """List all agents."""
        query = select(AgentModel)
        if enabled_only:
            query = query.where(AgentModel.enabled == True)
        query = query.order_by(AgentModel.priority.desc(), AgentModel.name)
        query = query.limit(limit).offset(offset)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_agent(
        self,
        agent_id: str,
        updates: dict[str, Any],
    ) -> AgentModel | None:
        """Update an agent definition."""
        agent = await self.get_agent_by_id(agent_id)
        if not agent:
            return None

        for key, value in updates.items():
            if hasattr(agent, key) and value is not None:
                if key == "metadata":
                    setattr(agent, "metadata_", value)
                else:
                    setattr(agent, key, value)
        agent.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(agent)
        logger.info("agent_updated", agent_id=agent_id)
        return agent

    async def delete_agent(self, agent_id: str) -> bool:
        """Delete an agent definition."""
        result = await self.session.execute(
            delete(AgentModel).where(AgentModel.id == agent_id)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("agent_deleted", agent_id=agent_id)
        return deleted

    async def delete_agent_by_name(self, name: str) -> bool:
        """Delete an agent by name."""
        result = await self.session.execute(
            delete(AgentModel).where(AgentModel.name == name)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("agent_deleted", name=name)
        return deleted

    async def agent_exists(self, name: str) -> bool:
        """Check if an agent with the given name exists."""
        result = await self.session.execute(
            select(AgentModel.id).where(AgentModel.name == name)
        )
        return result.scalar_one_or_none() is not None

    async def list_agents_by_capability(
        self,
        capability: str,
        enabled_only: bool = True,
    ) -> list[AgentModel]:
        """List agents with a specific capability."""
        query = select(AgentModel).where(
            AgentModel.capabilities.contains([capability])
        )
        if enabled_only:
            query = query.where(AgentModel.enabled == True)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_all_agent_names(self) -> list[str]:
        """Get all agent names."""
        result = await self.session.execute(select(AgentModel.name))
        return [row[0] for row in result.fetchall()]

    # ==========================================================================
    # Template Agent operations
    # ==========================================================================

    async def create_template_agent(
        self,
        template_data: dict[str, Any],
    ) -> TemplateAgentModel:
        """Create a new template agent."""
        template = TemplateAgentModel.from_dict(template_data)
        self.session.add(template)
        await self.session.commit()
        await self.session.refresh(template)
        logger.info("template_agent_created", name=template.name)
        return template

    async def get_template_agent_by_name(
        self,
        name: str,
    ) -> TemplateAgentModel | None:
        """Get a template agent by name."""
        result = await self.session.execute(
            select(TemplateAgentModel).where(TemplateAgentModel.name == name)
        )
        return result.scalar_one_or_none()

    async def list_template_agents(
        self,
        enabled_only: bool = False,
    ) -> list[TemplateAgentModel]:
        """List all template agents."""
        query = select(TemplateAgentModel)
        if enabled_only:
            query = query.where(TemplateAgentModel.enabled == True)
        query = query.order_by(TemplateAgentModel.name)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def update_template_agent(
        self,
        name: str,
        updates: dict[str, Any],
    ) -> TemplateAgentModel | None:
        """Update a template agent."""
        template = await self.get_template_agent_by_name(name)
        if not template:
            return None

        for key, value in updates.items():
            if hasattr(template, key) and value is not None:
                setattr(template, key, value)
        template.updated_at = datetime.now(timezone.utc)

        await self.session.commit()
        await self.session.refresh(template)
        logger.info("template_agent_updated", name=name)
        return template

    async def delete_template_agent(self, name: str) -> bool:
        """Delete a template agent."""
        result = await self.session.execute(
            delete(TemplateAgentModel).where(TemplateAgentModel.name == name)
        )
        await self.session.commit()
        deleted = result.rowcount > 0
        if deleted:
            logger.info("template_agent_deleted", name=name)
        return deleted

    async def get_all_template_names(self) -> list[str]:
        """Get all template agent names."""
        result = await self.session.execute(select(TemplateAgentModel.name))
        return [row[0] for row in result.fetchall()]
