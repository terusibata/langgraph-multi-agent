"""Agent and Tool registry for database-backed registration and lookup."""

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4
import asyncio

import structlog

from src.models.base import get_session_factory
from src.repositories.agent import AgentRepository
from src.repositories.tool import ToolRepository
from src.repositories.config import ConfigRepository

if TYPE_CHECKING:
    from src.agents.sub_agents.base import SubAgentBase
    from src.agents.tools.base import ToolBase

logger = structlog.get_logger()


# =============================================================================
# Mixin for listener pattern (reduces code duplication)
# =============================================================================


class ListenerMixin:
    """Mixin class for registry listener pattern."""

    def __init__(self) -> None:
        self._listeners: list[Callable] = []

    def add_listener(self, listener: Callable) -> None:
        """Add a listener for registry changes."""
        self._listeners.append(listener)

    def remove_listener(self, listener: Callable) -> None:
        """Remove a listener."""
        if listener in self._listeners:
            self._listeners.remove(listener)

    async def _notify_listeners(self, event: str, data: Any) -> None:
        """Notify all listeners of a change."""
        for listener in self._listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(event, data)
                else:
                    listener(event, data)
            except Exception as e:
                logger.error("listener_error", event=event, error=str(e))


# =============================================================================
# Definition classes for type compatibility
# =============================================================================


class ToolDefinition:
    """
    Dynamic tool definition that can be created/modified at runtime.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        category: str = "general",
        parameters: list[dict] | None = None,
        executor: dict | None = None,
        required_service_token: str | None = None,
        timeout_seconds: int = 30,
        enabled: bool = True,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        created_by: str | None = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.category = category
        self.parameters = parameters or []
        self.executor = executor or {}
        self.required_service_token = required_service_token
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.created_by = created_by

    def update(self, **kwargs) -> None:
        """Update definition fields."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "parameters": self.parameters,
            "executor": self.executor,
            "required_service_token": self.required_service_token,
            "timeout_seconds": self.timeout_seconds,
            "enabled": self.enabled,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolDefinition":
        """Create from dictionary."""
        return cls(**data)


class AgentDefinition:
    """
    Dynamic agent definition that can be created/modified at runtime.
    """

    def __init__(
        self,
        id: str,
        name: str,
        description: str,
        capabilities: list[str],
        tools: list[str] | None = None,
        executor: dict | None = None,
        retry_strategy: dict | None = None,
        priority: int = 0,
        enabled: bool = True,
        metadata: dict | None = None,
        created_at: datetime | None = None,
        updated_at: datetime | None = None,
        created_by: str | None = None,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.capabilities = capabilities
        self.tools = tools or []
        self.executor = executor or {"type": "llm"}
        self.retry_strategy = retry_strategy or {
            "max_attempts": 3,
            "retry_conditions": ["no_results"],
            "query_modification": "synonym",
            "backoff_seconds": 0.5,
        }
        self.priority = priority
        self.enabled = enabled
        self.metadata = metadata or {}
        self.created_at = created_at or datetime.now(timezone.utc)
        self.updated_at = updated_at or datetime.now(timezone.utc)
        self.created_by = created_by

    def update(self, **kwargs) -> None:
        """Update definition fields."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict:
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
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "created_by": self.created_by,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentDefinition":
        """Create from dictionary."""
        return cls(**data)


class TemplateAgentDefinition:
    """
    Template agent definition for pre-configured agent patterns.
    """

    def __init__(
        self,
        name: str,
        description: str,
        purpose: str,
        capabilities: list[str],
        tools: list[str],
        parallel_execution: bool = False,
        expected_output: str = "",
        enabled: bool = True,
    ):
        self.name = name
        self.description = description
        self.purpose = purpose
        self.capabilities = capabilities
        self.tools = tools
        self.parallel_execution = parallel_execution
        self.expected_output = expected_output
        self.enabled = enabled

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "purpose": self.purpose,
            "capabilities": self.capabilities,
            "tools": self.tools,
            "parallel_execution": self.parallel_execution,
            "expected_output": self.expected_output,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, name: str, data: dict) -> "TemplateAgentDefinition":
        """Create from dictionary."""
        return cls(
            name=name,
            description=data.get("description", ""),
            purpose=data.get("purpose", ""),
            capabilities=data.get("capabilities", []),
            tools=data.get("tools", []),
            parallel_execution=data.get("parallel_execution", False),
            expected_output=data.get("expected_output", ""),
            enabled=data.get("enabled", True),
        )


# =============================================================================
# Database-backed Tool Registry
# =============================================================================


class ToolRegistry(ListenerMixin):
    """
    Registry for tool instances and definitions.
    Uses database for persistence of dynamic tool definitions.
    Static tools (Python class implementations) are kept in memory.
    """

    def __init__(self):
        super().__init__()
        # Static tools (Python implementations) - in memory
        self._tools: dict[str, "ToolBase"] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register(self, tool: "ToolBase") -> None:
        """Register a static tool instance."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name, type="static")

    def unregister(self, name: str) -> bool:
        """Unregister a static tool by name."""
        if name in self._tools:
            del self._tools[name]
            logger.debug("tool_unregistered", tool_name=name, type="static")
            return True
        return False

    async def register_definition(
        self,
        definition: ToolDefinition,
        user_id: str | None = None,
    ) -> ToolDefinition:
        """Register a dynamic tool definition in database."""
        async with self._lock:
            if definition.name in self._tools:
                raise ValueError(f"Tool '{definition.name}' already exists as a static tool")

            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ToolRepository(session)
                if await repo.tool_exists(definition.name):
                    raise ValueError(f"Tool '{definition.name}' already exists")

                definition.created_by = user_id
                tool_data = definition.to_dict()
                await repo.create_tool(tool_data)

            logger.info(
                "tool_definition_registered",
                tool_name=definition.name,
                tool_id=definition.id,
            )

            await self._notify_listeners("tool_created", definition)
            return definition

    async def update_definition(
        self,
        name: str,
        updates: dict,
    ) -> ToolDefinition | None:
        """Update a dynamic tool definition in database."""
        async with self._lock:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ToolRepository(session)
                tool = await repo.get_tool_by_name(name)
                if not tool:
                    return None

                updated = await repo.update_tool(tool.id, updates)
                if not updated:
                    return None

                definition = ToolDefinition.from_dict(updated.to_dict())

            logger.info(
                "tool_definition_updated",
                tool_name=name,
                tool_id=definition.id,
            )

            await self._notify_listeners("tool_updated", definition)
            return definition

    async def delete_definition(self, name: str) -> bool:
        """Delete a dynamic tool definition from database."""
        async with self._lock:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = ToolRepository(session)
                deleted = await repo.delete_tool_by_name(name)

            if deleted:
                logger.info("tool_definition_deleted", tool_name=name)
                await self._notify_listeners("tool_deleted", {"name": name})

            return deleted

    def get(self, name: str) -> "ToolBase | None":
        """Get a static tool by name."""
        return self._tools.get(name)

    async def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a dynamic tool definition by name from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ToolRepository(session)
            tool = await repo.get_tool_by_name(name)
            if tool:
                return ToolDefinition.from_dict(tool.to_dict())
        return None

    async def get_definition_by_id(self, id: str) -> ToolDefinition | None:
        """Get a dynamic tool definition by ID from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ToolRepository(session)
            tool = await repo.get_tool_by_id(id)
            if tool:
                return ToolDefinition.from_dict(tool.to_dict())
        return None

    def list_all(self) -> list["ToolBase"]:
        """List all registered static tools."""
        return list(self._tools.values())

    async def list_all_definitions(self) -> list[ToolDefinition]:
        """List all dynamic tool definitions from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ToolRepository(session)
            tools = await repo.list_tools()
            return [ToolDefinition.from_dict(t.to_dict()) for t in tools]

    def list_by_service(self, service: str) -> list["ToolBase"]:
        """List static tools that require a specific service token."""
        return [
            tool
            for tool in self._tools.values()
            if tool.required_service_token == service
        ]

    def list_enabled(self) -> list["ToolBase"]:
        """List all enabled static tools."""
        return list(self._tools.values())

    async def list_enabled_definitions(self) -> list[ToolDefinition]:
        """List all enabled dynamic tool definitions from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ToolRepository(session)
            tools = await repo.list_tools(enabled_only=True)
            return [ToolDefinition.from_dict(t.to_dict()) for t in tools]

    async def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        # Static tools are always enabled
        if name in self._tools:
            return True
        # Check database for dynamic tools
        definition = await self.get_definition(name)
        return definition.enabled if definition else False

    async def get_tool_info(self, name: str) -> dict | None:
        """Get tool information (static or dynamic)."""
        # Check database first for dynamic tools
        definition = await self.get_definition(name)
        if definition:
            return definition.to_dict()

        # Then check static tools
        tool = self.get(name)
        if not tool:
            return None
        return {
            "id": f"static_{name}",
            "name": tool.name,
            "description": tool.description,
            "category": "static",
            "parameters": [],
            "executor": {"type": "python"},
            "required_service_token": tool.required_service_token,
            "timeout_seconds": tool.timeout_seconds,
            "enabled": True,
            "metadata": {"type": "static"},
            "created_at": None,
            "updated_at": None,
            "created_by": None,
        }

    async def get_all_tool_names(self) -> list[str]:
        """Get all tool names (static and dynamic)."""
        names = set(self._tools.keys())
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = ToolRepository(session)
            db_names = await repo.get_all_tool_names()
            names.update(db_names)
        return sorted(list(names))


# =============================================================================
# Database-backed Agent Registry
# =============================================================================


class AgentRegistry(ListenerMixin):
    """
    Registry for SubAgent instances and definitions.
    Uses database for persistence of dynamic agent definitions.
    Static agents (Python class implementations) are kept in memory.
    """

    def __init__(self):
        super().__init__()
        # Static agents (Python implementations) - in memory
        self._agents: dict[str, "SubAgentBase"] = {}
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register(self, agent: "SubAgentBase") -> None:
        """Register a static agent instance."""
        self._agents[agent.name] = agent
        logger.debug("agent_registered", agent_name=agent.name, type="static")

    def unregister(self, name: str) -> bool:
        """Unregister a static agent by name."""
        if name in self._agents:
            del self._agents[name]
            logger.debug("agent_unregistered", agent_name=name, type="static")
            return True
        return False

    async def register_definition(
        self,
        definition: AgentDefinition,
        user_id: str | None = None,
    ) -> AgentDefinition:
        """Register a dynamic agent definition in database."""
        async with self._lock:
            if definition.name in self._agents:
                raise ValueError(f"Agent '{definition.name}' already exists as a static agent")

            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = AgentRepository(session)
                if await repo.agent_exists(definition.name):
                    raise ValueError(f"Agent '{definition.name}' already exists")

                definition.created_by = user_id
                agent_data = definition.to_dict()
                await repo.create_agent(agent_data)

            logger.info(
                "agent_definition_registered",
                agent_name=definition.name,
                agent_id=definition.id,
            )

            await self._notify_listeners("agent_created", definition)
            return definition

    async def update_definition(
        self,
        name: str,
        updates: dict,
    ) -> AgentDefinition | None:
        """Update a dynamic agent definition in database."""
        async with self._lock:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = AgentRepository(session)
                agent = await repo.get_agent_by_name(name)
                if not agent:
                    return None

                updated = await repo.update_agent(agent.id, updates)
                if not updated:
                    return None

                definition = AgentDefinition.from_dict(updated.to_dict())

            logger.info(
                "agent_definition_updated",
                agent_name=name,
                agent_id=definition.id,
            )

            await self._notify_listeners("agent_updated", definition)
            return definition

    async def delete_definition(self, name: str) -> bool:
        """Delete a dynamic agent definition from database."""
        async with self._lock:
            session_factory = get_session_factory()
            async with session_factory() as session:
                repo = AgentRepository(session)
                deleted = await repo.delete_agent_by_name(name)

            if deleted:
                logger.info("agent_definition_deleted", agent_name=name)
                await self._notify_listeners("agent_deleted", {"name": name})

            return deleted

    def get(self, name: str) -> "SubAgentBase | None":
        """Get a static agent by name."""
        return self._agents.get(name)

    async def get_definition(self, name: str) -> AgentDefinition | None:
        """Get a dynamic agent definition by name from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            agent = await repo.get_agent_by_name(name)
            if agent:
                return AgentDefinition.from_dict(agent.to_dict())
        return None

    async def get_definition_by_id(self, id: str) -> AgentDefinition | None:
        """Get a dynamic agent definition by ID from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            agent = await repo.get_agent_by_id(id)
            if agent:
                return AgentDefinition.from_dict(agent.to_dict())
        return None

    def list_all(self) -> list["SubAgentBase"]:
        """List all registered static agents."""
        return list(self._agents.values())

    async def list_all_definitions(self) -> list[AgentDefinition]:
        """List all dynamic agent definitions from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            agents = await repo.list_agents()
            return [AgentDefinition.from_dict(a.to_dict()) for a in agents]

    def list_by_capability(self, capability: str) -> list["SubAgentBase"]:
        """List static agents with a specific capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    async def list_definitions_by_capability(self, capability: str) -> list[AgentDefinition]:
        """List dynamic agents with a specific capability from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            agents = await repo.list_agents_by_capability(capability)
            return [AgentDefinition.from_dict(a.to_dict()) for a in agents]

    def list_enabled(self) -> list["SubAgentBase"]:
        """List all enabled static agents."""
        return list(self._agents.values())

    async def list_enabled_definitions(self) -> list[AgentDefinition]:
        """List all enabled dynamic agent definitions from database."""
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            agents = await repo.list_agents(enabled_only=True)
            return [AgentDefinition.from_dict(a.to_dict()) for a in agents]

    async def is_enabled(self, name: str) -> bool:
        """Check if an agent is enabled."""
        # Static agents are always enabled
        if name in self._agents:
            return True
        # Check database for dynamic agents
        definition = await self.get_definition(name)
        return definition.enabled if definition else False

    async def get_agent_config(self, name: str) -> dict:
        """Get agent configuration."""
        # Check database first for dynamic agents
        definition = await self.get_definition(name)
        if definition:
            return definition.to_dict()
        # Return empty dict for static agents
        return {}

    async def get_agent_info(self, name: str) -> dict | None:
        """Get agent information (static or dynamic)."""
        # Check database first for dynamic agents
        definition = await self.get_definition(name)
        if definition:
            return definition.to_dict()

        # Then check static agents
        agent = self.get(name)
        if not agent:
            return None
        return {
            "id": f"static_{name}",
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "tools": [tool.name for tool in agent.tools],
            "executor": {"type": "python"},
            "retry_strategy": {
                "max_attempts": agent.retry_strategy.max_attempts,
                "retry_conditions": agent.retry_strategy.retry_conditions,
                "query_modification": agent.retry_strategy.query_modification,
                "backoff_seconds": agent.retry_strategy.backoff_seconds,
            },
            "priority": 0,
            "enabled": True,
            "metadata": {"type": "static"},
            "created_at": None,
            "updated_at": None,
            "created_by": None,
        }

    async def get_all_agent_names(self) -> list[str]:
        """Get all agent names (static and dynamic)."""
        names = set(self._agents.keys())
        session_factory = get_session_factory()
        async with session_factory() as session:
            repo = AgentRepository(session)
            db_names = await repo.get_all_agent_names()
            names.update(db_names)
        return sorted(list(names))


# Global registry instances
_tool_registry: ToolRegistry | None = None
_agent_registry: AgentRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry."""
    global _tool_registry
    if _tool_registry is None:
        _tool_registry = ToolRegistry()
    return _tool_registry


def get_agent_registry() -> AgentRegistry:
    """Get the global agent registry."""
    global _agent_registry
    if _agent_registry is None:
        _agent_registry = AgentRegistry()
    return _agent_registry


def reset_registries() -> None:
    """Reset registries (useful for testing)."""
    global _tool_registry, _agent_registry
    _tool_registry = None
    _agent_registry = None


# =============================================================================
# Template Agents (database-backed)
# =============================================================================


async def get_template_agents() -> dict[str, TemplateAgentDefinition]:
    """Get all template agent definitions from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = AgentRepository(session)
        templates = await repo.list_template_agents()
        return {t.name: TemplateAgentDefinition.from_dict(t.name, t.to_dict()) for t in templates}


async def get_template_agent(name: str) -> TemplateAgentDefinition | None:
    """Get a template agent by name from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = AgentRepository(session)
        template = await repo.get_template_agent_by_name(name)
        if template:
            return TemplateAgentDefinition.from_dict(template.name, template.to_dict())
    return None


async def list_enabled_templates() -> list[TemplateAgentDefinition]:
    """List all enabled template agents from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = AgentRepository(session)
        templates = await repo.list_template_agents(enabled_only=True)
        return [TemplateAgentDefinition.from_dict(t.name, t.to_dict()) for t in templates]


# =============================================================================
# Planning configuration (database-backed)
# =============================================================================


async def get_planning_config() -> dict:
    """Get the planning configuration from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = ConfigRepository(session)
        return await repo.get_planning_config()


async def is_dynamic_mode() -> bool:
    """Check if dynamic mode is enabled from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = ConfigRepository(session)
        return await repo.is_dynamic_mode()


async def should_prefer_templates() -> bool:
    """Check if templates should be preferred over ad-hoc agents from database."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = ConfigRepository(session)
        return await repo.should_prefer_templates()


# =============================================================================
# Initialization
# =============================================================================


async def initialize_registries() -> None:
    """Initialize registries and database with default configuration."""
    # Initialize database with default configs
    session_factory = get_session_factory()
    async with session_factory() as session:
        repo = ConfigRepository(session)
        await repo.initialize_defaults()

    logger.info("registries_initialized", storage="database")


def register_all_tools() -> None:
    """
    Register all static tool implementations.

    NOTE: Static tools have been moved to examples/ directory.
    All tools should now be registered via Admin API as dynamic tools.
    This function is kept for backward compatibility but does nothing.
    """
    logger.info("static_tools_deprecated", message="Use dynamic tools via Admin API")


def register_all_agents() -> None:
    """
    Register all static agent implementations.

    NOTE: Static agents have been moved to examples/ directory.
    All agents should now be registered via Admin API as dynamic agents.
    This function is kept for backward compatibility but does nothing.
    """
    logger.info("static_agents_deprecated", message="Use dynamic agents via Admin API")


# =============================================================================
# Utility functions for dynamic definitions
# =============================================================================


def create_tool_definition(
    name: str,
    description: str,
    **kwargs,
) -> ToolDefinition:
    """Create a new tool definition."""
    return ToolDefinition(
        id=f"tool_{uuid4().hex[:12]}",
        name=name,
        description=description,
        **kwargs,
    )


def create_agent_definition(
    name: str,
    description: str,
    capabilities: list[str],
    **kwargs,
) -> AgentDefinition:
    """Create a new agent definition."""
    return AgentDefinition(
        id=f"agent_{uuid4().hex[:12]}",
        name=name,
        description=description,
        capabilities=capabilities,
        **kwargs,
    )
