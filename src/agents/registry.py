"""Agent and Tool registry for dynamic registration and lookup."""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar
from uuid import uuid4
import asyncio

import structlog

from src.db.repositories.agent_repository import AgentRepository
from src.db.repositories.tool_repository import ToolRepository

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
# Definition base class (reduces code duplication)
# =============================================================================


class BaseDefinition(ABC):
    """Base class for dynamic definitions."""

    id: str
    name: str
    description: str
    enabled: bool
    metadata: dict
    created_at: datetime
    updated_at: datetime
    created_by: str | None

    def update(self, **kwargs) -> None:
        """Update definition fields."""
        for key, value in kwargs.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        self.updated_at = datetime.now(timezone.utc)

    @abstractmethod
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> "BaseDefinition":
        """Create from dictionary."""
        pass


class ToolDefinition(BaseDefinition):
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


class AgentDefinition(BaseDefinition):
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


class ToolRegistry(ListenerMixin):
    """
    Registry for tool instances and definitions.

    Supports both:
    - Static tools (Python class implementations)
    - Dynamic tools (runtime-defined via API, stored in database)
    """

    def __init__(self):
        super().__init__()  # Initialize listener mixin
        # Static tools (Python implementations)
        self._tools: dict[str, "ToolBase"] = {}
        self._config: dict[str, dict] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register(self, tool: "ToolBase") -> None:
        """Register a static tool instance."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name, type="static")

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name (static tools only)."""
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
        """
        Register a dynamic tool definition.

        Args:
            definition: Tool definition to register
            user_id: ID of the user creating the definition

        Returns:
            The registered definition
        """
        async with self._lock:
            if definition.name in self._tools:
                raise ValueError(f"Tool '{definition.name}' already exists as a static tool")

            # Check if already exists in database
            existing = await ToolRepository.get_by_name(definition.name)
            if existing:
                raise ValueError(f"Tool '{definition.name}' already exists")

            definition.created_by = user_id

            # Save to database
            await ToolRepository.create(definition)

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
        """
        Update a dynamic tool definition.

        Args:
            name: Tool name
            updates: Dictionary of fields to update

        Returns:
            Updated definition or None if not found
        """
        async with self._lock:
            # Get definition from database
            definition = await ToolRepository.get_by_name(name)
            if not definition:
                return None

            # Update definition
            definition.update(**updates)

            # Save to database
            await ToolRepository.update(definition)

            logger.info(
                "tool_definition_updated",
                tool_name=name,
                tool_id=definition.id,
            )

            await self._notify_listeners("tool_updated", definition)
            return definition

    async def delete_definition(self, name: str) -> bool:
        """
        Delete a dynamic tool definition.

        Args:
            name: Tool name

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            # Get definition from database before deleting
            definition = await ToolRepository.get_by_name(name)
            if not definition:
                return False

            # Delete from database
            deleted = await ToolRepository.delete(name)
            if not deleted:
                return False

            logger.info(
                "tool_definition_deleted",
                tool_name=name,
                tool_id=definition.id,
            )

            await self._notify_listeners("tool_deleted", definition)
            return True

    def get(self, name: str) -> "ToolBase | None":
        """Get a static tool by name."""
        return self._tools.get(name)

    async def get_definition(self, name: str) -> ToolDefinition | None:
        """Get a dynamic tool definition by name."""
        return await ToolRepository.get_by_name(name)

    async def get_definition_by_id(self, id: str) -> ToolDefinition | None:
        """Get a dynamic tool definition by ID."""
        all_definitions = await ToolRepository.get_all()
        for definition in all_definitions:
            if definition.id == id:
                return definition
        return None

    def list_all(self) -> list["ToolBase"]:
        """List all registered static tools."""
        return list(self._tools.values())

    async def list_all_definitions(self) -> list[ToolDefinition]:
        """List all dynamic tool definitions."""
        return await ToolRepository.get_all()

    def list_by_service(self, service: str) -> list["ToolBase"]:
        """List static tools that require a specific service token."""
        return [
            tool
            for tool in self._tools.values()
            if tool.required_service_token == service
        ]

    def list_enabled(self) -> list["ToolBase"]:
        """List all enabled static tools."""
        return [
            tool
            for tool in self._tools.values()
            if self._config.get(tool.name, {}).get("enabled", True)
        ]

    async def list_enabled_definitions(self) -> list[ToolDefinition]:
        """List all enabled dynamic tool definitions."""
        return await ToolRepository.get_all(enabled_only=True)

    def load_config(self, config: dict[str, dict]) -> None:
        """Load tool configuration."""
        self._config = config

    async def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        # Check dynamic definitions first
        definition = await ToolRepository.get_by_name(name)
        if definition:
            return definition.enabled
        # Then check static config
        return self._config.get(name, {}).get("enabled", True)

    async def get_tool_info(self, name: str) -> dict | None:
        """Get tool information (static or dynamic)."""
        # Check dynamic definitions first
        definition = await ToolRepository.get_by_name(name)
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
            "enabled": await self.is_enabled(name),
            "metadata": {"type": "static"},
            "created_at": None,
            "updated_at": None,
            "created_by": None,
        }

    async def get_all_tool_names(self) -> list[str]:
        """Get all tool names (static and dynamic)."""
        names = set(self._tools.keys())
        all_definitions = await ToolRepository.get_all()
        names.update([d.name for d in all_definitions])
        return sorted(list(names))


class AgentRegistry(ListenerMixin):
    """
    Registry for SubAgent instances and definitions.

    Supports both:
    - Static agents (Python class implementations)
    - Dynamic agents (runtime-defined via API, stored in database)
    """

    def __init__(self):
        super().__init__()  # Initialize listener mixin
        # Static agents (Python implementations)
        self._agents: dict[str, "SubAgentBase"] = {}
        self._config: dict[str, dict] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

    def register(self, agent: "SubAgentBase") -> None:
        """Register a static agent instance."""
        self._agents[agent.name] = agent
        logger.debug("agent_registered", agent_name=agent.name, type="static")

    def unregister(self, name: str) -> bool:
        """Unregister an agent by name (static agents only)."""
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
        """
        Register a dynamic agent definition.

        Args:
            definition: Agent definition to register
            user_id: ID of the user creating the definition

        Returns:
            The registered definition
        """
        async with self._lock:
            if definition.name in self._agents:
                raise ValueError(f"Agent '{definition.name}' already exists as a static agent")

            # Check if already exists in database
            existing = await AgentRepository.get_by_name(definition.name)
            if existing:
                raise ValueError(f"Agent '{definition.name}' already exists")

            definition.created_by = user_id

            # Save to database
            await AgentRepository.create(definition)

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
        """
        Update a dynamic agent definition.

        Args:
            name: Agent name
            updates: Dictionary of fields to update

        Returns:
            Updated definition or None if not found
        """
        async with self._lock:
            # Get definition from database
            definition = await AgentRepository.get_by_name(name)
            if not definition:
                return None

            # Update definition
            definition.update(**updates)

            # Save to database
            await AgentRepository.update(definition)

            logger.info(
                "agent_definition_updated",
                agent_name=name,
                agent_id=definition.id,
            )

            await self._notify_listeners("agent_updated", definition)
            return definition

    async def delete_definition(self, name: str) -> bool:
        """
        Delete a dynamic agent definition.

        Args:
            name: Agent name

        Returns:
            True if deleted, False if not found
        """
        async with self._lock:
            # Get definition from database before deleting
            definition = await AgentRepository.get_by_name(name)
            if not definition:
                return False

            # Delete from database
            deleted = await AgentRepository.delete(name)
            if not deleted:
                return False

            logger.info(
                "agent_definition_deleted",
                agent_name=name,
                agent_id=definition.id,
            )

            await self._notify_listeners("agent_deleted", definition)
            return True

    def get(self, name: str) -> "SubAgentBase | None":
        """Get a static agent by name."""
        return self._agents.get(name)

    async def get_definition(self, name: str) -> AgentDefinition | None:
        """Get a dynamic agent definition by name."""
        return await AgentRepository.get_by_name(name)

    async def get_definition_by_id(self, id: str) -> AgentDefinition | None:
        """Get a dynamic agent definition by ID."""
        all_definitions = await AgentRepository.get_all()
        for definition in all_definitions:
            if definition.id == id:
                return definition
        return None

    def list_all(self) -> list["SubAgentBase"]:
        """List all registered static agents."""
        return list(self._agents.values())

    async def list_all_definitions(self) -> list[AgentDefinition]:
        """List all dynamic agent definitions."""
        return await AgentRepository.get_all()

    def list_by_capability(self, capability: str) -> list["SubAgentBase"]:
        """List static agents with a specific capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    async def list_definitions_by_capability(self, capability: str) -> list[AgentDefinition]:
        """List dynamic agents with a specific capability."""
        all_definitions = await AgentRepository.get_all()
        return [
            definition
            for definition in all_definitions
            if capability in definition.capabilities
        ]

    def list_enabled(self) -> list["SubAgentBase"]:
        """List all enabled static agents."""
        return [
            agent
            for agent in self._agents.values()
            if self._config.get(agent.name, {}).get("enabled", True)
        ]

    async def list_enabled_definitions(self) -> list[AgentDefinition]:
        """List all enabled dynamic agent definitions."""
        return await AgentRepository.get_all(enabled_only=True)

    def load_config(self, config: dict[str, dict]) -> None:
        """Load agent configuration."""
        self._config = config

    async def is_enabled(self, name: str) -> bool:
        """Check if an agent is enabled."""
        # Check dynamic definitions first
        definition = await AgentRepository.get_by_name(name)
        if definition:
            return definition.enabled
        # Then check static config
        return self._config.get(name, {}).get("enabled", True)

    async def get_agent_config(self, name: str) -> dict:
        """Get agent configuration."""
        # Check dynamic definitions first
        definition = await AgentRepository.get_by_name(name)
        if definition:
            return definition.to_dict()
        # Then check static config
        return self._config.get(name, {})

    async def get_agent_info(self, name: str) -> dict | None:
        """Get agent information (static or dynamic)."""
        # Check dynamic definitions first
        definition = await AgentRepository.get_by_name(name)
        if definition:
            return definition.to_dict()

        # Then check static agents
        agent = self.get(name)
        if not agent:
            return None
        config = await self.get_agent_config(name)
        return {
            "id": f"static_{name}",
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "tools": config.get("tools", [tool.name for tool in agent.tools]),
            "executor": {"type": "python"},
            "retry_strategy": {
                "max_attempts": agent.retry_strategy.max_attempts,
                "retry_conditions": agent.retry_strategy.retry_conditions,
                "query_modification": agent.retry_strategy.query_modification,
                "backoff_seconds": agent.retry_strategy.backoff_seconds,
            },
            "priority": 0,
            "enabled": await self.is_enabled(name),
            "metadata": {"type": "static"},
            "created_at": None,
            "updated_at": None,
            "created_by": None,
        }

    async def get_all_agent_names(self) -> list[str]:
        """Get all agent names (static and dynamic)."""
        names = set(self._agents.keys())
        all_definitions = await AgentRepository.get_all()
        names.update([d.name for d in all_definitions])
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


def initialize_registries() -> None:
    """
    Initialize registries.

    Note: Agent and tool definitions are now stored in the database.
    This function is kept for backward compatibility and initialization of static registries.
    """
    logger.info("registries_initialized")


def register_all_tools() -> None:
    """Register all static tool implementations."""
    # Import tools to trigger registration
    from src.agents.tools import (
        servicenow,
        vector_db,
        catalog,
    )

    logger.info("all_tools_registered")


def register_all_agents() -> None:
    """Register all static agent implementations."""
    # Import agents to trigger registration
    from src.agents.sub_agents import (
        knowledge_search,
        vector_search,
        catalog as catalog_agent,
    )

    logger.info("all_agents_registered")


# =============================================================================
# Utility functions for dynamic definitions
# =============================================================================


def create_tool_definition(
    name: str,
    description: str,
    **kwargs,
) -> ToolDefinition:
    """
    Create a new tool definition.

    Args:
        name: Tool name
        description: Tool description
        **kwargs: Additional fields

    Returns:
        ToolDefinition instance
    """
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
    """
    Create a new agent definition.

    Args:
        name: Agent name
        description: Agent description
        capabilities: Agent capabilities
        **kwargs: Additional fields

    Returns:
        AgentDefinition instance
    """
    return AgentDefinition(
        id=f"agent_{uuid4().hex[:12]}",
        name=name,
        description=description,
        capabilities=capabilities,
        **kwargs,
    )
