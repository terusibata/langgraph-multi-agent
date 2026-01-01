"""Agent and Tool registry for dynamic registration and lookup."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
import structlog

if TYPE_CHECKING:
    from src.agents.sub_agents.base import SubAgentBase
    from src.agents.tools.base import ToolBase

logger = structlog.get_logger()


class ToolRegistry:
    """Registry for tool instances."""

    def __init__(self):
        self._tools: dict[str, "ToolBase"] = {}
        self._config: dict[str, dict] = {}

    def register(self, tool: "ToolBase") -> None:
        """Register a tool instance."""
        self._tools[tool.name] = tool
        logger.debug("tool_registered", tool_name=tool.name)

    def get(self, name: str) -> "ToolBase | None":
        """Get a tool by name."""
        return self._tools.get(name)

    def list_all(self) -> list["ToolBase"]:
        """List all registered tools."""
        return list(self._tools.values())

    def list_by_service(self, service: str) -> list["ToolBase"]:
        """List tools that require a specific service token."""
        return [
            tool
            for tool in self._tools.values()
            if tool.required_service_token == service
        ]

    def list_enabled(self) -> list["ToolBase"]:
        """List all enabled tools."""
        return [
            tool
            for tool in self._tools.values()
            if self._config.get(tool.name, {}).get("enabled", True)
        ]

    def load_config(self, config: dict[str, dict]) -> None:
        """Load tool configuration."""
        self._config = config

    def is_enabled(self, name: str) -> bool:
        """Check if a tool is enabled."""
        return self._config.get(name, {}).get("enabled", True)

    def get_tool_info(self, name: str) -> dict | None:
        """Get tool information."""
        tool = self.get(name)
        if not tool:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "required_service_token": tool.required_service_token,
            "enabled": self.is_enabled(name),
        }


class AgentRegistry:
    """Registry for SubAgent instances."""

    def __init__(self):
        self._agents: dict[str, "SubAgentBase"] = {}
        self._config: dict[str, dict] = {}

    def register(self, agent: "SubAgentBase") -> None:
        """Register an agent instance."""
        self._agents[agent.name] = agent
        logger.debug("agent_registered", agent_name=agent.name)

    def get(self, name: str) -> "SubAgentBase | None":
        """Get an agent by name."""
        return self._agents.get(name)

    def list_all(self) -> list["SubAgentBase"]:
        """List all registered agents."""
        return list(self._agents.values())

    def list_by_capability(self, capability: str) -> list["SubAgentBase"]:
        """List agents with a specific capability."""
        return [
            agent
            for agent in self._agents.values()
            if capability in agent.capabilities
        ]

    def list_enabled(self) -> list["SubAgentBase"]:
        """List all enabled agents."""
        return [
            agent
            for agent in self._agents.values()
            if self._config.get(agent.name, {}).get("enabled", True)
        ]

    def load_config(self, config: dict[str, dict]) -> None:
        """Load agent configuration."""
        self._config = config

    def is_enabled(self, name: str) -> bool:
        """Check if an agent is enabled."""
        return self._config.get(name, {}).get("enabled", True)

    def get_agent_config(self, name: str) -> dict:
        """Get agent configuration."""
        return self._config.get(name, {})

    def get_agent_info(self, name: str) -> dict | None:
        """Get agent information."""
        agent = self.get(name)
        if not agent:
            return None
        config = self.get_agent_config(name)
        return {
            "name": agent.name,
            "description": agent.description,
            "capabilities": agent.capabilities,
            "enabled": self.is_enabled(name),
            "tools": config.get("tools", []),
        }


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


def load_agents_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load agent configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Parsed configuration dictionary
    """
    if config_path is None:
        # Default path relative to this file
        config_path = str(Path(__file__).parent.parent / "config" / "agents.yaml")

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        logger.info("agents_config_loaded", path=config_path)
        return config or {}
    except FileNotFoundError:
        logger.warning("agents_config_not_found", path=config_path)
        return {}
    except yaml.YAMLError as e:
        logger.error("agents_config_parse_error", path=config_path, error=str(e))
        return {}


def initialize_registries(config_path: str | None = None) -> None:
    """
    Initialize registries with configuration.

    Args:
        config_path: Optional path to agents.yaml
    """
    config = load_agents_config(config_path)

    # Load configurations
    tool_registry = get_tool_registry()
    agent_registry = get_agent_registry()

    tool_registry.load_config(config.get("tools", {}))
    agent_registry.load_config(config.get("sub_agents", {}))

    logger.info(
        "registries_initialized",
        tools_configured=len(config.get("tools", {})),
        agents_configured=len(config.get("sub_agents", {})),
    )


def register_all_tools() -> None:
    """Register all tool implementations."""
    # Import tools to trigger registration
    from src.agents.tools import (
        servicenow,
        vector_db,
        catalog,
    )

    logger.info("all_tools_registered")


def register_all_agents() -> None:
    """Register all agent implementations."""
    # Import agents to trigger registration
    from src.agents.sub_agents import (
        knowledge_search,
        vector_search,
        catalog as catalog_agent,
    )

    logger.info("all_agents_registered")
