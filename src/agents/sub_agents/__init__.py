"""SubAgents module.

All SubAgents are now dynamically created via:
- Dynamic Agent Definitions (stored in database)
- Ad-hoc Agent Generation (created by Planner at runtime)

For examples of static agent implementations, see the examples/agents/ directory.
"""

from src.agents.sub_agents.base import SubAgentBase, RetryStrategy
from src.agents.sub_agents.dynamic import (
    DynamicAgent,
    DynamicAgentFactory,
    get_dynamic_agent,
    get_all_available_agents,
    get_agents_by_capability,
)
from src.agents.sub_agents.adhoc import (
    AdHocAgent,
    AdHocAgentFactory,
)

__all__ = [
    "SubAgentBase",
    "RetryStrategy",
    "DynamicAgent",
    "DynamicAgentFactory",
    "get_dynamic_agent",
    "get_all_available_agents",
    "get_agents_by_capability",
    "AdHocAgent",
    "AdHocAgentFactory",
]
