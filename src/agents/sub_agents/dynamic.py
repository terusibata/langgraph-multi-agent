"""Dynamic agent execution based on runtime definitions."""

from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel
import structlog

from src.agents.registry import AgentDefinition, get_agent_registry, get_tool_registry
from src.agents.state import AgentState, SubAgentResult
from src.agents.sub_agents.base import SubAgentBase, RetryStrategy
from src.agents.tools.base import ToolBase, ToolContext
from src.agents.tools.dynamic import DynamicTool, DynamicToolFactory
from src.services.llm import get_llm
from src.config import get_settings

logger = structlog.get_logger()


class DynamicAgent(SubAgentBase):
    """
    An agent that executes based on a dynamic definition.

    Supports multiple executor types:
    - llm: Use LLM for reasoning and tool calling
    - rule_based: Use predefined rules for decision making
    - hybrid: Combine rules with LLM fallback
    """

    def __init__(self, definition: AgentDefinition):
        """
        Initialize from an agent definition.

        Args:
            definition: The agent definition
        """
        # Build retry strategy from definition
        retry_config = definition.retry_strategy
        retry_strategy = RetryStrategy(
            max_attempts=retry_config.get("max_attempts", 3),
            retry_conditions=retry_config.get("retry_conditions", ["no_results"]),
            query_modification=retry_config.get("query_modification", "synonym"),
            backoff_seconds=retry_config.get("backoff_seconds", 0.5),
        )

        # Get tools
        tools = self._build_tools(definition.tools)

        # Get model ID from executor config
        executor = definition.executor
        model_id = executor.get("model_id") or get_settings().sub_agent_model_id

        super().__init__(
            name=definition.name,
            description=definition.description,
            capabilities=definition.capabilities,
            tools=tools,
            model_id=model_id,
            retry_strategy=retry_strategy,
        )

        self.definition = definition
        self.executor_type = executor.get("type", "llm")
        self.priority = definition.priority

    def _build_tools(self, tool_names: list[str]) -> list[ToolBase]:
        """
        Build tool instances from tool names.

        Args:
            tool_names: List of tool names

        Returns:
            List of tool instances
        """
        tools = []
        tool_registry = get_tool_registry()

        for name in tool_names:
            # First check for static tools
            static_tool = tool_registry.get(name)
            if static_tool:
                tools.append(static_tool)
                continue

            # Then check for dynamic tools
            definition = tool_registry.get_definition(name)
            if definition:
                tools.append(DynamicToolFactory.create(definition))
                continue

            logger.warning("tool_not_found", tool_name=name, agent=self.name)

        return tools

    async def execute(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute the agent's task.

        Args:
            state: Current agent state
            task_params: Optional parameters for the task

        Returns:
            SubAgentResult with execution status and data
        """
        started_at = datetime.now(timezone.utc)

        try:
            # Execute based on type
            if self.executor_type == "llm":
                result = await self._execute_llm(state, task_params)
            elif self.executor_type == "rule_based":
                result = await self._execute_rule_based(state, task_params)
            elif self.executor_type == "hybrid":
                result = await self._execute_hybrid(state, task_params)
            else:
                result = SubAgentResult(
                    agent_name=self.name,
                    status="failed",
                    error=f"Unknown executor type: {self.executor_type}",
                )

            # Set timing
            result.started_at = started_at
            result.completed_at = datetime.now(timezone.utc)
            result.duration_ms = int(
                (result.completed_at - started_at).total_seconds() * 1000
            )

            return result

        except Exception as e:
            logger.error(
                "dynamic_agent_error",
                agent=self.name,
                error=str(e),
            )
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error=str(e),
                started_at=started_at,
                completed_at=datetime.now(timezone.utc),
            )

    async def _execute_llm(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute using LLM for reasoning.

        Args:
            state: Current agent state
            task_params: Task parameters

        Returns:
            SubAgentResult
        """
        executor = self.definition.executor
        system_prompt = executor.get("system_prompt") or self.get_system_prompt()
        temperature = executor.get("temperature", 0.0)
        max_tokens = executor.get("max_tokens", 4096)
        output_format = executor.get("output_format", "text")

        # Build messages
        messages = [SystemMessage(content=system_prompt)]

        # Add user query
        query = task_params.get("query", state["user_input"]) if task_params else state["user_input"]
        messages.append(HumanMessage(content=query))

        # Get LLM
        llm = get_llm(self.model_id)

        # If we have tools, bind them
        if self.tools:
            tool_schemas = [tool.get_langchain_tool_schema() for tool in self.tools]
            llm = llm.bind_tools(tool_schemas)

        # Invoke LLM
        response = await llm.ainvoke(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Process tool calls if any
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_results = await self._execute_tool_calls(
                response.tool_calls,
                state,
            )
            return SubAgentResult(
                agent_name=self.name,
                status="success" if tool_results else "partial",
                data=tool_results,
            )

        # Parse output based on format
        content = response.content
        if output_format == "json":
            try:
                import json
                content = json.loads(content)
            except json.JSONDecodeError:
                pass
        elif output_format == "structured":
            output_schema = executor.get("output_schema")
            if output_schema:
                content = self._parse_structured_output(content, output_schema)

        return SubAgentResult(
            agent_name=self.name,
            status="success",
            data=content,
        )

    async def _execute_rule_based(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute using predefined rules.

        Args:
            state: Current agent state
            task_params: Task parameters

        Returns:
            SubAgentResult
        """
        executor = self.definition.executor
        rules = executor.get("rules", [])

        if not rules:
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error="No rules configured for rule-based executor",
            )

        query = task_params.get("query", state["user_input"]) if task_params else state["user_input"]
        query_lower = query.lower()

        # Evaluate rules
        for rule in rules:
            condition = rule.get("condition", {})
            action = rule.get("action", {})

            if self._evaluate_condition(condition, query_lower, state, task_params):
                result = await self._execute_action(action, state, task_params)
                return result

        # No rule matched
        return SubAgentResult(
            agent_name=self.name,
            status="partial",
            data={"message": "No matching rule found"},
        )

    async def _execute_hybrid(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute using rules first, then fall back to LLM.

        Args:
            state: Current agent state
            task_params: Task parameters

        Returns:
            SubAgentResult
        """
        # Try rule-based first
        result = await self._execute_rule_based(state, task_params)

        # If rule-based succeeded, return result
        if result.status == "success":
            return result

        # Fall back to LLM
        logger.info(
            "hybrid_fallback_to_llm",
            agent=self.name,
            rule_result=result.status,
        )
        return await self._execute_llm(state, task_params)

    async def _execute_tool_calls(
        self,
        tool_calls: list[dict],
        state: AgentState,
    ) -> list[dict]:
        """
        Execute tool calls from LLM response.

        Args:
            tool_calls: List of tool calls
            state: Current agent state

        Returns:
            List of tool results
        """
        results = []
        context = ToolContext(
            service_tokens=state["request_context"].service_tokens,
            tenant_id=state["request_context"].tenant_id,
            user_id=state["request_context"].user_id,
            request_id=state["request_context"].request_id,
        )

        for call in tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})

            tool = self.get_tool(tool_name)
            if not tool:
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                })
                continue

            try:
                result = await tool.execute_with_validation(tool_args, context)
                results.append({
                    "tool": tool_name,
                    "success": result.success,
                    "data": result.data,
                    "error": result.error,
                    "duration_ms": result.duration_ms,
                })
            except Exception as e:
                results.append({
                    "tool": tool_name,
                    "success": False,
                    "error": str(e),
                })

        return results

    def _evaluate_condition(
        self,
        condition: dict,
        query: str,
        state: AgentState,
        task_params: dict | None,
    ) -> bool:
        """
        Evaluate a rule condition.

        Args:
            condition: Condition configuration
            query: Lowercased query string
            state: Current state
            task_params: Task parameters

        Returns:
            True if condition matches
        """
        condition_type = condition.get("type", "contains")

        if condition_type == "contains":
            keywords = condition.get("keywords", [])
            return any(kw.lower() in query for kw in keywords)

        elif condition_type == "matches":
            import re
            pattern = condition.get("pattern", "")
            return bool(re.search(pattern, query, re.IGNORECASE))

        elif condition_type == "equals":
            value = condition.get("value", "")
            return query == value.lower()

        elif condition_type == "always":
            return True

        return False

    async def _execute_action(
        self,
        action: dict,
        state: AgentState,
        task_params: dict | None,
    ) -> SubAgentResult:
        """
        Execute a rule action.

        Args:
            action: Action configuration
            state: Current state
            task_params: Task parameters

        Returns:
            SubAgentResult
        """
        action_type = action.get("type", "respond")

        if action_type == "respond":
            return SubAgentResult(
                agent_name=self.name,
                status="success",
                data=action.get("response", ""),
            )

        elif action_type == "tool":
            tool_name = action.get("tool")
            tool_params = action.get("params", {})

            tool = self.get_tool(tool_name)
            if not tool:
                return SubAgentResult(
                    agent_name=self.name,
                    status="failed",
                    error=f"Tool '{tool_name}' not found",
                )

            context = ToolContext(
                service_tokens=state["request_context"].service_tokens,
                tenant_id=state["request_context"].tenant_id,
                user_id=state["request_context"].user_id,
                request_id=state["request_context"].request_id,
            )

            result = await tool.execute_with_validation(tool_params, context)
            return SubAgentResult(
                agent_name=self.name,
                status="success" if result.success else "failed",
                data=result.data,
                error=result.error,
            )

        elif action_type == "delegate":
            # Delegate to another agent (for future implementation)
            target_agent = action.get("agent")
            return SubAgentResult(
                agent_name=self.name,
                status="partial",
                data={"delegate_to": target_agent},
            )

        return SubAgentResult(
            agent_name=self.name,
            status="failed",
            error=f"Unknown action type: {action_type}",
        )

    def _parse_structured_output(
        self,
        content: str,
        schema: dict,
    ) -> Any:
        """
        Parse structured output according to schema.

        Args:
            content: Raw content
            schema: Output schema

        Returns:
            Parsed content
        """
        # Try to extract JSON from content
        import json
        import re

        # Try to find JSON in content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Return raw content if parsing fails
        return content


class DynamicAgentFactory:
    """Factory for creating DynamicAgent instances from definitions."""

    @staticmethod
    def create(definition: AgentDefinition) -> DynamicAgent:
        """
        Create a DynamicAgent from a definition.

        Args:
            definition: Agent definition

        Returns:
            DynamicAgent instance
        """
        return DynamicAgent(definition)

    @staticmethod
    def create_from_name(name: str) -> DynamicAgent | None:
        """
        Create a DynamicAgent from a registered definition name.

        Args:
            name: Agent name

        Returns:
            DynamicAgent instance or None if not found
        """
        registry = get_agent_registry()
        definition = registry.get_definition(name)
        if not definition:
            return None
        return DynamicAgent(definition)


def get_dynamic_agent(name: str) -> DynamicAgent | None:
    """
    Get a dynamic agent by name.

    Args:
        name: Agent name

    Returns:
        DynamicAgent instance or None
    """
    return DynamicAgentFactory.create_from_name(name)


def get_all_available_agents() -> list[SubAgentBase]:
    """
    Get all available agents (static and dynamic).

    Returns:
        List of agent instances
    """
    agents = []
    registry = get_agent_registry()

    # Get static agents
    for agent in registry.list_enabled():
        agents.append(agent)

    # Get dynamic agents
    for definition in registry.list_enabled_definitions():
        agents.append(DynamicAgentFactory.create(definition))

    # Sort by priority (higher first)
    agents.sort(key=lambda a: getattr(a, "priority", 0), reverse=True)

    return agents


def get_agents_by_capability(capability: str) -> list[SubAgentBase]:
    """
    Get all agents with a specific capability.

    Args:
        capability: Capability to match

    Returns:
        List of matching agents
    """
    agents = []
    registry = get_agent_registry()

    # Get static agents
    for agent in registry.list_by_capability(capability):
        if registry.is_enabled(agent.name):
            agents.append(agent)

    # Get dynamic agents
    for definition in registry.list_definitions_by_capability(capability):
        if definition.enabled:
            agents.append(DynamicAgentFactory.create(definition))

    return agents
