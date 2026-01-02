"""Ad-hoc agent that is dynamically generated at runtime."""

from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import structlog

from src.agents.state import AdHocAgentSpec, AgentState, SubAgentResult
from src.agents.tools.base import ToolBase, ToolContext
from src.agents.registry import get_tool_registry
from src.agents.tools.dynamic import DynamicToolFactory
from src.services.llm import get_llm
from src.config import get_settings

logger = structlog.get_logger()


class AdHocAgent:
    """
    Ad-hoc agent that is dynamically generated at runtime.

    Unlike pre-defined agents, ad-hoc agents are created on-the-fly
    by the Planner based on the available tools and task requirements.
    They execute a specific purpose using a selected set of tools.
    """

    def __init__(self, spec: AdHocAgentSpec, model_id: str | None = None):
        """
        Initialize the ad-hoc agent.

        Args:
            spec: The ad-hoc agent specification
            model_id: Optional model ID (defaults to settings)
        """
        self.spec = spec
        self.name = spec.name
        self.model_id = model_id or get_settings().sub_agent_model_id
        self._tools: list[ToolBase] = []
        self._tool_map: dict[str, ToolBase] = {}
        self._initialize_tools()

    def _initialize_tools(self) -> None:
        """Initialize tools from the specification."""
        tool_registry = get_tool_registry()

        for tool_name in self.spec.tools:
            # Try to get static tool first
            tool = tool_registry.get(tool_name)

            if not tool:
                # Try to create dynamic tool
                definition = tool_registry.get_definition(tool_name)
                if definition:
                    tool = DynamicToolFactory.create(definition)

            if tool:
                self._tools.append(tool)
                self._tool_map[tool_name] = tool
            else:
                logger.warning(
                    "adhoc_agent_tool_not_found",
                    agent=self.name,
                    tool=tool_name,
                )

    def get_tool(self, name: str) -> ToolBase | None:
        """Get a tool by name."""
        return self._tool_map.get(name)

    @property
    def tools(self) -> list[ToolBase]:
        """Get all tools for this agent."""
        return self._tools

    def get_system_prompt(self) -> str:
        """Generate the system prompt for this agent."""
        # If a custom system prompt is provided, use it
        if self.spec.system_prompt:
            return self.spec.system_prompt

        # Otherwise, generate a default one
        tool_descriptions = "\n".join(
            f"- {tool.name}: {tool.description}"
            for tool in self._tools
        )

        return f"""あなたは特定のタスクを実行するために動的に生成されたエージェントです。

## あなたの目的
{self.spec.purpose}

## 利用可能なツール
{tool_descriptions}

## 期待される出力
{self.spec.expected_output}

## 指示
1. 与えられた目的を達成するために、適切なツールを使用してください
2. ツールの実行結果を分析し、目的に沿った情報を抽出してください
3. 結果を構造化された形式で報告してください
4. エラーが発生した場合は、詳細な情報を報告してください

タスクパラメータに基づいて実行を開始してください。
"""

    async def execute(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute the ad-hoc agent's task.

        Args:
            state: Current agent state
            task_params: Optional parameters for the task

        Returns:
            SubAgentResult with execution status and data
        """
        started_at = datetime.utcnow()
        task_params = task_params or {}

        logger.info(
            "adhoc_agent_execute_start",
            agent=self.name,
            purpose=self.spec.purpose,
            tools=self.spec.tools,
            session_id=state["session_id"],
        )

        try:
            # Create tool context from state
            context = self._create_tool_context(state)

            # Execute using LLM with tools
            result_data = await self._execute_with_llm(
                state,
                task_params,
                context,
            )

            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            status = "success" if result_data else "partial"

            logger.info(
                "adhoc_agent_execute_complete",
                agent=self.name,
                status=status,
                duration_ms=duration_ms,
                session_id=state["session_id"],
            )

            return SubAgentResult(
                agent_name=self.name,
                status=status,
                data=result_data,
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

        except Exception as e:
            completed_at = datetime.utcnow()
            duration_ms = int((completed_at - started_at).total_seconds() * 1000)

            logger.error(
                "adhoc_agent_execute_error",
                agent=self.name,
                error=str(e),
                session_id=state["session_id"],
            )

            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error=str(e),
                started_at=started_at,
                completed_at=completed_at,
                duration_ms=duration_ms,
            )

    async def execute_with_retry(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute with retry logic.

        For ad-hoc agents, we use a simplified retry strategy.
        """
        max_attempts = 2
        last_result: SubAgentResult | None = None

        for attempt in range(1, max_attempts + 1):
            result = await self.execute(state, task_params)

            if result.status == "success":
                result.retry_count = attempt - 1
                return result

            last_result = result

            if attempt < max_attempts:
                logger.info(
                    "adhoc_agent_retry",
                    agent=self.name,
                    attempt=attempt,
                )

        if last_result:
            last_result.retry_count = max_attempts - 1
            return last_result

        return SubAgentResult(
            agent_name=self.name,
            status="failed",
            error="All retry attempts exhausted",
        )

    def _create_tool_context(self, state: AgentState) -> ToolContext:
        """Create tool context from agent state."""
        request_context = state["request_context"]
        return ToolContext(
            service_tokens=request_context.service_tokens,
            tenant_id=request_context.tenant_id,
            user_id=request_context.user_id,
            request_id=request_context.request_id,
        )

    async def _execute_with_llm(
        self,
        state: AgentState,
        task_params: dict,
        context: ToolContext,
    ) -> Any:
        """Execute using LLM with tool calling."""
        llm = get_llm(self.model_id)

        # Build messages
        messages = [
            SystemMessage(content=self.get_system_prompt()),
            HumanMessage(content=self._format_task_message(state, task_params)),
        ]

        # If we have tools, bind them to the LLM
        if self._tools:
            langchain_tools = [tool.to_langchain_tool() for tool in self._tools]
            llm_with_tools = llm.bind_tools(langchain_tools)

            # Execute LLM with tools
            response = await llm_with_tools.ainvoke(messages)

            # Process tool calls if any
            if hasattr(response, "tool_calls") and response.tool_calls:
                tool_results = await self._execute_tool_calls(
                    response.tool_calls,
                    context,
                )
                return {
                    "response": response.content,
                    "tool_results": tool_results,
                }
            else:
                return {
                    "response": response.content,
                    "tool_results": [],
                }
        else:
            # No tools, just use LLM
            response = await llm.ainvoke(messages)
            return {
                "response": response.content,
                "tool_results": [],
            }

    def _format_task_message(self, state: AgentState, task_params: dict) -> str:
        """Format the task message for the LLM."""
        user_input = state["user_input"]
        query = task_params.get("query", user_input)

        message_parts = [
            f"ユーザーの質問: {user_input}",
        ]

        if query != user_input:
            message_parts.append(f"検索クエリ: {query}")

        if task_params:
            params_str = "\n".join(
                f"- {k}: {v}" for k, v in task_params.items() if k != "query"
            )
            if params_str:
                message_parts.append(f"追加パラメータ:\n{params_str}")

        return "\n\n".join(message_parts)

    async def _execute_tool_calls(
        self,
        tool_calls: list,
        context: ToolContext,
    ) -> list[dict]:
        """Execute tool calls and collect results."""
        results = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("args", {})

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


class AdHocAgentFactory:
    """Factory for creating ad-hoc agents from specifications."""

    @staticmethod
    def create(spec: AdHocAgentSpec, model_id: str | None = None) -> AdHocAgent:
        """
        Create an ad-hoc agent from a specification.

        Args:
            spec: The ad-hoc agent specification
            model_id: Optional model ID

        Returns:
            AdHocAgent instance
        """
        return AdHocAgent(spec, model_id)

    @staticmethod
    def create_from_dict(spec_dict: dict, model_id: str | None = None) -> AdHocAgent:
        """
        Create an ad-hoc agent from a dictionary specification.

        Args:
            spec_dict: Dictionary containing spec fields
            model_id: Optional model ID

        Returns:
            AdHocAgent instance
        """
        spec = AdHocAgentSpec(**spec_dict)
        return AdHocAgent(spec, model_id)
