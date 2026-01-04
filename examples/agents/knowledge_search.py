"""Knowledge Search SubAgent."""

from typing import Any

import structlog

from src.agents.state import AgentState, SubAgentResult
from src.agents.sub_agents.base import SubAgentBase, RetryStrategy
from src.agents.tools.base import ToolContext
from src.agents.registry import get_tool_registry

logger = structlog.get_logger()


class KnowledgeSearchAgent(SubAgentBase):
    """
    SubAgent for searching ServiceNow knowledge base.

    This agent searches the official ServiceNow knowledge base
    for articles and case information related to user queries.
    """

    def __init__(self):
        """Initialize the KnowledgeSearchAgent."""
        # Get tools from registry
        tool_registry = get_tool_registry()
        tools = [
            tool for tool in [
                tool_registry.get("servicenow_knowledge_search"),
                tool_registry.get("servicenow_case_search"),
            ]
            if tool is not None
        ]

        super().__init__(
            name="knowledge_search",
            description="ServiceNow公式ナレッジベースを検索",
            capabilities=["knowledge_search", "case_search"],
            tools=tools,
            retry_strategy=RetryStrategy(
                max_attempts=2,
                retry_conditions=["no_results"],
                query_modification="synonym",
                backoff_seconds=0.5,
            ),
        )

    async def execute(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute knowledge search.

        Args:
            state: Current agent state
            task_params: Task parameters including query

        Returns:
            SubAgentResult with search results
        """
        params = task_params or {}
        query = params.get("query", state["user_input"])

        logger.info(
            "knowledge_search_executing",
            session_id=state["session_id"],
            query=query,
        )

        # Build tool context
        context = ToolContext(
            service_tokens=state["request_context"].service_tokens,
            tenant_id=state["request_context"].tenant_id,
            user_id=state["request_context"].user_id,
            request_id=state["request_context"].request_id,
        )

        results = []
        errors = []

        # Search knowledge base
        kb_tool = self.get_tool("servicenow_knowledge_search")
        if kb_tool:
            try:
                kb_result = await kb_tool.execute_with_validation(
                    {"query": query, "limit": 5},
                    context,
                )
                if kb_result.success and kb_result.data:
                    results.extend(kb_result.data)
                elif not kb_result.success:
                    errors.append(f"KB検索エラー: {kb_result.error}")
            except Exception as e:
                errors.append(f"KB検索例外: {str(e)}")

        # Search cases if no KB results
        if len(results) < 2:
            case_tool = self.get_tool("servicenow_case_search")
            if case_tool:
                try:
                    case_result = await case_tool.execute_with_validation(
                        {"query": query, "limit": 3},
                        context,
                    )
                    if case_result.success and case_result.data:
                        results.extend(case_result.data)
                    elif not case_result.success:
                        errors.append(f"ケース検索エラー: {case_result.error}")
                except Exception as e:
                    errors.append(f"ケース検索例外: {str(e)}")

        # Determine status
        if results:
            status = "success"
        elif errors:
            status = "failed"
        else:
            status = "partial"

        return SubAgentResult(
            agent_name=self.name,
            status=status,
            data=results if results else None,
            error="; ".join(errors) if errors else None,
        )


# Register agent
def register():
    """Register the agent with the registry."""
    from src.agents.registry import get_agent_registry
    registry = get_agent_registry()
    registry.register(KnowledgeSearchAgent())


# Auto-register on import
try:
    register()
except Exception:
    pass  # Registry may not be initialized yet
