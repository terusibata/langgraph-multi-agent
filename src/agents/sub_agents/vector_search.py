"""Vector Search SubAgent."""

from typing import Any

import structlog

from src.agents.state import AgentState, SubAgentResult
from src.agents.sub_agents.base import SubAgentBase, RetryStrategy
from src.agents.tools.base import ToolContext
from src.agents.registry import get_tool_registry

logger = structlog.get_logger()


class VectorSearchAgent(SubAgentBase):
    """
    SubAgent for semantic vector search.

    This agent searches pre-built knowledge vectors using
    semantic similarity for more flexible query matching.
    """

    def __init__(self):
        """Initialize the VectorSearchAgent."""
        # Get tools from registry
        tool_registry = get_tool_registry()
        tools = [
            tool for tool in [
                tool_registry.get("vector_db_search"),
            ]
            if tool is not None
        ]

        super().__init__(
            name="vector_search",
            description="事前構築済みナレッジのベクトル検索",
            capabilities=["semantic_search", "similarity_search"],
            tools=tools,
            retry_strategy=RetryStrategy(
                max_attempts=3,
                retry_conditions=["no_results", "low_relevance"],
                query_modification="llm_rewrite",
                backoff_seconds=0.5,
            ),
        )

        # Relevance threshold for results
        self.relevance_threshold = 0.7

    async def execute(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute vector search.

        Args:
            state: Current agent state
            task_params: Task parameters including query

        Returns:
            SubAgentResult with search results
        """
        params = task_params or {}
        query = params.get("query", state["user_input"])
        top_k = params.get("top_k", 5)
        threshold = params.get("threshold", self.relevance_threshold)

        logger.info(
            "vector_search_executing",
            session_id=state["session_id"],
            query=query,
            top_k=top_k,
            threshold=threshold,
        )

        # Build tool context
        context = ToolContext(
            service_tokens=state["request_context"].service_tokens,
            tenant_id=state["request_context"].tenant_id,
            user_id=state["request_context"].user_id,
            request_id=state["request_context"].request_id,
        )

        # Execute vector search
        search_tool = self.get_tool("vector_db_search")
        if not search_tool:
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error="vector_db_search tool not available",
            )

        try:
            result = await search_tool.execute_with_validation(
                {
                    "query": query,
                    "top_k": top_k,
                    "threshold": threshold,
                },
                context,
            )

            if not result.success:
                return SubAgentResult(
                    agent_name=self.name,
                    status="failed",
                    error=result.error,
                )

            # Filter results by relevance threshold
            if result.data:
                filtered_results = [
                    item for item in result.data
                    if self._get_score(item) >= threshold
                ]

                if filtered_results:
                    return SubAgentResult(
                        agent_name=self.name,
                        status="success",
                        data=filtered_results,
                    )
                else:
                    # Results exist but below threshold
                    return SubAgentResult(
                        agent_name=self.name,
                        status="partial",
                        data=result.data[:3],  # Return top 3 anyway
                    )

            # No results
            return SubAgentResult(
                agent_name=self.name,
                status="partial",
                data=None,
            )

        except Exception as e:
            logger.error(
                "vector_search_error",
                session_id=state["session_id"],
                error=str(e),
            )
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error=str(e),
            )

    def _get_score(self, item: Any) -> float:
        """Extract relevance score from result item."""
        if isinstance(item, dict):
            return item.get("score", item.get("relevance", item.get("similarity", 0.0)))
        return 0.0

    async def _apply_llm_rewrite(self, original_query: str, attempt: int) -> str:
        """
        Use LLM to rewrite the query for better semantic matching.

        For now, uses rule-based modification as fallback.
        In production, this would call the LLM.
        """
        # Rule-based modifications for now
        modifications = [
            lambda q: f"{q} 解決方法",
            lambda q: f"{q} 原因 対処",
            lambda q: q.replace("できない", "エラー").replace("しない", "失敗"),
        ]

        idx = (attempt - 2) % len(modifications)
        return modifications[idx](original_query)


# Register agent
def register():
    """Register the agent with the registry."""
    from src.agents.registry import get_agent_registry
    registry = get_agent_registry()
    registry.register(VectorSearchAgent())


# Auto-register on import
try:
    register()
except Exception:
    pass  # Registry may not be initialized yet
