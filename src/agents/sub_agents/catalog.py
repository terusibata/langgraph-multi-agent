"""Catalog SubAgent."""

from typing import Any

import structlog

from src.agents.state import AgentState, SubAgentResult
from src.agents.sub_agents.base import SubAgentBase, RetryStrategy
from src.agents.tools.base import ToolContext
from src.agents.registry import get_tool_registry

logger = structlog.get_logger()


class CatalogAgent(SubAgentBase):
    """
    SubAgent for ServiceNow catalog operations.

    This agent searches the service catalog to find appropriate
    request forms and guides users to the right submission channels.
    """

    def __init__(self):
        """Initialize the CatalogAgent."""
        # Get tools from registry
        tool_registry = get_tool_registry()
        tools = [
            tool for tool in [
                tool_registry.get("catalog_list"),
                tool_registry.get("catalog_detail"),
            ]
            if tool is not None
        ]

        super().__init__(
            name="catalog",
            description="ServiceNowカタログから適切な申請先を調査",
            capabilities=["catalog_search", "catalog_recommendation"],
            tools=tools,
            retry_strategy=RetryStrategy(
                max_attempts=1,  # No retry for catalog
                retry_conditions=[],
                query_modification="none",
                backoff_seconds=0,
            ),
        )

    async def execute(
        self,
        state: AgentState,
        task_params: dict | None = None,
    ) -> SubAgentResult:
        """
        Execute catalog search and recommendation.

        Args:
            state: Current agent state
            task_params: Task parameters

        Returns:
            SubAgentResult with catalog recommendations
        """
        params = task_params or {}
        query = params.get("query", state["user_input"])
        category = params.get("category")

        logger.info(
            "catalog_search_executing",
            session_id=state["session_id"],
            query=query,
            category=category,
        )

        # Build tool context
        context = ToolContext(
            service_tokens=state["request_context"].service_tokens,
            tenant_id=state["request_context"].tenant_id,
            user_id=state["request_context"].user_id,
            request_id=state["request_context"].request_id,
        )

        # Step 1: Get catalog list
        list_tool = self.get_tool("catalog_list")
        if not list_tool:
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error="catalog_list tool not available",
            )

        try:
            list_result = await list_tool.execute_with_validation(
                {"query": query, "category": category, "limit": 10},
                context,
            )

            if not list_result.success:
                return SubAgentResult(
                    agent_name=self.name,
                    status="failed",
                    error=list_result.error,
                )

            catalogs = list_result.data or []

            if not catalogs:
                return SubAgentResult(
                    agent_name=self.name,
                    status="partial",
                    data={"message": "該当するカタログが見つかりませんでした"},
                )

            # Step 2: Get details for top recommendations
            detail_tool = self.get_tool("catalog_detail")
            detailed_catalogs = []

            for catalog in catalogs[:3]:  # Top 3
                if detail_tool:
                    try:
                        catalog_id = catalog.get("sys_id", catalog.get("id"))
                        if catalog_id:
                            detail_result = await detail_tool.execute_with_validation(
                                {"catalog_id": catalog_id},
                                context,
                            )
                            if detail_result.success and detail_result.data:
                                detailed_catalogs.append(detail_result.data)
                            else:
                                detailed_catalogs.append(catalog)
                        else:
                            detailed_catalogs.append(catalog)
                    except Exception:
                        detailed_catalogs.append(catalog)
                else:
                    detailed_catalogs.append(catalog)

            # Build recommendation
            recommendation = self._build_recommendation(detailed_catalogs, query)

            return SubAgentResult(
                agent_name=self.name,
                status="success",
                data=recommendation,
            )

        except Exception as e:
            logger.error(
                "catalog_search_error",
                session_id=state["session_id"],
                error=str(e),
            )
            return SubAgentResult(
                agent_name=self.name,
                status="failed",
                error=str(e),
            )

    def _build_recommendation(self, catalogs: list, query: str) -> dict:
        """Build a recommendation response from catalog data."""
        recommendations = []

        for catalog in catalogs:
            rec = {
                "name": catalog.get("name", catalog.get("title", "不明")),
                "description": catalog.get("short_description", catalog.get("description", "")),
                "category": catalog.get("category", ""),
                "sys_id": catalog.get("sys_id", catalog.get("id", "")),
            }

            # Add URL if available
            if "url" in catalog:
                rec["url"] = catalog["url"]

            # Add required fields info
            if "variables" in catalog:
                rec["required_fields"] = [
                    v.get("name") for v in catalog["variables"]
                    if v.get("mandatory")
                ]

            recommendations.append(rec)

        return {
            "query": query,
            "recommendations": recommendations,
            "total_found": len(catalogs),
            "suggestion": self._generate_suggestion(recommendations, query),
        }

    def _generate_suggestion(self, recommendations: list, query: str) -> str:
        """Generate a suggestion message."""
        if not recommendations:
            return "該当する申請先が見つかりませんでした。サポート窓口にお問い合わせください。"

        if len(recommendations) == 1:
            name = recommendations[0].get("name", "申請フォーム")
            return f"「{name}」から申請することをお勧めします。"

        names = [r.get("name", "") for r in recommendations[:2]]
        return f"「{names[0]}」または「{names[1]}」からの申請をお勧めします。"


# Register agent
def register():
    """Register the agent with the registry."""
    from src.agents.registry import get_agent_registry
    registry = get_agent_registry()
    registry.register(CatalogAgent())


# Auto-register on import
try:
    register()
except Exception:
    pass  # Registry may not be initialized yet
