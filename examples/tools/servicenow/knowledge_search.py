"""ServiceNow Knowledge Search Tool."""

from typing import Optional

import httpx
from pydantic import BaseModel, Field
import structlog

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters
from src.config import get_settings

logger = structlog.get_logger()


class KnowledgeSearchParams(ToolParameters):
    """Parameters for knowledge search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    categories: list[str] = Field(default_factory=list, description="Category filter")


class ServiceNowKnowledgeSearchTool(ToolBase):
    """
    Tool for searching ServiceNow knowledge base.

    Uses ServiceNow Table API to search kb_knowledge table.
    """

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="servicenow_knowledge_search",
            description="ServiceNowのナレッジベースを検索します",
            required_service_token="servicenow",
            parameters_schema=KnowledgeSearchParams,
            timeout_seconds=30,
        )

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute knowledge base search.

        Args:
            params: Search parameters
            context: Execution context

        Returns:
            ToolResult with search results
        """
        query = params.get("query", "")
        limit = params.get("limit", 10)
        categories = params.get("categories", [])

        logger.debug(
            "servicenow_kb_search",
            query=query,
            limit=limit,
            request_id=context.request_id,
        )

        # Get ServiceNow credentials
        token = context.get_token("servicenow")
        instance = context.get_instance("servicenow")

        if not token or not instance:
            return ToolResult(
                success=False,
                error="ServiceNow credentials not available",
            )

        # Build API request
        base_url = f"https://{instance}/api/now/table/kb_knowledge"

        # Build query parameters
        query_params = {
            "sysparm_limit": limit,
            "sysparm_fields": "sys_id,short_description,text,number,sys_created_on,category",
            "sysparm_query": f"short_descriptionLIKE{query}^ORtextLIKE{query}^workflow_state=published",
        }

        if categories:
            category_query = "^".join([f"category={cat}" for cat in categories])
            query_params["sysparm_query"] += f"^{category_query}"

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.get(
                    base_url,
                    params=query_params,
                    headers=headers,
                )

                if response.status_code == 401:
                    return ToolResult(
                        success=False,
                        error="ServiceNow認証エラー: トークンが無効または期限切れです",
                    )

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"ServiceNow API error: {response.status_code}",
                    )

                data = response.json()
                results = data.get("result", [])

                # Transform results
                transformed = [
                    {
                        "sys_id": item.get("sys_id"),
                        "title": item.get("short_description", ""),
                        "content": self._clean_html(item.get("text", ""))[:500],
                        "number": item.get("number", ""),
                        "category": item.get("category", ""),
                        "created_at": item.get("sys_created_on", ""),
                        "source": f"KB{item.get('number', '')}",
                    }
                    for item in results
                ]

                return ToolResult(
                    success=True,
                    data=transformed,
                    metadata={"total": len(transformed), "query": query},
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="ServiceNow API timeout",
            )
        except Exception as e:
            logger.error(
                "servicenow_kb_search_error",
                error=str(e),
                request_id=context.request_id,
            )
            return ToolResult(
                success=False,
                error=f"ServiceNow API error: {str(e)}",
            )

    def _clean_html(self, text: str) -> str:
        """Remove HTML tags from text."""
        import re
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean)
        return clean.strip()


# Register tool
def register():
    """Register the tool with the registry."""
    from src.agents.registry import get_tool_registry
    registry = get_tool_registry()
    registry.register(ServiceNowKnowledgeSearchTool())


# Auto-register on import
try:
    register()
except Exception:
    pass
