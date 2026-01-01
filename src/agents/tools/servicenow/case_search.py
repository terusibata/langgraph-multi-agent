"""ServiceNow Case Search Tool."""

import httpx
from pydantic import Field
import structlog

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters

logger = structlog.get_logger()


class CaseSearchParams(ToolParameters):
    """Parameters for case search."""

    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")
    state: str | None = Field(default=None, description="Case state filter")


class ServiceNowCaseSearchTool(ToolBase):
    """
    Tool for searching ServiceNow cases/incidents.

    Uses ServiceNow Table API to search incident table.
    """

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="servicenow_case_search",
            description="ServiceNowのケース（チケット）を検索します",
            required_service_token="servicenow",
            parameters_schema=CaseSearchParams,
            timeout_seconds=30,
        )

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute case search.

        Args:
            params: Search parameters
            context: Execution context

        Returns:
            ToolResult with search results
        """
        query = params.get("query", "")
        limit = params.get("limit", 10)
        state = params.get("state")

        logger.debug(
            "servicenow_case_search",
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
        base_url = f"https://{instance}/api/now/table/incident"

        # Build query
        sysparm_query = f"short_descriptionLIKE{query}^ORdescriptionLIKE{query}"
        if state:
            sysparm_query += f"^state={state}"

        query_params = {
            "sysparm_limit": limit,
            "sysparm_fields": "sys_id,number,short_description,description,state,priority,sys_created_on,resolved_at,close_notes",
            "sysparm_query": sysparm_query,
            "sysparm_display_value": "true",
        }

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
                        "number": item.get("number", ""),
                        "title": item.get("short_description", ""),
                        "description": item.get("description", "")[:300],
                        "state": item.get("state", ""),
                        "priority": item.get("priority", ""),
                        "created_at": item.get("sys_created_on", ""),
                        "resolved_at": item.get("resolved_at", ""),
                        "resolution": item.get("close_notes", "")[:200],
                        "source": f"INC{item.get('number', '')}",
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
                "servicenow_case_search_error",
                error=str(e),
                request_id=context.request_id,
            )
            return ToolResult(
                success=False,
                error=f"ServiceNow API error: {str(e)}",
            )


# Register tool
def register():
    """Register the tool with the registry."""
    from src.agents.registry import get_tool_registry
    registry = get_tool_registry()
    registry.register(ServiceNowCaseSearchTool())


# Auto-register on import
try:
    register()
except Exception:
    pass
