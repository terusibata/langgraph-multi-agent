"""Catalog List Tool."""

import httpx
from pydantic import Field
import structlog

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters

logger = structlog.get_logger()


class CatalogListParams(ToolParameters):
    """Parameters for catalog list retrieval."""

    query: str | None = Field(default=None, description="Search query")
    category: str | None = Field(default=None, description="Category filter")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum results")


class CatalogListTool(ToolBase):
    """
    Tool for listing ServiceNow service catalog items.

    Uses ServiceNow Table API to search sc_cat_item table.
    """

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="catalog_list",
            description="ServiceNowサービスカタログの一覧を取得します",
            required_service_token="servicenow",
            parameters_schema=CatalogListParams,
            timeout_seconds=30,
        )

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute catalog list retrieval.

        Args:
            params: Request parameters
            context: Execution context

        Returns:
            ToolResult with catalog items
        """
        query = params.get("query")
        category = params.get("category")
        limit = params.get("limit", 10)

        logger.debug(
            "catalog_list",
            query=query,
            category=category,
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
        base_url = f"https://{instance}/api/now/table/sc_cat_item"

        # Build query
        sysparm_query = "active=true"
        if query:
            sysparm_query += f"^nameLIKE{query}^ORshort_descriptionLIKE{query}"
        if category:
            sysparm_query += f"^category={category}"

        query_params = {
            "sysparm_limit": limit,
            "sysparm_fields": "sys_id,name,short_description,category,price,order,sc_catalogs",
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
                        "name": item.get("name", ""),
                        "short_description": item.get("short_description", ""),
                        "category": item.get("category", ""),
                        "price": item.get("price", ""),
                        "catalogs": item.get("sc_catalogs", ""),
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
                "catalog_list_error",
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
    registry.register(CatalogListTool())


# Auto-register on import
try:
    register()
except Exception:
    pass
