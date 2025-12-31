"""Catalog Detail Tool."""

import httpx
from pydantic import Field
import structlog

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters

logger = structlog.get_logger()


class CatalogDetailParams(ToolParameters):
    """Parameters for catalog detail retrieval."""

    catalog_id: str = Field(..., description="Catalog item sys_id")


class CatalogDetailTool(ToolBase):
    """
    Tool for retrieving ServiceNow catalog item details.

    Gets detailed information including variables (form fields).
    """

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="catalog_detail",
            description="ServiceNowサービスカタログの詳細情報を取得します",
            required_service_token="servicenow",
            parameters_schema=CatalogDetailParams,
            timeout_seconds=30,
        )

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute catalog detail retrieval.

        Args:
            params: Request parameters
            context: Execution context

        Returns:
            ToolResult with catalog details
        """
        catalog_id = params.get("catalog_id", "")

        if not catalog_id:
            return ToolResult(
                success=False,
                error="catalog_id is required",
            )

        logger.debug(
            "catalog_detail",
            catalog_id=catalog_id,
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

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                # Get catalog item details
                item_url = f"https://{instance}/api/now/table/sc_cat_item/{catalog_id}"
                item_response = await client.get(
                    item_url,
                    params={"sysparm_display_value": "true"},
                    headers=headers,
                )

                if item_response.status_code == 401:
                    return ToolResult(
                        success=False,
                        error="ServiceNow認証エラー",
                    )

                if item_response.status_code == 404:
                    return ToolResult(
                        success=False,
                        error=f"Catalog item not found: {catalog_id}",
                    )

                if item_response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"ServiceNow API error: {item_response.status_code}",
                    )

                item_data = item_response.json().get("result", {})

                # Get catalog item variables (form fields)
                vars_url = f"https://{instance}/api/now/table/item_option_new"
                vars_response = await client.get(
                    vars_url,
                    params={
                        "sysparm_query": f"cat_item={catalog_id}",
                        "sysparm_fields": "sys_id,name,question_text,type,mandatory,order",
                        "sysparm_display_value": "true",
                    },
                    headers=headers,
                )

                variables = []
                if vars_response.status_code == 200:
                    variables = vars_response.json().get("result", [])

                # Build result
                result = {
                    "sys_id": item_data.get("sys_id"),
                    "name": item_data.get("name", ""),
                    "short_description": item_data.get("short_description", ""),
                    "description": item_data.get("description", ""),
                    "category": item_data.get("category", ""),
                    "price": item_data.get("price", ""),
                    "delivery_time": item_data.get("delivery_time", ""),
                    "url": f"https://{instance}/sp?id=sc_cat_item&sys_id={catalog_id}",
                    "variables": [
                        {
                            "name": var.get("name", ""),
                            "label": var.get("question_text", ""),
                            "type": var.get("type", ""),
                            "mandatory": var.get("mandatory", "false") == "true",
                            "order": var.get("order", 0),
                        }
                        for var in variables
                    ],
                }

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"catalog_id": catalog_id},
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="ServiceNow API timeout",
            )
        except Exception as e:
            logger.error(
                "catalog_detail_error",
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
    registry.register(CatalogDetailTool())


# Auto-register on import
try:
    register()
except Exception:
    pass
