"""ServiceNow Table Detail Tool."""

import httpx
from pydantic import Field
import structlog

from src.agents.tools.base import ToolBase, ToolResult, ToolContext, ToolParameters

logger = structlog.get_logger()


class TableDetailParams(ToolParameters):
    """Parameters for table detail retrieval."""

    table_name: str = Field(..., description="Table name (e.g., kb_knowledge, incident)")
    sys_id: str = Field(..., description="Record sys_id")
    fields: list[str] = Field(default_factory=list, description="Fields to retrieve")


class ServiceNowTableDetailTool(ToolBase):
    """
    Tool for retrieving ServiceNow table record details.

    Uses ServiceNow Table API to get detailed record information.
    """

    def __init__(self):
        """Initialize the tool."""
        super().__init__(
            name="servicenow_table_detail",
            description="ServiceNowテーブルの詳細情報を取得します",
            required_service_token="servicenow",
            parameters_schema=TableDetailParams,
            timeout_seconds=30,
        )

    async def execute(self, params: dict, context: ToolContext) -> ToolResult:
        """
        Execute table detail retrieval.

        Args:
            params: Request parameters
            context: Execution context

        Returns:
            ToolResult with record details
        """
        table_name = params.get("table_name", "")
        sys_id = params.get("sys_id", "")
        fields = params.get("fields", [])

        if not table_name or not sys_id:
            return ToolResult(
                success=False,
                error="table_name and sys_id are required",
            )

        logger.debug(
            "servicenow_table_detail",
            table_name=table_name,
            sys_id=sys_id,
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
        base_url = f"https://{instance}/api/now/table/{table_name}/{sys_id}"

        query_params = {
            "sysparm_display_value": "true",
        }
        if fields:
            query_params["sysparm_fields"] = ",".join(fields)

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

                if response.status_code == 404:
                    return ToolResult(
                        success=False,
                        error=f"Record not found: {table_name}/{sys_id}",
                    )

                if response.status_code != 200:
                    return ToolResult(
                        success=False,
                        error=f"ServiceNow API error: {response.status_code}",
                    )

                data = response.json()
                result = data.get("result", {})

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"table": table_name, "sys_id": sys_id},
                )

        except httpx.TimeoutException:
            return ToolResult(
                success=False,
                error="ServiceNow API timeout",
            )
        except Exception as e:
            logger.error(
                "servicenow_table_detail_error",
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
    registry.register(ServiceNowTableDetailTool())


# Auto-register on import
try:
    register()
except Exception:
    pass
