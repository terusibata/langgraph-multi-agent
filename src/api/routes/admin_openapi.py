"""Admin API routes for OpenAPI import."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
import structlog

from src.api.middleware.access_key import RequestContext, check_permission
from src.api.schemas.admin import (
    OpenAPIImportRequest,
    OpenAPIImportResponse,
)
from src.agents.tools.openapi import (
    OpenAPIParser,
    OpenAPIToolGenerator,
    OpenAPIParseError,
)

logger = structlog.get_logger()

router = APIRouter(tags=["Admin - OpenAPI"])


@router.post(
    "/openapi/import",
    response_model=OpenAPIImportResponse,
    summary="Import tools from OpenAPI specification",
)
async def import_openapi_spec(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    body: OpenAPIImportRequest,
):
    """
    Import tools from an OpenAPI specification.

    The specification can be provided as either:
    - YAML/JSON string in the `spec` field
    - URL to fetch the specification from in the `spec_url` field
    """
    generator = OpenAPIToolGenerator()

    try:
        if body.spec_url:
            # Fetch from URL
            tools = await generator.generate_from_url(
                body.spec_url,
                options={
                    "base_url": body.base_url,
                    "prefix": body.prefix,
                    "include_operations": body.include_operations,
                    "exclude_operations": body.exclude_operations,
                    "include_tags": body.include_tags,
                    "exclude_tags": body.exclude_tags,
                    "auth_config": body.auth_config.model_dump() if body.auth_config else None,
                    "enabled": body.enabled,
                    "register": True,
                },
            )
        elif body.spec:
            # Parse from string
            tools = await generator.generate_tools_from_spec(
                body.spec,
                options={
                    "base_url": body.base_url,
                    "prefix": body.prefix,
                    "include_operations": body.include_operations,
                    "exclude_operations": body.exclude_operations,
                    "include_tags": body.include_tags,
                    "exclude_tags": body.exclude_tags,
                    "auth_config": body.auth_config.model_dump() if body.auth_config else None,
                    "enabled": body.enabled,
                    "register": True,
                },
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'spec' or 'spec_url' must be provided",
            )

    except OpenAPIParseError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse OpenAPI specification: {str(e)}",
        )
    except Exception as e:
        logger.error("openapi_import_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import tools: {str(e)}",
        )

    logger.info(
        "openapi_tools_imported",
        tools_count=len(tools),
        imported_by=context.user_id,
    )

    return OpenAPIImportResponse(
        success=True,
        tools_created=[t.name for t in tools],
        message=f"Successfully imported {len(tools)} tools",
    )


@router.post(
    "/openapi/upload",
    response_model=OpenAPIImportResponse,
    summary="Upload and import OpenAPI specification file",
)
async def upload_openapi_spec(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:write"))],
    file: UploadFile = File(..., description="OpenAPI specification file (YAML or JSON)"),
    base_url: str | None = None,
    prefix: str | None = None,
    enabled: bool = True,
):
    """
    Upload an OpenAPI specification file and import tools.

    Supports both YAML and JSON formats.
    """
    # Read file content
    content = await file.read()

    try:
        spec_content = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File must be valid UTF-8 encoded",
        )

    # Determine if JSON or YAML
    generator = OpenAPIToolGenerator()

    try:
        tools = await generator.generate_tools_from_spec(
            spec_content,
            options={
                "base_url": base_url,
                "prefix": prefix,
                "enabled": enabled,
                "register": True,
            },
        )
    except OpenAPIParseError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse OpenAPI specification: {str(e)}",
        )
    except Exception as e:
        logger.error("openapi_upload_error", error=str(e), filename=file.filename)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import tools: {str(e)}",
        )

    logger.info(
        "openapi_file_imported",
        filename=file.filename,
        tools_count=len(tools),
        imported_by=context.user_id,
    )

    return OpenAPIImportResponse(
        success=True,
        tools_created=[t.name for t in tools],
        message=f"Successfully imported {len(tools)} tools from {file.filename}",
    )


@router.post(
    "/openapi/validate",
    summary="Validate an OpenAPI specification",
)
async def validate_openapi_spec(
    context: Annotated[RequestContext, Depends(check_permission("admin:tools:read"))],
    body: OpenAPIImportRequest,
):
    """
    Validate an OpenAPI specification without importing.

    Returns information about the operations that would be imported.
    """
    try:
        if body.spec_url:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.get(body.spec_url)
                response.raise_for_status()
                spec_content = response.text
                if "json" in response.headers.get("content-type", ""):
                    spec = response.json()
                else:
                    spec = spec_content
        elif body.spec:
            spec = body.spec
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either 'spec' or 'spec_url' must be provided",
            )

        # Parse spec
        if isinstance(spec, str):
            parser = OpenAPIParser.from_yaml(spec)
        else:
            parser = OpenAPIParser.from_json(spec)

        # Get operations
        operations = parser.parse_all_operations()

        # Build operation info
        operation_info = []
        for op in operations:
            op_data = op["operation"]
            operation_info.append({
                "path": op["path"],
                "method": op["method"],
                "operation_id": op_data.get("operationId"),
                "summary": op_data.get("summary"),
                "tags": op_data.get("tags", []),
                "parameters_count": len(op_data.get("parameters", [])),
                "has_request_body": "requestBody" in op_data,
            })

        return {
            "valid": True,
            "api_title": parser.info.get("title", "Unknown"),
            "api_version": parser.info.get("version", ""),
            "openapi_version": parser.spec.get("openapi", ""),
            "servers": parser.servers,
            "operations_count": len(operations),
            "operations": operation_info,
        }

    except OpenAPIParseError as e:
        return {
            "valid": False,
            "error": str(e),
        }
    except Exception as e:
        logger.error("openapi_validation_error", error=str(e))
        return {
            "valid": False,
            "error": str(e),
        }
