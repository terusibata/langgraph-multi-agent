"""Access key verification middleware."""

from datetime import datetime, timezone
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from jose import JWTError, jwt
from pydantic import BaseModel, Field
import structlog

from src.config import get_settings

logger = structlog.get_logger()


class AccessKeyPayload(BaseModel):
    """Decoded access key payload."""

    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier")
    permissions: list[str] = Field(
        default_factory=lambda: ["agent:invoke"],
        description="Allowed operations",
    )
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")


class RequestContext(BaseModel):
    """Request context containing verified access key info."""

    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier")
    permissions: list[str] = Field(..., description="Allowed operations")
    request_id: str = Field(..., description="Unique request identifier")
    client_ip: str | None = Field(default=None, description="Client IP address")
    user_agent: str | None = Field(default=None, description="User agent string")


async def verify_access_key(request: Request) -> RequestContext:
    """
    Verify the access key from request headers.

    Args:
        request: FastAPI request object

    Returns:
        RequestContext with verified tenant and user info

    Raises:
        HTTPException: If access key is invalid or expired
    """
    settings = get_settings()

    # Get access key from header
    access_key = request.headers.get("X-Access-Key")
    if not access_key:
        logger.warning("missing_access_key", path=request.url.path)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTH_001",
                "category": "authentication",
                "message": "アクセスキーが必要です",
                "recoverable": False,
            },
        )

    try:
        # Decode and verify JWT
        payload = jwt.decode(
            access_key,
            settings.access_key_secret,
            algorithms=[settings.access_key_algorithm],
        )
        access_payload = AccessKeyPayload(**payload)

        # Check expiration
        now = datetime.now(timezone.utc).timestamp()
        if access_payload.exp < now:
            logger.warning(
                "access_key_expired",
                tenant_id=access_payload.tenant_id,
                expired_at=access_payload.exp,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "code": "AUTH_002",
                    "category": "authentication",
                    "message": "アクセスキーの期限が切れています",
                    "recoverable": False,
                    "user_action": "新しいアクセスキーを取得してください",
                },
            )

        # Build request context
        request_id = request.headers.get("X-Request-ID", f"req_{id(request)}")
        client_ip = request.client.host if request.client else None
        user_agent = request.headers.get("User-Agent")

        context = RequestContext(
            tenant_id=access_payload.tenant_id,
            user_id=access_payload.user_id,
            permissions=access_payload.permissions,
            request_id=request_id,
            client_ip=client_ip,
            user_agent=user_agent,
        )

        logger.debug(
            "access_key_verified",
            tenant_id=context.tenant_id,
            user_id=context.user_id,
            request_id=context.request_id,
        )

        return context

    except JWTError as e:
        logger.warning("invalid_access_key", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "code": "AUTH_001",
                "category": "authentication",
                "message": "アクセスキーが無効です",
                "detail": str(e),
                "recoverable": False,
                "user_action": "新しいアクセスキーを取得してください",
            },
        )


def check_permission(required_permission: str):
    """
    Create a dependency that checks for a specific permission.

    Args:
        required_permission: The permission required for the endpoint

    Returns:
        Dependency function that validates the permission
    """

    async def permission_checker(
        context: Annotated[RequestContext, Depends(verify_access_key)]
    ) -> RequestContext:
        if required_permission not in context.permissions:
            logger.warning(
                "permission_denied",
                tenant_id=context.tenant_id,
                user_id=context.user_id,
                required=required_permission,
                available=context.permissions,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "code": "AUTH_003",
                    "category": "authorization",
                    "message": "この操作を行う権限がありません",
                    "recoverable": False,
                },
            )
        return context

    return permission_checker


class AccessKeyMiddleware:
    """
    Middleware for access key verification.

    This can be used as an alternative to the dependency injection approach
    for routes that need custom handling.
    """

    def __init__(self, excluded_paths: list[str] | None = None):
        """
        Initialize middleware.

        Args:
            excluded_paths: Paths to exclude from verification
        """
        self.excluded_paths = excluded_paths or ["/health", "/metrics", "/docs", "/openapi.json"]

    async def __call__(self, request: Request, call_next):
        """Process the request."""
        # Skip verification for excluded paths
        if any(request.url.path.startswith(path) for path in self.excluded_paths):
            return await call_next(request)

        # Verify access key
        try:
            context = await verify_access_key(request)
            request.state.context = context
        except HTTPException:
            raise

        return await call_next(request)
