"""Service token extraction and validation middleware."""

import base64
import json
from datetime import datetime, timezone

from fastapi import HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError
import structlog

logger = structlog.get_logger()


class ServiceToken(BaseModel):
    """Individual service token."""

    token: str = Field(..., description="Token value")
    instance: str | None = Field(
        default=None,
        description="Service instance URL",
    )
    expires_at: str | None = Field(
        default=None,
        description="Expiration time (ISO 8601)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )

    def is_expired(self) -> bool:
        """Check if token is expired."""
        if not self.expires_at:
            return False
        try:
            exp_time = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return exp_time < datetime.now(timezone.utc)
        except ValueError:
            return False


class ServiceTokens(BaseModel):
    """Collection of service tokens."""

    servicenow: ServiceToken | None = Field(
        default=None,
        description="ServiceNow OAuth token",
    )
    vector_db: ServiceToken | None = Field(
        default=None,
        description="Vector DB API key",
    )

    def get(self, service_name: str) -> ServiceToken | None:
        """Get token for a specific service."""
        return getattr(self, service_name.replace("-", "_"), None)

    def has_token(self, service_name: str) -> bool:
        """Check if token exists for a service."""
        token = self.get(service_name)
        return token is not None

    def validate_token(self, service_name: str) -> tuple[bool, str | None]:
        """
        Validate a service token.

        Returns:
            Tuple of (is_valid, error_message)
        """
        token = self.get(service_name)
        if not token:
            return False, f"{service_name}のトークンがありません"
        if token.is_expired():
            return False, f"{service_name}のトークンが期限切れです"
        return True, None


async def extract_service_tokens(request: Request) -> ServiceTokens:
    """
    Extract and parse service tokens from request header.

    The X-Service-Tokens header should contain a Base64-encoded JSON object
    with service tokens.

    Args:
        request: FastAPI request object

    Returns:
        Parsed ServiceTokens object

    Raises:
        HTTPException: If tokens are malformed
    """
    header_value = request.headers.get("X-Service-Tokens")

    if not header_value:
        logger.debug("no_service_tokens_header")
        return ServiceTokens()

    try:
        # Decode Base64
        decoded = base64.b64decode(header_value)
        tokens_dict = json.loads(decoded)

        # Parse into model
        tokens = ServiceTokens(**tokens_dict)

        # Log available tokens (without exposing values)
        available = [name for name in ["servicenow", "vector_db"] if tokens.has_token(name)]
        logger.debug("service_tokens_extracted", available_tokens=available)

        return tokens

    except (ValueError, json.JSONDecodeError) as e:
        logger.warning("invalid_service_tokens_format", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TOKEN_001",
                "category": "service_token",
                "message": "サービストークンの形式が不正です",
                "detail": "Base64エンコードされたJSONを指定してください",
                "recoverable": False,
            },
        )
    except ValidationError as e:
        logger.warning("invalid_service_tokens_structure", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": "TOKEN_001",
                "category": "service_token",
                "message": "サービストークンの構造が不正です",
                "detail": str(e),
                "recoverable": False,
            },
        )


def require_service_token(service_name: str):
    """
    Create a dependency that requires a specific service token.

    Args:
        service_name: Name of the required service

    Returns:
        Dependency function that validates the token
    """

    async def token_checker(request: Request) -> ServiceTokens:
        tokens = await extract_service_tokens(request)
        is_valid, error_message = tokens.validate_token(service_name)

        if not is_valid:
            token = tokens.get(service_name)
            if token is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail={
                        "code": "TOKEN_003",
                        "category": "service_token",
                        "message": f"{service_name}のトークンがありません",
                        "service": service_name,
                        "recoverable": False,
                        "user_action": f"{service_name}との連携を設定してください",
                    },
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail={
                        "code": "TOKEN_002",
                        "category": "service_token",
                        "message": f"{service_name}のトークンが期限切れです",
                        "service": service_name,
                        "recoverable": False,
                        "user_action": f"{service_name}に再ログインしてください",
                    },
                )

        return tokens

    return token_checker


class ToolContext(BaseModel):
    """Context passed to tools for service token access."""

    service_tokens: ServiceTokens = Field(..., description="Available service tokens")
    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier")
    request_id: str = Field(..., description="Request identifier")

    def get_token(self, service_name: str) -> str | None:
        """Get token value for a service."""
        token = self.service_tokens.get(service_name)
        return token.token if token else None

    def get_instance(self, service_name: str) -> str | None:
        """Get instance URL for a service."""
        token = self.service_tokens.get(service_name)
        return token.instance if token else None

    def require_token(self, service_name: str) -> str:
        """
        Get token value or raise an error.

        Raises:
            ValueError: If token is not available or expired
        """
        token = self.service_tokens.get(service_name)
        if not token:
            raise ValueError(f"Token for {service_name} is not available")
        if token.is_expired():
            raise ValueError(f"Token for {service_name} has expired")
        return token.token
