"""Request schemas for API endpoints."""

from typing import Literal

from pydantic import BaseModel, Field


class CompanyContext(BaseModel):
    """Company-specific context for multi-tenant operation."""

    company_id: str = Field(..., description="Company identifier")
    company_name: str | None = Field(
        default=None,
        description="Company name",
    )
    vision: str | None = Field(
        default=None,
        description="Company vision or mission statement",
    )
    terminology: dict[str, str] = Field(
        default_factory=dict,
        description="Company-specific terminology (e.g., {'システム': 'プラットフォーム'})",
    )
    reference_info: dict = Field(
        default_factory=dict,
        description="Additional reference information",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )


class FileInput(BaseModel):
    """File input attached to a request."""

    filename: str = Field(..., description="Original filename")
    content_type: str = Field(..., description="MIME type of the file")
    content: str = Field(..., description="Base64 encoded file content")
    size_bytes: int = Field(..., description="File size in bytes")


class AgentStreamRequest(BaseModel):
    """Request body for agent streaming endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=32000,
        description="User message to process",
    )
    thread_id: str | None = Field(
        default=None,
        description="Thread ID for conversation continuation",
    )
    model_id: str | None = Field(
        default=None,
        description="Model ID for MainAgent (uses default if not specified)",
    )
    files: list[FileInput] = Field(
        default_factory=list,
        max_length=10,
        description="Attached files (max 10)",
    )
    company_context: CompanyContext | None = Field(
        default=None,
        description="Company-specific context (optional)",
    )
    fast_response: bool = Field(
        default=False,
        description="Enable fast response mode (MainAgent only, no sub-agents or tools)",
    )
    direct_tool_mode: bool = Field(
        default=False,
        description="Enable direct tool mode (MainAgent uses tools directly, no sub-agents)",
    )
    response_format: Literal["text", "json"] | None = Field(
        default=None,
        description="Response format: 'text' for plain text, 'json' for structured JSON",
    )
    response_schema: dict | None = Field(
        default=None,
        description="JSON schema for structured response (only used when response_format='json')",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "message": "プリンターに接続できない不具合が発生していますが、どうすればいいですか？",
                    "thread_id": None,
                    "model_id": None,
                    "files": [],
                    "company_context": None,
                },
                {
                    "message": "先ほどの問題の続きですが、再起動しても解決しませんでした。",
                    "thread_id": "thread_xyz789",
                    "model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0",
                    "files": [],
                    "company_context": {
                        "company_id": "acme_corp",
                        "company_name": "Acme Corporation",
                        "vision": "革新的なソリューションで顧客の課題を解決する",
                        "terminology": {
                            "システム": "プラットフォーム",
                            "ユーザー": "メンバー"
                        },
                        "reference_info": {
                            "support_hours": "平日9:00-18:00"
                        }
                    },
                },
            ]
        }
    }


class AgentInvokeRequest(AgentStreamRequest):
    """Request body for agent synchronous invoke endpoint."""

    timeout_seconds: int = Field(
        default=300,
        ge=10,
        le=600,
        description="Request timeout in seconds",
    )


class ThreadDeleteRequest(BaseModel):
    """Request body for thread deletion."""

    confirm: bool = Field(
        default=False,
        description="Confirmation flag for deletion",
    )


class AccessKeyPayload(BaseModel):
    """Payload structure for access key JWT."""

    tenant_id: str = Field(..., description="Tenant identifier")
    user_id: str = Field(..., description="User identifier (for logging)")
    permissions: list[str] = Field(
        default_factory=lambda: ["agent:invoke"],
        description="Allowed operations",
    )
    iat: int = Field(..., description="Issued at timestamp")
    exp: int = Field(..., description="Expiration timestamp")


class ServiceToken(BaseModel):
    """Structure for a service token."""

    token: str = Field(..., description="Token value")
    instance: str | None = Field(
        default=None,
        description="Service instance (e.g., ServiceNow instance URL)",
    )
    expires_at: str | None = Field(
        default=None,
        description="Token expiration time (ISO 8601)",
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Additional metadata",
    )


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
        return getattr(self, service_name, None)

    def has_token(self, service_name: str) -> bool:
        """Check if token exists for a service."""
        return self.get(service_name) is not None
