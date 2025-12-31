"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends

from src.api.middleware.access_key import verify_access_key, RequestContext
from src.api.middleware.service_tokens import extract_service_tokens, ServiceTokens

# Re-export for convenience
VerifiedContext = Annotated[RequestContext, Depends(verify_access_key)]
