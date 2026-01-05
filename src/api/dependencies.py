"""FastAPI dependencies."""

from typing import Annotated

from fastapi import Depends

from src.api.middleware.access_key import verify_access_key, RequestContext

# Re-export for convenience
VerifiedContext = Annotated[RequestContext, Depends(verify_access_key)]
