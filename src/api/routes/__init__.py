"""API routes module."""

from src.api.routes.agent import router as agent_router
from src.api.routes.threads import router as threads_router
from src.api.routes.health import router as health_router
from src.api.routes.admin import router as admin_router

__all__ = ["agent_router", "threads_router", "health_router", "admin_router"]
