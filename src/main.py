"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import structlog

from src.api.routes import agent_router, threads_router, health_router, admin_router
from src.agents.registry import initialize_registries, register_all_tools, register_all_agents
from src.config import get_settings
from src.models import init_db, close_db
from src import __version__

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("application_starting", version=__version__)

    settings = get_settings()
    logger.info(
        "configuration_loaded",
        default_model=settings.default_model_id,
        sub_agent_model=settings.sub_agent_model_id,
        aws_region=settings.aws_region,
    )

    # Initialize database tables
    try:
        await init_db()
        logger.info("database_initialized")
    except Exception as e:
        logger.error("database_initialization_failed", error=str(e))
        raise

    # Initialize registries with database configuration
    try:
        await initialize_registries()
        register_all_tools()
        register_all_agents()
        logger.info("registries_initialized", storage="database")
    except Exception as e:
        logger.error("registry_initialization_failed", error=str(e))

    yield

    # Shutdown
    logger.info("application_shutting_down")
    await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="LangGraph Multi-Agent Backend",
        description="階層型マルチエージェントバックエンドAPI",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error=str(exc),
            exc_info=exc,
        )
        return JSONResponse(
            status_code=500,
            content={
                "code": "INTERNAL_003",
                "category": "internal",
                "message": "予期しないエラーが発生しました",
                "recoverable": False,
            },
        )

    # Include routers
    app.include_router(health_router)
    app.include_router(agent_router, prefix=settings.api_prefix)
    app.include_router(threads_router, prefix=settings.api_prefix)
    app.include_router(admin_router, prefix=settings.api_prefix)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level=settings.log_level.lower(),
    )
