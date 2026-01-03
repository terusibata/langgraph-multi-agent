"""Database connection pool management."""

import asyncpg
from src.config.settings import get_settings

_pool: asyncpg.Pool | None = None


async def init_db_pool() -> asyncpg.Pool:
    """Initialize database connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        # Extract connection parameters from database_url
        # Format: postgresql+asyncpg://user:password@host:port/database
        db_url = settings.database_url.replace("postgresql+asyncpg://", "postgresql://")
        _pool = await asyncpg.create_pool(db_url, min_size=5, max_size=20)
    return _pool


async def get_db_pool() -> asyncpg.Pool:
    """Get database connection pool."""
    global _pool
    if _pool is None:
        _pool = await init_db_pool()
    return _pool


async def close_db_pool() -> None:
    """Close database connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
