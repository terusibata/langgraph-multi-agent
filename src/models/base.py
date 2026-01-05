"""Base database models and session management."""

from datetime import datetime, timezone
from typing import AsyncGenerator

from sqlalchemy import DateTime, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from src.config import get_settings


class Base(DeclarativeBase):
    """Base class for all database models."""

    pass


class TimestampMixin:
    """Mixin for created_at and updated_at timestamps."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


# Async engine and session factory
_async_engine = None
_async_session_factory = None


def get_async_engine():
    """Get or create the async database engine."""
    global _async_engine
    if _async_engine is None:
        settings = get_settings()
        _async_engine = create_async_engine(
            settings.database_url,
            echo=False,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20,
        )
    return _async_engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the async session factory."""
    global _async_session_factory
    if _async_session_factory is None:
        engine = get_async_engine()
        _async_session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )
    return _async_session_factory


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Get an async database session."""
    session_factory = get_session_factory()
    async with session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database connection and verify it's working.

    Note: Table creation is managed by Alembic migrations.
    Run 'alembic upgrade head' to create/update database schema.
    """
    engine = get_async_engine()
    # Test database connection
    async with engine.begin() as conn:
        await conn.execute(text("SELECT 1"))

    # Note: We don't call Base.metadata.create_all() here
    # All schema changes are managed through Alembic migrations


async def close_db() -> None:
    """Close database connections."""
    global _async_engine, _async_session_factory
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
        _async_session_factory = None
