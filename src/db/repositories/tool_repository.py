"""Repository for managing dynamic tool definitions in the database."""

import json
import uuid
from datetime import datetime, timezone
from typing import Any

import asyncpg

from src.agents.registry import ToolDefinition
from src.db.connection import get_db_pool


class ToolRepository:
    """Repository for managing tool definitions in PostgreSQL."""

    @staticmethod
    async def create(definition: ToolDefinition) -> ToolDefinition:
        """Create a new tool definition in the database."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO dynamic_tools (
                    id, name, description, category, parameters,
                    executor, required_service_token, timeout_seconds,
                    enabled, metadata, created_at, updated_at, created_by
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                """,
                uuid.UUID(definition.id),
                definition.name,
                definition.description,
                definition.category,
                json.dumps(definition.parameters),
                json.dumps(definition.executor),
                definition.required_service_token,
                definition.timeout_seconds,
                definition.enabled,
                json.dumps(definition.metadata),
                definition.created_at,
                definition.updated_at,
                definition.created_by,
            )
        return definition

    @staticmethod
    async def get_by_name(name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id, name, description, category, parameters,
                       executor, required_service_token, timeout_seconds,
                       enabled, metadata, created_at, updated_at, created_by
                FROM dynamic_tools
                WHERE name = $1
                """,
                name,
            )
            if row:
                return ToolRepository._row_to_definition(row)
            return None

    @staticmethod
    async def get_all(enabled_only: bool = False) -> list[ToolDefinition]:
        """Get all tool definitions."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            query = """
                SELECT id, name, description, category, parameters,
                       executor, required_service_token, timeout_seconds,
                       enabled, metadata, created_at, updated_at, created_by
                FROM dynamic_tools
            """
            if enabled_only:
                query += " WHERE enabled = TRUE"
            query += " ORDER BY name"

            rows = await conn.fetch(query)
            return [ToolRepository._row_to_definition(row) for row in rows]

    @staticmethod
    async def update(definition: ToolDefinition) -> ToolDefinition:
        """Update a tool definition."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            definition.updated_at = datetime.now(timezone.utc)
            await conn.execute(
                """
                UPDATE dynamic_tools
                SET description = $2, category = $3, parameters = $4,
                    executor = $5, required_service_token = $6,
                    timeout_seconds = $7, enabled = $8, metadata = $9,
                    updated_at = $10
                WHERE name = $1
                """,
                definition.name,
                definition.description,
                definition.category,
                json.dumps(definition.parameters),
                json.dumps(definition.executor),
                definition.required_service_token,
                definition.timeout_seconds,
                definition.enabled,
                json.dumps(definition.metadata),
                definition.updated_at,
            )
        return definition

    @staticmethod
    async def delete(name: str) -> bool:
        """Delete a tool definition."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM dynamic_tools WHERE name = $1",
                name,
            )
            # Check if any rows were deleted
            return result.split()[-1] != "0"

    @staticmethod
    async def exists(name: str) -> bool:
        """Check if a tool definition exists."""
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT 1 FROM dynamic_tools WHERE name = $1",
                name,
            )
            return row is not None

    @staticmethod
    def _row_to_definition(row: asyncpg.Record) -> ToolDefinition:
        """Convert a database row to a ToolDefinition object."""
        return ToolDefinition(
            id=str(row["id"]),
            name=row["name"],
            description=row["description"],
            category=row["category"],
            parameters=json.loads(row["parameters"]) if isinstance(row["parameters"], str) else row["parameters"],
            executor=json.loads(row["executor"]) if isinstance(row["executor"], str) else row["executor"],
            required_service_token=row["required_service_token"],
            timeout_seconds=row["timeout_seconds"],
            enabled=row["enabled"],
            metadata=json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            created_by=row["created_by"],
        )
