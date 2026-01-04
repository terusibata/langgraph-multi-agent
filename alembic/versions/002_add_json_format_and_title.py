"""Add response_format, response_schema, and title fields.

Revision ID: 002
Revises: 001
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add new fields for JSON format support and thread titles."""
    # Add response_format and response_schema to agents table
    op.add_column("agents", sa.Column("response_format", sa.String(50), nullable=True))
    op.add_column("agents", sa.Column("response_schema", sa.JSON(), nullable=True))

    # Add title to threads table
    op.add_column("threads", sa.Column("title", sa.String(100), nullable=True))


def downgrade() -> None:
    """Remove the added fields."""
    # Remove from agents table
    op.drop_column("agents", "response_schema")
    op.drop_column("agents", "response_format")

    # Remove from threads table
    op.drop_column("threads", "title")
