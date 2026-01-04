"""add thread title

Revision ID: add_thread_title
Revises: 
Create Date: 2026-01-04

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'add_thread_title'
down_revision: Union[str, None] = '001'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add title column to threads table
    op.add_column('threads', sa.Column('title', sa.String(length=255), nullable=True))


def downgrade() -> None:
    # Remove title column from threads table
    op.drop_column('threads', 'title')
