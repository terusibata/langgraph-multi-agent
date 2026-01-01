"""Thread management API routes."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
import structlog

from src.api.middleware.access_key import verify_access_key, RequestContext
from src.api.schemas.response import (
    ErrorResponse,
    ThreadResponse,
    ThreadStatusResponse,
    ThreadMetrics,
)
from src.services.thread import get_thread_manager
from src.config import get_settings

logger = structlog.get_logger()

router = APIRouter(prefix="/threads", tags=["Threads"])


@router.get(
    "/{thread_id}",
    response_model=ThreadResponse,
    summary="Get thread information",
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def get_thread(
    thread_id: str,
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Get detailed information about a conversation thread."""
    thread_manager = get_thread_manager()
    thread = await thread_manager.get_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "THREAD_002",
                "category": "thread",
                "message": "スレッドが見つかりません",
                "recoverable": False,
            },
        )

    # Check tenant access
    if thread.tenant_id != context.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "AUTH_003",
                "category": "authorization",
                "message": "このスレッドへのアクセス権限がありません",
                "recoverable": False,
            },
        )

    settings = get_settings()
    context_usage = (
        thread.context_tokens_used / settings.context_max_tokens
    ) * 100

    return ThreadResponse(
        thread_id=thread.thread_id,
        tenant_id=thread.tenant_id,
        status=thread.status,
        created_at=thread.created_at,
        updated_at=thread.updated_at,
        metrics=ThreadMetrics(
            total_tokens_used=thread.total_tokens_used,
            total_cost_usd=thread.total_cost_usd,
            message_count=thread.message_count,
            context_tokens_used=thread.context_tokens_used,
            context_usage_percent=context_usage,
        ),
        metadata=thread.metadata,
    )


@router.get(
    "/{thread_id}/status",
    response_model=ThreadStatusResponse,
    summary="Get thread status only",
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def get_thread_status(
    thread_id: str,
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Get minimal status information for a thread."""
    thread_manager = get_thread_manager()
    thread = await thread_manager.get_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "THREAD_002",
                "category": "thread",
                "message": "スレッドが見つかりません",
                "recoverable": False,
            },
        )

    # Check tenant access
    if thread.tenant_id != context.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "AUTH_003",
                "category": "authorization",
                "message": "このスレッドへのアクセス権限がありません",
                "recoverable": False,
            },
        )

    settings = get_settings()
    context_usage = (
        thread.context_tokens_used / settings.context_max_tokens
    ) * 100

    return ThreadStatusResponse(
        thread_id=thread.thread_id,
        status=thread.status,
        context_usage_percent=context_usage,
        can_send_message=(thread.status != "locked"),
    )


@router.delete(
    "/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a thread",
    responses={
        404: {"model": ErrorResponse, "description": "Thread not found"},
    },
)
async def delete_thread(
    thread_id: str,
    context: Annotated[RequestContext, Depends(verify_access_key)],
):
    """Delete a conversation thread and its history."""
    thread_manager = get_thread_manager()
    thread = await thread_manager.get_thread(thread_id)

    if not thread:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "THREAD_002",
                "category": "thread",
                "message": "スレッドが見つかりません",
                "recoverable": False,
            },
        )

    # Check tenant access
    if thread.tenant_id != context.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "code": "AUTH_003",
                "category": "authorization",
                "message": "このスレッドへのアクセス権限がありません",
                "recoverable": False,
            },
        )

    await thread_manager.delete_thread(thread_id)

    logger.info(
        "thread_deleted",
        thread_id=thread_id,
        tenant_id=context.tenant_id,
        user_id=context.user_id,
    )
