"""Retry handling service."""

import asyncio
from typing import Callable, Any, TypeVar

import structlog

from src.config import get_settings

logger = structlog.get_logger()

T = TypeVar("T")


class RetryHandler:
    """
    Service for handling retries with exponential backoff.
    """

    def __init__(
        self,
        max_retries: int | None = None,
        backoff_seconds: float | None = None,
    ):
        """
        Initialize the handler.

        Args:
            max_retries: Maximum retry attempts
            backoff_seconds: Base backoff time
        """
        settings = get_settings()
        self.max_retries = max_retries or settings.default_max_retries
        self.backoff_seconds = backoff_seconds or settings.retry_backoff_seconds

    async def execute_with_retry(
        self,
        func: Callable[..., Any],
        *args,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        max_retries: int | None = None,
        backoff_multiplier: float = 2.0,
        **kwargs,
    ) -> T:
        """
        Execute a function with retry logic.

        Args:
            func: Async function to execute
            *args: Positional arguments for the function
            retry_on: Exception types to retry on
            max_retries: Override max retries
            backoff_multiplier: Multiplier for exponential backoff
            **kwargs: Keyword arguments for the function

        Returns:
            Function result

        Raises:
            Exception: If all retries exhausted
        """
        retries = max_retries if max_retries is not None else self.max_retries
        last_exception = None

        for attempt in range(retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)

            except retry_on as e:
                last_exception = e

                if attempt < retries:
                    wait_time = self.backoff_seconds * (backoff_multiplier ** attempt)
                    logger.warning(
                        "retry_attempt",
                        attempt=attempt + 1,
                        max_retries=retries,
                        wait_time=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        "retry_exhausted",
                        attempts=retries + 1,
                        error=str(e),
                    )

        raise last_exception

    async def execute_with_fallback(
        self,
        func: Callable[..., T],
        fallback: Callable[..., T],
        *args,
        retry_on: tuple[type[Exception], ...] = (Exception,),
        **kwargs,
    ) -> T:
        """
        Execute a function with fallback on failure.

        Args:
            func: Primary function to execute
            fallback: Fallback function if primary fails
            *args: Arguments for functions
            retry_on: Exception types that trigger fallback
            **kwargs: Keyword arguments for functions

        Returns:
            Result from primary or fallback function
        """
        try:
            return await self.execute_with_retry(
                func,
                *args,
                retry_on=retry_on,
                **kwargs,
            )
        except retry_on as e:
            logger.warning(
                "executing_fallback",
                primary_error=str(e),
            )
            if asyncio.iscoroutinefunction(fallback):
                return await fallback(*args, **kwargs)
            else:
                return fallback(*args, **kwargs)
