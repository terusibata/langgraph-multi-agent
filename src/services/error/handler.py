"""Error handling service."""

from typing import Any

import structlog

from src.services.metrics import get_metrics_collector

logger = structlog.get_logger()


# Error code definitions
ERROR_CODES = {
    # Authentication errors
    "AUTH_001": {
        "category": "authentication",
        "message": "アクセスキーが無効です",
        "recoverable": False,
        "user_action": "新しいアクセスキーを取得してください",
    },
    "AUTH_002": {
        "category": "authentication",
        "message": "アクセスキーの期限が切れています",
        "recoverable": False,
        "user_action": "新しいアクセスキーを取得してください",
    },
    "AUTH_003": {
        "category": "authorization",
        "message": "この操作を行う権限がありません",
        "recoverable": False,
    },
    # Service token errors
    "TOKEN_001": {
        "category": "service_token",
        "message": "サービストークンが無効です",
        "recoverable": False,
        "user_action": "サービスに再ログインしてください",
    },
    "TOKEN_002": {
        "category": "service_token",
        "message": "サービストークンの期限が切れています",
        "recoverable": False,
        "user_action": "サービスに再ログインしてください",
    },
    "TOKEN_003": {
        "category": "service_token",
        "message": "必要なサービストークンがありません",
        "recoverable": False,
        "user_action": "サービス連携を設定してください",
    },
    # Thread errors
    "THREAD_001": {
        "category": "thread",
        "message": "スレッドの上限に達しました",
        "recoverable": False,
        "user_action": "新しいスレッドを作成してください",
    },
    "THREAD_002": {
        "category": "thread",
        "message": "スレッドが見つかりません",
        "recoverable": False,
    },
    # AWS/Bedrock errors
    "AWS_001": {
        "category": "aws",
        "message": "Bedrockタイムアウト",
        "recoverable": True,
        "user_action": "しばらく待ってから再試行してください",
    },
    "AWS_002": {
        "category": "aws",
        "message": "Bedrockサービス一時障害",
        "recoverable": True,
        "user_action": "しばらく待ってから再試行してください",
    },
    "AWS_003": {
        "category": "aws",
        "message": "AWS認証エラー",
        "recoverable": False,
    },
    # External service errors
    "EXT_001": {
        "category": "external_service",
        "message": "外部サービスAPIエラー",
        "recoverable": True,
    },
    "EXT_002": {
        "category": "external_service",
        "message": "外部サービスタイムアウト",
        "recoverable": True,
        "user_action": "しばらく待ってから再試行してください",
    },
    # Model errors
    "MODEL_001": {
        "category": "model",
        "message": "指定されたモデルは利用できません",
        "recoverable": False,
    },
    "MODEL_002": {
        "category": "model",
        "message": "モデル応答エラー",
        "recoverable": True,
    },
    # Internal errors
    "INTERNAL_001": {
        "category": "internal",
        "message": "チェックポイント保存に失敗しました",
        "recoverable": True,
    },
    "INTERNAL_002": {
        "category": "internal",
        "message": "状態復元に失敗しました",
        "recoverable": False,
    },
    "INTERNAL_003": {
        "category": "internal",
        "message": "予期しないエラーが発生しました",
        "recoverable": False,
    },
    # Input errors
    "INPUT_001": {
        "category": "input",
        "message": "リクエスト形式が不正です",
        "recoverable": False,
    },
    "INPUT_002": {
        "category": "input",
        "message": "ファイル形式が不正です",
        "recoverable": False,
    },
}


class AgentError(Exception):
    """Custom exception for agent errors."""

    def __init__(
        self,
        code: str,
        detail: str | None = None,
        service: str | None = None,
    ):
        """
        Initialize the error.

        Args:
            code: Error code (e.g., "AUTH_001")
            detail: Additional error details
            service: Service that caused the error
        """
        self.code = code
        self.detail = detail
        self.service = service

        error_info = ERROR_CODES.get(code, ERROR_CODES["INTERNAL_003"])
        self.category = error_info["category"]
        self.message = error_info["message"]
        self.recoverable = error_info.get("recoverable", False)
        self.user_action = error_info.get("user_action")

        super().__init__(self.message)

    def to_dict(self) -> dict:
        """Convert to dictionary for response."""
        result = {
            "code": self.code,
            "category": self.category,
            "message": self.message,
            "recoverable": self.recoverable,
        }
        if self.detail:
            result["detail"] = self.detail
        if self.service:
            result["service"] = self.service
        if self.user_action:
            result["user_action"] = self.user_action
        return result


class ErrorHandler:
    """
    Centralized error handling service.
    """

    def __init__(self):
        """Initialize the handler."""
        self.metrics = get_metrics_collector()

    def handle_exception(
        self,
        exception: Exception,
        context: dict | None = None,
    ) -> AgentError:
        """
        Handle an exception and convert to AgentError.

        Args:
            exception: The exception to handle
            context: Additional context information

        Returns:
            AgentError instance
        """
        context = context or {}

        # If already an AgentError, just record and return
        if isinstance(exception, AgentError):
            self._record_error(exception)
            return exception

        # Map known exception types to error codes
        error = self._map_exception(exception, context)
        self._record_error(error)
        return error

    def _map_exception(
        self,
        exception: Exception,
        context: dict,
    ) -> AgentError:
        """Map an exception to an AgentError."""
        exception_type = type(exception).__name__
        message = str(exception)

        # Timeout errors
        if "timeout" in message.lower() or "TimeoutError" in exception_type:
            if context.get("service") == "bedrock":
                return AgentError("AWS_001", detail=message)
            else:
                return AgentError("EXT_002", detail=message, service=context.get("service"))

        # Authentication errors
        if "401" in message or "unauthorized" in message.lower():
            if context.get("service"):
                return AgentError("TOKEN_001", detail=message, service=context.get("service"))
            return AgentError("AUTH_001", detail=message)

        # Permission errors
        if "403" in message or "forbidden" in message.lower():
            return AgentError("AUTH_003", detail=message)

        # Connection errors
        if "ConnectionError" in exception_type or "connection" in message.lower():
            return AgentError("EXT_001", detail=message, service=context.get("service"))

        # Default to internal error
        return AgentError("INTERNAL_003", detail=message)

    def _record_error(self, error: AgentError) -> None:
        """Record error in metrics."""
        self.metrics.record_error(error.code, error.category)

        logger.error(
            "agent_error",
            code=error.code,
            category=error.category,
            message=error.message,
            detail=error.detail,
            service=error.service,
        )

    def create_error_response(
        self,
        error: AgentError,
        session_id: str | None = None,
        thread_id: str | None = None,
        partial_results: dict | None = None,
        partial_metrics: dict | None = None,
        thread_state: dict | None = None,
    ) -> dict:
        """
        Create error response data for SSE.

        Args:
            error: The AgentError
            session_id: Session identifier
            thread_id: Thread identifier
            partial_results: Any partial results
            partial_metrics: Any partial metrics
            thread_state: Current thread state

        Returns:
            Error response dictionary
        """
        return {
            "session_id": session_id,
            "thread_id": thread_id,
            "error": error.to_dict(),
            "partial_results": partial_results,
            "partial_metrics": partial_metrics,
            "thread_state": thread_state,
        }


# Global error handler
_error_handler: ErrorHandler | None = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler
