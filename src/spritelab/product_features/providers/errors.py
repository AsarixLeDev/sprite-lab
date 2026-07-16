"""Normalized provider failures; messages never include credentials or response bodies."""

from __future__ import annotations


class ProviderError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    def __init__(self, message: str = "The provider request timed out.") -> None:
        super().__init__("provider_timeout", message, retryable=True)


class ProviderDeadlineExceededError(ProviderError):
    def __init__(self, message: str = "The provider run exceeded its elapsed-time deadline.") -> None:
        super().__init__("provider_deadline_exceeded", message, retryable=False)


class ProviderCancelledError(ProviderError):
    def __init__(self, message: str = "The provider request was cancelled.") -> None:
        super().__init__("provider_cancelled", message, retryable=False)


class ProviderInvalidOutputError(ProviderError):
    def __init__(self, message: str = "The provider returned output that does not match the required schema.") -> None:
        super().__init__("provider_invalid_output", message, retryable=False)


class ProviderPolicyError(ProviderError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, retryable=False)
