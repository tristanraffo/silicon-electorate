"""Model backends (SPEC §9). Import concrete backends from their own modules so that
`mock` never pulls in torch."""

from scv.backends.base import Backend, BackendError, ChatPrompt

__all__ = ["Backend", "BackendError", "ChatPrompt"]
