"""Manual KV-prefix-cache flush, requested via SIGUSR1 to the runner.

The handler only sets a flag; the actual clear happens between generator
steps (batch_generate.step), where no parked cache entry can be in use.
"""

from __future__ import annotations

import signal

from loguru import logger

_requested = False


def _on_signal(signum, frame):  # noqa: ARG001
    global _requested
    _requested = True


def install() -> None:
    try:
        signal.signal(signal.SIGUSR1, _on_signal)
        logger.info("prefix-flush: SIGUSR1 handler installed")
    except (ValueError, OSError) as exc:  # not main thread / unsupported
        logger.warning(f"prefix-flush: handler not installed: {exc!r}")


def take_request() -> bool:
    """Return True once per received signal."""
    global _requested
    if _requested:
        _requested = False
        return True
    return False
