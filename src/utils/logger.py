"""
logger.py
---------
Centralised Loguru logger factory.
All modules import `get_logger(__name__)` — never configure loguru themselves.

Features:
  - Structured output with timing context (FR-20)
  - File sink in prod, stderr only in dev/test
  - Intercepts stdlib `logging` so third-party libs (transformers, faiss) are captured
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from loguru import logger as _loguru_logger

# ── Stdlib → Loguru bridge ────────────────────────────────────────────────────

class _InterceptHandler(logging.Handler):
    """Route stdlib logging records through Loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = _loguru_logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = sys._getframe(6), 6
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back  # type: ignore[assignment]
            depth += 1

        _loguru_logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def _configure_logger(log_level: str = "INFO", app_env: str = "dev") -> None:
    """
    Configure Loguru sinks.
    Called once at application startup — idempotent due to remove().
    """
    _loguru_logger.remove()  # Remove default sink first

    # ── Console sink (always active) ──────────────────────────────────────────
    _loguru_logger.add(
        sys.stderr,
        level=log_level,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=(app_env == "dev"),
    )

    # ── File sink (prod only) ─────────────────────────────────────────────────
    if app_env == "prod":
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        _loguru_logger.add(
            log_dir / "pobot_{time:YYYY-MM-DD}.log",
            level=log_level,
            rotation="00:00",        # rotate at midnight
            retention="7 days",
            compression="zip",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            serialize=False,
            backtrace=True,
            diagnose=False,          # no sensitive data in prod tracebacks
        )

    # ── Intercept stdlib logging ──────────────────────────────────────────────
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for noisy_lib in ["urllib3", "httpx", "transformers", "faiss"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)


def get_logger(name: str):
    """
    Return a Loguru logger bound to *name*.

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Embeddings generated in {:.2f}s", elapsed)
    """
    return _loguru_logger.bind(module=name)


def setup_logging(log_level: str = "INFO", app_env: str = "dev") -> None:
    """
    Call this ONCE from main.py before importing any other module.

    Example:
        from utils.logger import setup_logging
        setup_logging(log_level="DEBUG", app_env="dev")
    """
    _configure_logger(log_level=log_level, app_env=app_env)