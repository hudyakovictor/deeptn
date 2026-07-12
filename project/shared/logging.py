"""
DeepTN Logging System
=====================
Цветное логирование с file handler.

Уровни:
- DEBUG: серо-голубой — детальная отладка
- INFO: белый — нормальная работа
- SUCCESS: зелёный — успешные операции
- WARNING: жёлтый — потенциальные проблемы (мягкие ошибки)
- ERROR: красный — критические ошибки
- CRITICAL: красный жирный — фатальные ошибки

File logging: все уровни ≥ DEBUG пишутся в deeputin.log
"""

from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from rich.console import Console
    from rich.logging import RichHandler
    from rich.theme import Theme
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ─────────────────────────────────────────────
# Custom log levels
# ─────────────────────────────────────────────
SUCCESS = 25  # between INFO (20) and WARNING (30)
logging.addLevelName(SUCCESS, "SUCCESS")

ATTENTION = 22  # between INFO (20) and WARNING (30) — for "expected vs actual" mismatches
logging.addLevelName(ATTENTION, "ATTENTION")


# ─────────────────────────────────────────────
# Rich theme
# ─────────────────────────────────────────────
if HAS_RICH:
    DEEPUTIN_THEME = Theme({
        "logging.level.debug": "dim cyan",
        "logging.level.info": "white",
        "logging.level.success": "bold green",
        "logging.level.attention": "bold yellow",
        "logging.level.warning": "bold yellow",
        "logging.level.error": "bold red",
        "logging.level.critical": "bold white on red",
        "log.time": "dim",
        "log.path": "dim cyan",
    })


# ─────────────────────────────────────────────
# File handler setup
# ─────────────────────────────────────────────
_file_handler: Optional[logging.FileHandler] = None
_log_dir: Optional[Path] = None


def setup_file_logging(log_dir: str | Path | None = None) -> Path:
    """
    Set up file logging. All log messages ≥ DEBUG go to deeputin.log.
    Returns the log file path.
    """
    global _file_handler, _log_dir

    if log_dir is None:
        log_dir = Path(os.environ.get("DPTN_LOG_DIR", "."))
    _log_dir = Path(log_dir)
    _log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = _log_dir / f"deeputin_{timestamp}.log"

    _file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-10s | %(name)-30s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _file_handler.setFormatter(fmt)

    # Add to root logger
    logging.getLogger("deeputin").addHandler(_file_handler)

    return log_file


def get_log_dir() -> Optional[Path]:
    return _log_dir


# ─────────────────────────────────────────────
# Main logger factory
# ─────────────────────────────────────────────
_initialized = False


def _init_logging():
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("deeputin")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # Remove any existing handlers to avoid duplicates
    for h in list(root.handlers):
        root.removeHandler(h)

    if HAS_RICH:
        console = Console(theme=DEEPUTIN_THEME, stderr=True)
        handler = RichHandler(
            console=console,
            show_time=True,
            show_path=False,
            show_level=True,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            level=logging.INFO,
        )
    else:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(logging.INFO)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-10s | %(name)-20s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)

    root.addHandler(handler)


def setup_logger(name: str) -> "DeepTNLogger":
    """Create a named logger for a DeepTN module."""
    _init_logging()
    logger = logging.getLogger(f"deeputin.{name}")
    return DeepTNLogger(logger)


# ─────────────────────────────────────────────
# Enhanced logger class
# ─────────────────────────────────────────────
class DeepTNLogger:
    """
    Enhanced logger with color-coded output and soft validation.
    
    Usage:
        log = setup_logger("s2_metrics")
        log.info("Processing photo %s", photo_id)
        log.success("Completed 50/100 photos")
        log.attention("Expected quality > 0.5, got 0.32 — may affect results")
        log.warning("Missing metric: pore_density — using fallback")
        log.error("Failed to load 3DDFA model")
    """

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._error_count = 0
        self._warning_count = 0
        self._attention_count = 0
        self._start_time = time.time()

    # ── Standard log levels ──

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def success(self, msg: str, *args, **kwargs):
        self._logger.log(SUCCESS, msg, *args, **kwargs)

    def attention(self, msg: str, *args, **kwargs):
        """
        Soft warning: expected value didn't match actual.
        Not an error, but may distort results.
        Yellow in console, logged as ATTENTION.
        """
        self._attention_count += 1
        self._logger.log(ATTENTION, f"⚠ {msg}", *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._warning_count += 1
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._error_count += 1
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._error_count += 1
        self._logger.critical(msg, *args, **kwargs)

    def exception(self, msg: str, *args, **kwargs):
        self._error_count += 1
        self._logger.exception(msg, *args, **kwargs)

    # ── Soft validation helpers ──

    def expect_range(self, value: float, lo: float, hi: float,
                     name: str = "value", photo_id: str = ""):
        """
        Check if value is in expected range.
        Logs ATTENTION if outside range (yellow, non-fatal).
        Returns True if OK, False if out of range.
        """
        if not (lo <= value <= hi):
            prefix = f"[{photo_id}] " if photo_id else ""
            self.attention(
                f"{prefix}{name}={value:.4f} outside expected [{lo:.4f}, {hi:.4f}]"
            )
            return False
        return True

    def expect_positive(self, value: float, name: str = "value",
                        photo_id: str = ""):
        """Check if value is positive."""
        if value is None or (isinstance(value, float) and not __import__('math').isfinite(value)):
            prefix = f"[{photo_id}] " if photo_id else ""
            self.attention(f"{prefix}{name} is None/NaN/Inf (expected positive)")
            return False
        if value <= 0:
            prefix = f"[{photo_id}] " if photo_id else ""
            self.attention(f"{prefix}{name}={value:.4f} (expected > 0)")
            return False
        return True

    def expect_not_none(self, value, name: str = "value",
                        photo_id: str = "", fallback=None):
        """Check if value is not None. Logs warning if None."""
        if value is None:
            prefix = f"[{photo_id}] " if photo_id else ""
            if fallback is not None:
                self.warning(f"{prefix}{name} is None → using fallback={fallback}")
            else:
                self.warning(f"{prefix}{name} is None — no fallback available")
            return False
        return True

    def expect_metric_present(self, metrics: dict, key: str,
                             photo_id: str = ""):
        """Check if a metric key exists and is not NaN."""
        import math
        prefix = f"[{photo_id}] " if photo_id else ""
        if key not in metrics:
            self.attention(f"{prefix}Metric '{key}' missing from output")
            return False
        val = metrics[key]
        if isinstance(val, float) and not math.isfinite(val):
            self.attention(f"{prefix}Metric '{key}' = {val} (NaN/Inf)")
            return False
        return True

    # ── Summary ──

    def summary(self, total: int = 0, processed: int = 0):
        """Print session summary."""
        elapsed = time.time() - self._start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        self.info("─" * 60)
        if total > 0:
            self.info(f"Processed: {processed}/{total}")
        self.info(f"Time: {mins}m {secs}s")
        
        if self._error_count > 0:
            self.error(f"Errors: {self._error_count}")
        if self._warning_count > 0:
            self.warning(f"Warnings: {self._warning_count}")
        if self._attention_count > 0:
            self.attention(f"Attentions: {self._attention_count}")
        
        if self._error_count == 0 and self._warning_count == 0:
            self.success("Completed successfully ✓")
        elif self._error_count == 0:
            self.success(f"Completed with {self._warning_count} warnings")
        else:
            self.error(f"Completed with {self._error_count} errors, {self._warning_count} warnings")
        
        self.info("─" * 60)

    @property
    def error_count(self) -> int:
        return self._error_count

    @property
    def warning_count(self) -> int:
        return self._warning_count

    @property
    def attention_count(self) -> int:
        return self._attention_count
