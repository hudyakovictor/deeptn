"""
DeepTN Progress Display
========================
Dynamic progress bars and status display using Rich.

Usage:
    from deeputin.shared.progress import ProgressTracker
    
    tracker = ProgressTracker(total=100, description="Processing photos")
    for photo in photos:
        tracker.update(photo_id=photo.name, status="extracting metrics")
        # ... do work ...
        tracker.advance()
    tracker.finish()
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        SpinnerColumn,
        TaskProgressColumn,
    )
    from rich.table import Table
    from rich.live import Live
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.text import Text
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ─────────────────────────────────────────────
# Simple fallback progress (no Rich)
# ─────────────────────────────────────────────
class SimpleProgress:
    """Fallback progress display without Rich."""

    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = time.time()
        self._last_print = 0

    def update(self, **kwargs):
        pass

    def advance(self, n: int = 1):
        self.current += n
        now = time.time()
        if now - self._last_print > 1.0 or self.current == self.total:
            self._last_print = now
            elapsed = now - self.start_time
            rate = self.current / max(elapsed, 0.001)
            eta = (self.total - self.current) / max(rate, 0.001)
            pct = self.current / max(self.total, 1) * 100
            bar_len = 30
            filled = int(bar_len * self.current / max(self.total, 1))
            bar = "█" * filled + "░" * (bar_len - filled)
            print(
                f"\r{self.description}: [{bar}] {pct:5.1f}% "
                f"({self.current}/{self.total}) "
                f"ETA: {eta:.0f}s",
                end="",
                flush=True,
            )
            if self.current == self.total:
                print()

    def finish(self):
        elapsed = time.time() - self.start_time
        print(f"{self.description}: Done in {elapsed:.1f}s")


# ─────────────────────────────────────────────
# Rich-based progress tracker
# ─────────────────────────────────────────────
class ProgressTracker:
    """
    Dynamic progress tracker with Rich display.
    
    Shows:
    - Progress bar with percentage
    - Current photo being processed
    - Current operation status
    - Elapsed time and ETA
    - Error/warning counts
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        show_stats: bool = True,
    ):
        self.total = total
        self.description = description
        self.show_stats = show_stats
        self.current = 0
        self.errors = 0
        self.warnings = 0
        self.current_photo = ""
        self.current_status = ""
        self.start_time = time.time()
        self._progress = None
        self._task_id = None
        self._live = None

        if HAS_RICH:
            self._console = Console(stderr=True)
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(
                    bar_width=40,
                    complete_style="green",
                    finished_style="bold green",
                ),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self._console,
                transient=False,
            )
            self._task_id = self._progress.add_task(description, total=total)

    def __enter__(self):
        if self._progress:
            self._progress.start()
        return self

    def __exit__(self, *args):
        self.finish()

    def update(
        self,
        photo_id: str = "",
        status: str = "",
        error: bool = False,
        warning: bool = False,
    ):
        """Update current photo/status display."""
        self.current_photo = photo_id
        self.current_status = status
        if error:
            self.errors += 1
        if warning:
            self.warnings += 1

        if self._progress and self._task_id is not None:
            desc = self.description
            if photo_id:
                desc += f" [dim]({photo_id})[/dim]"
            if status:
                color = "red" if error else ("yellow" if warning else "dim")
                desc += f" [{color}]{status}[/{color}]"
            self._progress.update(self._task_id, description=desc)

    def advance(self, n: int = 1):
        """Advance progress by n steps."""
        self.current += n
        if self._progress and self._task_id is not None:
            self._progress.update(self._task_id, advance=n)

    def finish(self):
        """Finish progress display and show summary."""
        if self._progress:
            self._progress.stop()

        elapsed = time.time() - self.start_time
        mins = int(elapsed // 60)
        secs = int(elapsed % 60)

        if HAS_RICH:
            console = Console(stderr=True)
            # Build summary
            status_parts = []
            if self.errors > 0:
                status_parts.append(f"[bold red]✗ {self.errors} errors[/bold red]")
            if self.warnings > 0:
                status_parts.append(f"[yellow]⚠ {self.warnings} warnings[/yellow]")
            if self.errors == 0 and self.warnings == 0:
                status_parts.append("[bold green]✓ All OK[/bold green]")

            status_text = " | ".join(status_parts) if status_parts else "✓"
            console.print(
                f"[bold]{self.description}[/bold]: "
                f"{self.current}/{self.total} in {mins}m{secs:02d}s | "
                f"{status_parts[0] if status_parts else '✓'}"
            )
        else:
            status = "OK" if self.errors == 0 else f"{self.errors} errors"
            print(f"{self.description}: {self.current}/{self.total} in {mins}m{secs:02d}s | {status}")

    @property
    def elapsed(self) -> float:
        return time.time() - self.start_time


# ─────────────────────────────────────────────
# Multi-stage pipeline progress
# ─────────────────────────────────────────────
class PipelineProgress:
    """
    Track progress across multiple pipeline stages.
    
    Usage:
        pipeline = PipelineProgress([
            ("S1: Extract", 100),
            ("S2: Metrics", 100),
            ("S3: Identity", 100),
        ])
        pipeline.start_stage("S1: Extract")
        for photo in photos:
            pipeline.update(photo_id=photo.name)
            pipeline.advance()
        pipeline.finish_stage()
    """

    def __init__(self, stages: list[tuple[str, int]]):
        self.stages = stages
        self.current_stage_idx = 0
        self.current_tracker: Optional[ProgressTracker] = None
        self._completed_stages: list[dict] = []

    def start_stage(self, name: str = ""):
        if name:
            for i, (n, _) in enumerate(self.stages):
                if n == name:
                    self.current_stage_idx = i
                    break

        name, total = self.stages[self.current_stage_idx]
        prefix = f"[{self.current_stage_idx + 1}/{len(self.stages)}] "
        self.current_tracker = ProgressTracker(total=total, description=prefix + name)
        if self.current_tracker._progress:
            self.current_tracker._progress.start()

    def update(self, **kwargs):
        if self.current_tracker:
            self.current_tracker.update(**kwargs)

    def advance(self, n: int = 1):
        if self.current_tracker:
            self.current_tracker.advance(n)

    def finish_stage(self):
        if self.current_tracker:
            self.current_tracker.finish()
            self._completed_stages.append({
                "name": self.stages[self.current_stage_idx][0],
                "elapsed": self.current_tracker.elapsed,
                "errors": self.current_tracker.errors,
                "warnings": self.current_tracker.warnings,
            })
            self.current_tracker = None

    def finish_all(self):
        if HAS_RICH:
            console = Console(stderr=True)
            table = Table(title="Pipeline Summary", show_header=True)
            table.add_column("Stage", style="bold")
            table.add_column("Time", justify="right")
            table.add_column("Status", justify="center")

            total_time = 0
            total_errors = 0
            for stage in self._completed_stages:
                elapsed = stage["elapsed"]
                total_time += elapsed
                total_errors += stage["errors"]
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                time_str = f"{mins}m{secs:02d}s"

                if stage["errors"] > 0:
                    status = f"[red]✗ {stage['errors']} err[/red]"
                elif stage["warnings"] > 0:
                    status = f"[yellow]⚠ {stage['warnings']} warn[/yellow]"
                else:
                    status = "[green]✓[/green]"

                table.add_row(stage["name"], time_str, status)

            total_mins = int(total_time // 60)
            total_secs = int(total_time % 60)
            table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{total_mins}m{total_secs:02d}s[/bold]",
                f"[red]✗ {total_errors}[/red]" if total_errors > 0 else "[green]✓[/green]",
            )
            console.print(table)


# ─────────────────────────────────────────────
# Convenience function
# ─────────────────────────────────────────────
def create_progress(total: int, description: str = "Processing") -> ProgressTracker:
    """Create a progress tracker (Rich if available, simple fallback otherwise)."""
    if HAS_RICH:
        return ProgressTracker(total=total, description=description)
    else:
        return SimpleProgress(total=total, description=description)
