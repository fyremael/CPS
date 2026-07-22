from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


@dataclass
class ConsoleReporter:
    """Small dependency-free progress reporter suited to terminals and notebooks.

    The reporter deliberately emits complete lines and flushes after every event so
    Colab users can see where a long-running probe currently is. It is also useful
    for immutable execution logs produced by ``colab-cli``.
    """

    enabled: bool = True
    prefix: str = "CPS"
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    started_at: float = field(default_factory=time.perf_counter)
    _section: int = 0

    def _emit(self, marker: str, message: str) -> None:
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.started_at
        print(f"[{self.prefix} {elapsed:8.2f}s] {marker} {message}", file=self.stream, flush=True)

    def title(self, message: str) -> None:
        if not self.enabled:
            return
        rule = "=" * min(max(len(message) + 8, 36), 88)
        print(f"\n{rule}\n{message}\n{rule}", file=self.stream, flush=True)

    def section(self, message: str) -> None:
        self._section += 1
        self._emit(f"[{self._section:02d}]", message)

    def info(self, message: str) -> None:
        self._emit("   ", message)

    def metric(self, name: str, value: object, unit: str = "") -> None:
        suffix = f" {unit}" if unit else ""
        self._emit("  ·", f"{name}: {value}{suffix}")

    def progress(self, current: int, total: int, label: str, detail: str = "") -> None:
        if total < 1:
            total = 1
        width = 20
        fraction = min(max(current / total, 0.0), 1.0)
        filled = int(round(width * fraction))
        bar = "█" * filled + "·" * (width - filled)
        extra = f" — {detail}" if detail else ""
        self._emit("  ↳", f"{label} [{bar}] {current}/{total}{extra}")

    def warning(self, message: str) -> None:
        self._emit("WARN", message)

    def success(self, message: str) -> None:
        self._emit("DONE", message)


class NullReporter(ConsoleReporter):
    def __init__(self) -> None:
        super().__init__(enabled=False)
