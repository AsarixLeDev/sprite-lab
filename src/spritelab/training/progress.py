"""Lightweight live training progress for humans watching a run.

This module is intentionally dependency-free and side-effect friendly: it renders
a single self-updating status line to a terminal (carriage-return redraw) and
degrades to periodic plain-text lines when the stream is redirected to a file.

It does not touch the model, loss, optimizer, sampler or any numerical behaviour;
it only reports the numbers the training loop already computes.
"""

from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any, TextIO

_BLOCKS = "▁▂▃▄▅▆▇█"


def progress_enabled(default: bool = True) -> bool:
    """Whether live progress rendering is turned on.

    Controlled by the ``SPRITELAB_PROGRESS`` environment variable so callers do
    not have to thread a flag through every config. Unset -> ``default``.
    """
    value = os.environ.get("SPRITELAB_PROGRESS")
    if value is None:
        return default
    token = value.strip().lower()
    if token in {"0", "off", "false", "no", "none", "disable", "disabled"}:
        return False
    return True


def _supports_unicode(stream: TextIO) -> bool:
    encoding = getattr(stream, "encoding", None) or ""
    try:
        "█▁".encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def format_duration(seconds: float | None) -> str:
    """Human friendly ``H:MM:SS`` / ``MM:SS`` duration."""
    if seconds is None or seconds != seconds or seconds < 0:  # noqa: PLR0124 - NaN guard
        return "--:--"
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def sparkline(values: list[float], *, width: int = 24, ascii_only: bool = False) -> str:
    """Render a compact unicode sparkline for the tail of ``values``."""
    if not values:
        return ""
    tail = values[-width:]
    lo = min(tail)
    hi = max(tail)
    if ascii_only:
        ramp = " .:-=+*#%@"
    else:
        ramp = " " + _BLOCKS
    span = hi - lo
    out = []
    for value in tail:
        if span <= 0:
            idx = len(ramp) - 1
        else:
            idx = int(round((value - lo) / span * (len(ramp) - 1)))
        out.append(ramp[max(0, min(len(ramp) - 1, idx))])
    return "".join(out)


class StepProgressBar:
    """A single-line, throttled training progress renderer.

    ``update`` is safe to call every optimizer step; it self-throttles terminal
    redraws by wall-clock time and, when output is not a TTY, emits an occasional
    plain line so redirected logs stay readable.
    """

    def __init__(
        self,
        total: int,
        *,
        desc: str = "train",
        stream: TextIO | None = None,
        enabled: bool | None = None,
        min_interval: float = 0.25,
        plain_lines: int = 20,
        bar_width: int = 26,
        ema_alpha: float = 0.1,
        spark_width: int = 20,
    ) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.stream = stream if stream is not None else sys.stderr
        self.enabled = progress_enabled() if enabled is None else bool(enabled)
        self.min_interval = float(min_interval)
        self.bar_width = int(bar_width)
        self.ema_alpha = float(ema_alpha)
        self.spark_width = int(spark_width)
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._unicode = _supports_unicode(self.stream)
        self._plain_every = max(1, self.total // max(1, int(plain_lines))) if self.total else 1
        self._start = time.perf_counter()
        self._last_draw = 0.0
        self._samples: deque[tuple[float, int]] = deque(maxlen=64)
        self._recent_loss: deque[float] = deque(maxlen=max(spark_width, 1))
        self.ema_loss: float | None = None
        self.best_loss: float | None = None
        self._line_len = 0
        self._closed = False

    def _rate(self, now: float, step: int) -> float | None:
        self._samples.append((now, step))
        if len(self._samples) < 2:
            return None
        t0, s0 = self._samples[0]
        dt = now - t0
        ds = step - s0
        if dt <= 0 or ds <= 0:
            return None
        return ds / dt

    def update(self, step: int, loss: float | None = None, *, lr: float | None = None) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        rate = self._rate(now, step)
        if loss is not None and loss == loss:  # noqa: PLR0124 - NaN guard
            self.ema_loss = loss if self.ema_loss is None else (1 - self.ema_alpha) * self.ema_loss + self.ema_alpha * loss
            self.best_loss = loss if self.best_loss is None else min(self.best_loss, loss)
            self._recent_loss.append(loss)

        is_last = self.total and step >= self.total
        if self._is_tty:
            if not is_last and (now - self._last_draw) < self.min_interval:
                return
            self._last_draw = now
            self._render_tty(step, loss, lr, rate, now, transient=not is_last)
        else:
            if not is_last and step % self._plain_every != 0 and step != 1:
                return
            self._render_plain(step, loss, lr, rate, now)

    def _stats(self, step: int, loss: float | None, lr: float | None, rate: float | None, now: float) -> str:
        elapsed = now - self._start
        eta = None if not rate or not self.total else max(0.0, (self.total - step) / rate)
        parts = [f"{self.desc}"]
        if self.total:
            parts.append(f"{step}/{self.total}")
        else:
            parts.append(f"step {step}")
        if loss is not None:
            parts.append(f"loss {loss:.4f}")
        if self.ema_loss is not None:
            parts.append(f"ema {self.ema_loss:.4f}")
        parts.append(f"{(rate or 0.0):.1f} it/s")
        parts.append(f"elapsed {format_duration(elapsed)}")
        if self.total:
            parts.append(f"eta {format_duration(eta)}")
        if lr is not None:
            parts.append(f"lr {lr:.2e}")
        return " | ".join(parts)

    def _bar(self, step: int) -> str:
        if not self.total:
            return ""
        frac = max(0.0, min(1.0, step / self.total))
        filled = int(round(frac * self.bar_width))
        if self._unicode:
            bar = "█" * filled + "░" * (self.bar_width - filled)
        else:
            bar = "#" * filled + "-" * (self.bar_width - filled)
        return f"{int(frac * 100):3d}% [{bar}]"

    def _render_tty(self, step, loss, lr, rate, now, *, transient: bool) -> None:
        spark = sparkline(list(self._recent_loss), width=self.spark_width, ascii_only=not self._unicode)
        line = f"{self._bar(step)} {self._stats(step, loss, lr, rate, now)}"
        if spark:
            line += f" {spark}"
        pad = " " * max(0, self._line_len - len(line))
        self._line_len = len(line)
        self.stream.write("\r" + line + pad)
        if not transient:
            self.stream.write("\n")
        self.stream.flush()

    def _render_plain(self, step, loss, lr, rate, now) -> None:
        self.stream.write(f"[{self._bar(step)}] {self._stats(step, loss, lr, rate, now)}\n")
        self.stream.flush()

    def close(self, *, final_loss: float | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.enabled:
            return
        now = time.perf_counter()
        elapsed = now - self._start
        rate = self._rate(now, self.total)
        loss = final_loss if final_loss is not None else self.ema_loss
        summary = (
            f"{self.desc} done | {self.total} steps | "
            f"final loss {('%.4f' % loss) if loss is not None else 'n/a'} | "
            f"best {('%.4f' % self.best_loss) if self.best_loss is not None else 'n/a'} | "
            f"{format_duration(elapsed)} | {(rate or 0.0):.1f} it/s"
        )
        if self._is_tty and self._line_len:
            self.stream.write("\r" + " " * self._line_len + "\r")
        self.stream.write(summary + "\n")
        self.stream.flush()

    def __enter__(self) -> "StepProgressBar":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
