"""Small, dependency-free chart and product UI components.

Every chart is rendered as semantic HTML plus inline SVG. Values are never
interpolated: the visual and its textual alternative contain only supplied
points.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from html import escape
from typing import Any

from markupsafe import Markup

NO_DATA = '<div class="chart-empty" role="status">No data available</div>'


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _fmt(value: float) -> str:
    return f"{value:.4g}"


def _table(title: str, rows: Iterable[tuple[str, float]]) -> str:
    body = "".join(
        f'<tr><th scope="row">{escape(label)}</th><td>{escape(_fmt(value))}</td></tr>' for label, value in rows
    )
    return (
        '<details class="chart-table"><summary>Text data for '
        f"{escape(title)}</summary><table><thead><tr><th>Point</th><th>Value</th></tr></thead><tbody>{body}</tbody>"
        "</table></details>"
    )


def line_chart(points: Sequence[tuple[str, Any]], *, title: str = "Metric") -> Markup:
    """Render actual labeled points as a responsive line chart."""

    clean = [(str(label), number) for label, value in points if (number := _number(value)) is not None]
    if not clean:
        return Markup(NO_DATA)
    values = [value for _, value in clean]
    low, high = min(values), max(values)
    span = high - low or 1.0
    width, height, pad = 640.0, 220.0, 24.0
    denominator = max(1, len(clean) - 1)
    coordinates = [
        (
            pad + index * (width - 2 * pad) / denominator,
            height - pad - (value - low) * (height - 2 * pad) / span,
        )
        for index, (_, value) in enumerate(clean)
    ]
    polyline = " ".join(f"{x:.2f},{y:.2f}" for x, y in coordinates)
    dots = "".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"><title>{escape(label)}: {escape(_fmt(value))}</title></circle>'
        for (label, value), (x, y) in zip(clean, coordinates, strict=True)
    )
    svg = (
        f'<svg class="chart chart-line" viewBox="0 0 {width:.0f} {height:.0f}" role="img" '
        f'aria-label="{escape(title)} line chart with {len(clean)} points">'
        f'<polyline points="{polyline}" fill="none" vector-effect="non-scaling-stroke" />{dots}</svg>'
    )
    return Markup(
        f'<figure class="chart-frame"><figcaption>{escape(title)}</figcaption>{svg}{_table(title, clean)}</figure>'
    )


def bar_chart(bars: Sequence[tuple[str, Any]], *, title: str = "Values") -> Markup:
    """Render supplied categories as an offline bar chart."""

    clean = [(str(label), number) for label, value in bars if (number := _number(value)) is not None]
    if not clean:
        return Markup(NO_DATA)
    maximum = max((abs(value) for _, value in clean), default=0.0) or 1.0
    items = "".join(
        '<li><span class="bar-label">'
        f'{escape(label)}</span><span class="bar-track"><span class="bar-value" style="--bar-size:{abs(value) / maximum:.4f}"></span>'
        f'</span><span class="bar-number">{escape(_fmt(value))}</span></li>'
        for label, value in clean
    )
    return Markup(
        f'<figure class="chart-frame"><figcaption>{escape(title)}</figcaption><ul class="chart-bars">{items}</ul>'
        f"{_table(title, clean)}</figure>"
    )


def distribution(bins: Sequence[tuple[str, Any]], *, title: str = "Distribution") -> Markup:
    """Render caller-supplied bins without inventing or smoothing values."""

    return bar_chart(bins, title=title)


def metric_card(label: str, value: Any | None, *, hint: str = "") -> Markup:
    shown = "No data" if value is None or value == "" else str(value)
    hint_html = f'<span class="metric-hint">{escape(hint)}</span>' if hint else ""
    return Markup(
        '<article class="metric-card"><span class="metric-label">'
        f"{escape(label)}</span><strong>{escape(shown)}</strong>{hint_html}</article>"
    )


def image_gallery(images: Sequence[Mapping[str, Any]], *, title: str = "Images") -> Markup:
    """Render trusted local/route image URLs with mandatory alt text."""

    clean = [item for item in images if str(item.get("src", "")).startswith(("/", "data:image/"))]
    if not clean:
        return Markup(NO_DATA)
    cards = "".join(
        '<li><figure><img loading="lazy" src="'
        f'{escape(str(item["src"]), quote=True)}" alt="{escape(str(item.get("alt") or "Sprite preview"), quote=True)}">'
        f"<figcaption>{escape(str(item.get('caption') or item.get('alt') or 'Sprite'))}</figcaption></figure></li>"
        for item in clean
    )
    return Markup(
        f'<section class="gallery" aria-label="{escape(title, quote=True)}"><h3>{escape(title)}</h3><ul>{cards}</ul></section>'
    )


def run_timeline(stages: Sequence[Mapping[str, Any]], *, title: str = "Run timeline") -> Markup:
    if not stages:
        return Markup(NO_DATA)
    items = "".join(
        '<li><span class="timeline-dot" aria-hidden="true"></span><div><strong>'
        f'{escape(str(item.get("stage", "Stage")).replace("-", " ").replace("_", " ").title())}</strong><span class="status-text">'
        f"{escape(str(item.get('status', 'Unknown')).replace('_', ' ').title())}</span>"
        f"<p>{escape(str(item.get('message', '')))}</p></div></li>"
        for item in stages
    )
    return Markup(
        f'<section class="timeline" aria-label="{escape(title, quote=True)}"><h3>{escape(title)}</h3><ol>{items}</ol></section>'
    )


__all__ = ["bar_chart", "distribution", "image_gallery", "line_chart", "metric_card", "run_timeline"]
