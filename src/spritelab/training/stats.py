"""Small statistics helpers shared by v2 Phase 0 diagnostics reports.

No training or model code here; these are deterministic closed-form helpers for
turning small-n rate metrics (e.g. n=96 OOD samples) into confidence intervals
so report consumers don't over-interpret point-estimate deltas as significant.
"""

from __future__ import annotations

import math

DEFAULT_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_confidence_interval(successes: int, n: int, *, z: float = DEFAULT_Z_95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial rate.

    ``successes``/``n`` are raw counts, not a pre-divided rate. ``n <= 0`` returns the
    maximally uninformative ``(0.0, 1.0)`` interval rather than raising, since callers
    report this alongside a possibly-``None`` rate for empty samples.
    """

    n = int(n)
    if n <= 0:
        return (0.0, 1.0)
    successes = max(0, min(int(successes), n))
    p_hat = successes / float(n)
    z2 = float(z) * float(z)
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2 * n)) / denom
    margin = (float(z) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))) / denom
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (lower, upper)


def wilson_ci_from_rate(rate: float | None, n: int, *, z: float = DEFAULT_Z_95) -> list[float] | None:
    """Wilson interval for a rate already computed as ``successes / n``.

    Returns ``None`` when ``rate`` is ``None`` or ``n <= 0`` (matches the existing
    convention of rate fields being ``None`` when there's no denominator).
    """

    if rate is None or n <= 0:
        return None
    successes = round(float(rate) * int(n))
    lower, upper = wilson_confidence_interval(successes, n, z=z)
    return [lower, upper]
