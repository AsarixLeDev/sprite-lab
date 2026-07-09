"""Declarative sheet specializations (e.g., rpg_496 rules) keyed by profile capability.

Previously the ``oga_496_rpg_icons`` checks were hardcoded as ``profile.name == "..."``
across multiple modules.  The ``sheet_specialization`` capability field on
``SourceProfile`` lets call sites check ``profile.sheet_specialization == "rpg_496"``
instead — same behaviour, single source of truth.
"""

from __future__ import annotations


def is_rpg_496_profile(profile: object) -> bool:
    """Return True when *profile* declares ``sheet_specialization="rpg_496"``."""
    return bool(getattr(profile, "sheet_specialization", None) == "rpg_496")
