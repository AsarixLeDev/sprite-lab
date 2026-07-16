"""Unified, offline Sprite Lab v3 web product shell."""

from spritelab.product_web.app import create_app
from spritelab.product_web.cli import main, run_server
from spritelab.product_web.events import EventRepository, RunSnapshot

__all__ = ["EventRepository", "RunSnapshot", "create_app", "main", "run_server"]
