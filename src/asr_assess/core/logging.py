"""The one logging setup used by every command."""

import logging

from rich.logging import RichHandler


def setup_logging(level: str) -> None:
    """Route the root logger through a single Rich handler at ``level``; safe to call twice."""
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, RichHandler)]
    root.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    root.setLevel(level.upper())
