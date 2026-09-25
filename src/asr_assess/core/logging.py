"""The one logging setup used by every command."""

import logging

from rich.logging import RichHandler

# Libraries that log every HTTP request at INFO; their warnings still come through.
_CHATTY_LOGGERS = ("httpx", "httpcore", "huggingface_hub", "fsspec", "urllib3")


def setup_logging(level: str) -> None:
    """Route the root logger through a single Rich handler at ``level``; safe to call twice."""
    root = logging.getLogger()
    root.handlers = [h for h in root.handlers if not isinstance(h, RichHandler)]
    root.addHandler(RichHandler(rich_tracebacks=True, show_path=False))
    root.setLevel(level.upper())
    for name in _CHATTY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
