"""Tests for the single logging setup."""

import logging
from collections.abc import Iterator

import pytest
from rich.logging import RichHandler

from asr_assess.core.logging import setup_logging


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers, root.level = handlers, level


def rich_handlers() -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if isinstance(h, RichHandler)]


def test_installs_one_rich_handler_at_level() -> None:
    setup_logging("DEBUG")
    assert len(rich_handlers()) == 1
    assert logging.getLogger().level == logging.DEBUG


def test_is_idempotent() -> None:
    setup_logging("INFO")
    setup_logging("WARNING")
    assert len(rich_handlers()) == 1
    assert logging.getLogger().level == logging.WARNING
