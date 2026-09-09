"""Tests for the Gradio interface assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gradio as gr

from zoomy.comfy_connection import ComfyConnection
from zoomy.family_catalog import FAMILY_CATALOG
from zoomy.frame_repository import FrameRepository
from zoomy.interface import build_application
from zoomy.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


def test_build_application_returns_blocks(tmp_path: Path) -> None:
    """The full panel assembles without launching or touching the network."""
    settings = Settings(
        comfy_address="http://localhost:8188",
        output_directory=str(tmp_path),
        interface_address="127.0.0.1",
        interface_port=7861,
        operation_timeout_seconds=60.0,
        poll_interval_seconds=1.0,
    )
    application = build_application(
        settings=settings,
        connection=ComfyConnection("http://localhost:8188"),
        catalog=FAMILY_CATALOG,
        repository=FrameRepository(tmp_path),
    )
    assert isinstance(application, gr.Blocks)
