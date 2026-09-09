"""Application wiring and entry point."""

from __future__ import annotations

from pathlib import Path

from zoomy.comfy_connection import ComfyConnection
from zoomy.family_catalog import FAMILY_CATALOG
from zoomy.frame_repository import FrameRepository
from zoomy.interface import build_application
from zoomy.settings import Settings


def main() -> None:
    """Load settings, wire the collaborators, and launch the web interface."""
    settings = Settings.from_environment()
    connection = ComfyConnection(settings.comfy_address)
    repository = FrameRepository(Path(settings.output_directory))
    application = build_application(settings, connection, FAMILY_CATALOG, repository)
    application.queue(default_concurrency_limit=1)
    application.launch(
        server_name=settings.interface_address,
        server_port=settings.interface_port,
        allowed_paths=[settings.output_directory],
        show_error=True,
        inbrowser=False,
        # Server-side rendering needs a Node.js runtime the slim image does not
        # ship; the standard client-side rendering works everywhere.
        ssr_mode=False,
    )
