"""Application wiring and entry point."""

from __future__ import annotations

from pathlib import Path

from zoomy.family_catalog import FAMILY_CATALOG
from zoomy.frame_repository import FrameRepository
from zoomy.interface import STUDIO_CSS, STUDIO_THEME, build_application
from zoomy.local_engine import LocalEngine
from zoomy.settings import Settings


def main() -> None:
    """Load settings, wire the collaborators, and launch the web interface."""
    settings = Settings.from_environment()
    repository = FrameRepository(Path(settings.output_directory))
    engine = LocalEngine(
        models_directory=Path(settings.models_directory),
        seed_directory=Path(settings.seed_directory),
        repository=repository,
        device=settings.cuda_device,
        music_project_directory=Path(settings.music_project_directory),
    )
    application = build_application(settings, engine, FAMILY_CATALOG, repository)
    application.queue(default_concurrency_limit=1)
    application.launch(
        server_name=settings.interface_address,
        server_port=settings.interface_port,
        allowed_paths=[settings.output_directory],
        show_error=True,
        inbrowser=False,
        theme=STUDIO_THEME,
        css=STUDIO_CSS,
        # Server-side rendering needs a Node.js runtime the slim image does not
        # ship; the standard client-side rendering works everywhere.
        ssr_mode=False,
    )
