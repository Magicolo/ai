"""CivitAI redirect auth: the bearer token must not leak to the file CDN."""

from __future__ import annotations

import io
import sys
import urllib.request
from http.client import HTTPMessage
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from download_models import _CivitaiRedirectHandler, build_civitai_request


def test_redirect_to_cdn_strips_authorization_header() -> None:
    """The signed CDN target carries its own auth; ours gets a 403 there."""
    original = urllib.request.Request(
        "https://civitai.com/api/download/models/3028150",
        headers={"Authorization": "Bearer secret-token"},
    )
    headers = HTTPMessage()
    redirected = _CivitaiRedirectHandler().redirect_request(
        original, io.BytesIO(b""), 302, "Found", headers, "https://cdn.example/file?sig=abc"
    )
    assert redirected is not None
    assert redirected.full_url == "https://cdn.example/file?sig=abc"
    assert redirected.get_header("Authorization") is None


def test_civitai_request_carries_browser_user_agent() -> None:
    """Cloudflare answers urllib's default UA with 1010 (access denied)."""
    request = build_civitai_request("https://civitai.com/api/download/models/1", "token")
    user_agent = request.get_header("User-agent")
    assert user_agent is not None
    assert "Python-urllib" not in user_agent
    assert request.get_header("Authorization") == "Bearer token"
