from __future__ import annotations

import json
import threading
import urllib.request

from paperqa.research.config import CLOUDFLARE_MODEL
from paperqa.research.service import ResearchService
from paperqa.ui.app import Server, load_asset, safe_filename


def test_ui_assets_and_filename_validation():
    assert b"<html" in load_asset("index.html").lower()
    assert safe_filename("paper.pdf") == "paper.pdf"


def test_ui_config_exposes_only_cloudflare(tmp_path):
    service = ResearchService(tmp_path)
    server = Server(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{server.server_port}/api/config", timeout=5
        ) as response:
            payload = json.load(response)
        assert payload["default_model"] == CLOUDFLARE_MODEL
        assert [item["provider"] for item in payload["model_options"]] == ["cloudflare"]
        assert "groq_configured" not in payload and "openai_configured" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
