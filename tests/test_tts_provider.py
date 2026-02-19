from pathlib import Path

import pytest

from nanobot.providers.tts import OpenAITTSProvider


class _FakeResponse:
    def __init__(self, content: bytes):
        self.content = content

    def raise_for_status(self):
        return None


class _SuccessClient:
    def __init__(self, *args, **kwargs):
        self.calls = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return _FakeResponse(b"audio-bytes")


class _ErrorClient(_SuccessClient):
    async def post(self, *args, **kwargs):
        raise RuntimeError("boom")


@pytest.mark.asyncio
async def test_openai_tts_provider_success(monkeypatch, tmp_path):
    monkeypatch.setattr("nanobot.providers.tts.httpx.AsyncClient", _SuccessClient)
    provider = OpenAITTSProvider(api_key="sk-test", api_base="https://api.openai.com/v1")
    provider.output_dir = tmp_path
    provider.output_dir.mkdir(parents=True, exist_ok=True)

    out = await provider.synthesize("Hello world", "opus")

    assert out is not None
    assert out.suffix == ".opus"
    assert out.exists()
    assert out.read_bytes() == b"audio-bytes"


@pytest.mark.asyncio
async def test_openai_tts_provider_error_returns_none(monkeypatch, tmp_path):
    monkeypatch.setattr("nanobot.providers.tts.httpx.AsyncClient", _ErrorClient)
    provider = OpenAITTSProvider(api_key="sk-test", api_base="https://api.openai.com/v1")
    provider.output_dir = tmp_path
    provider.output_dir.mkdir(parents=True, exist_ok=True)

    out = await provider.synthesize("Hello world", "mp3")
    assert out is None


@pytest.mark.asyncio
async def test_openai_tts_provider_empty_text_short_circuit(tmp_path):
    provider = OpenAITTSProvider(api_key="sk-test", api_base="https://api.openai.com/v1")
    provider.output_dir = tmp_path
    provider.output_dir.mkdir(parents=True, exist_ok=True)

    out = await provider.synthesize("   ", "mp3")
    assert out is None
    assert list(Path(tmp_path).glob("*")) == []
