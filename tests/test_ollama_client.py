import httpx
import pytest

from app import ollama_client


@pytest.mark.asyncio
async def test_embed_posts_to_embeddings_endpoint_and_returns_vector(monkeypatch):
    captured = {}

    async def fake_post(self, url, json):
        captured["url"] = url
        captured["json"] = json
        return httpx.Response(200, json={"embedding": [0.1, 0.2, 0.3]})

    monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

    result = await ollama_client.embed("hello world")

    assert result == [0.1, 0.2, 0.3]
    assert captured["url"].endswith("/api/embeddings")
    assert captured["json"]["prompt"] == "hello world"


@pytest.mark.asyncio
async def test_chat_stream_yields_response_tokens(monkeypatch):
    lines = [
        b'{"response": "Hel", "done": false}',
        b'{"response": "lo", "done": false}',
        b'{"response": "", "done": true}',
    ]

    class FakeStreamResponse:
        async def aiter_lines(self):
            for line in lines:
                yield line.decode()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def raise_for_status(self):
            return None

    def fake_stream(self, method, url, json):
        return FakeStreamResponse()

    monkeypatch.setattr(httpx.AsyncClient, "stream", fake_stream)

    tokens = [token async for token in ollama_client.chat_stream("hi", max_tokens=10)]

    assert tokens == ["Hel", "lo"]
