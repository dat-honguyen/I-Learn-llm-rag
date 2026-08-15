import json
from typing import AsyncIterator

import httpx

from .config import settings


async def embed(text: str) -> list[float]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.ollama_url}/api/embeddings",
            json={"model": settings.embed_model, "prompt": text},
        )
        return response.json()["embedding"]


async def chat_stream(prompt: str, max_tokens: int) -> AsyncIterator[str]:
    async with httpx.AsyncClient(timeout=None) as client:
        stream_ctx = client.stream(
            "POST",
            f"{settings.ollama_url}/api/generate",
            json={
                "model": settings.chat_model,
                "prompt": prompt,
                "stream": True,
                "options": {"num_predict": max_tokens},
            },
        )
        async with stream_ctx as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if payload.get("response"):
                    yield payload["response"]
                if payload.get("done"):
                    break
