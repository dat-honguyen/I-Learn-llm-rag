# I-Learn-llm-rag Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `llm-rag-api` service. A small FastAPI app that answers questions
grounded in a self-authored "learning journal" doc corpus, backed by Ollama running a
small quantized model, with an embedded sqlite-vec store for retrieval.

**Architecture:** FastAPI app with four small modules (chunking, sqlite-vec store, Ollama
HTTP client, ingest pipeline) wired together in `main.py`. On startup, ingest re-embeds any
changed corpus doc into sqlite-vec. `/chat` retrieves top-k chunks, builds a grounded
prompt, streams the model's answer back. API-only, no HTML frontend (lives in a separate
portfolio repo, out of scope here).

**Tech Stack:** Python 3.12, FastAPI, httpx (async Ollama HTTP client), sqlite-vec,
pytest + pytest-asyncio, Docker.

**Spec:** `docs/superpowers/specs/2026-08-15-homelab-llm-rag-design.md`

## Global Constraints

- API-only: no server-rendered HTML/static frontend in this repo.
- No shared-postgres. Vector storage is a local `sqlite-vec` file (see spec's "Key
  decisions" section).
- Password check must use constant-time comparison (`hmac.compare_digest`).
- CORS origin must be configurable via env, not hardcoded to `*` in production config.
- Corpus lives under `docs_corpus/` (not `docs/`, which holds specs/plans) and is a
  first-person "today I learn LLM" journal, not fabricated personal/resume data.
- License: MIT.
- README: plain human tone, no AI-generated boilerplate phrasing, short.

---

### Task 1: Project scaffolding

**Files:**
- Create: `requirements.txt`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `pytest.ini`

**Interfaces:**
- Produces: `app.config.settings`, object with attributes `ollama_url: str`,
  `chat_model: str`, `embed_model: str`, `app_password: str`, `cors_origin: str`,
  `db_path: str`, `docs_dir: str`, `top_k: int`, `max_question_len: int`,
  `max_output_tokens: int`.

- [ ] **Step 1: Write `requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.32.0
httpx==0.27.2
sqlite-vec==0.1.6
pytest==8.3.3
pytest-asyncio==0.24.0
```

- [ ] **Step 2: Write `app/config.py`**

```python
import os
from dataclasses import dataclass


@dataclass
class Settings:
    ollama_url: str = os.environ.get("OLLAMA_URL", "http://ollama:11434")
    chat_model: str = os.environ.get("CHAT_MODEL", "llama3.2:3b-instruct-q4_K_M")
    embed_model: str = os.environ.get("EMBED_MODEL", "nomic-embed-text")
    app_password: str = os.environ.get("APP_PASSWORD", "")
    cors_origin: str = os.environ.get("CORS_ORIGIN", "*")
    db_path: str = os.environ.get("DB_PATH", "data/vectors.db")
    docs_dir: str = os.environ.get("DOCS_DIR", "docs_corpus")
    top_k: int = int(os.environ.get("TOP_K", "4"))
    max_question_len: int = int(os.environ.get("MAX_QUESTION_LEN", "500"))
    max_output_tokens: int = int(os.environ.get("MAX_OUTPUT_TOKENS", "512"))


settings = Settings()
```

- [ ] **Step 3: Write `app/__init__.py`** (empty file, marks `app` as a package)

- [ ] **Step 4: Write `.env.example`**

```
OLLAMA_URL=http://ollama:11434
CHAT_MODEL=llama3.2:3b-instruct-q4_K_M
EMBED_MODEL=nomic-embed-text
APP_PASSWORD=change-me
CORS_ORIGIN=https://your-portfolio-site.example
DB_PATH=data/vectors.db
DOCS_DIR=docs_corpus
TOP_K=4
MAX_QUESTION_LEN=500
MAX_OUTPUT_TOKENS=512
```

- [ ] **Step 5: Write `.gitignore`**

```
__pycache__/
*.pyc
.venv/
data/
.env
```

- [ ] **Step 6: Write `pytest.ini`**

```ini
[pytest]
asyncio_mode = auto
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app/__init__.py app/config.py .env.example .gitignore pytest.ini
git commit -m "chore: project scaffolding and settings"
```

---

### Task 2: Chunking module

**Files:**
- Create: `app/chunking.py`
- Test: `tests/test_chunking.py`

**Interfaces:**
- Produces: `chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_chunking.py
from app.chunking import chunk_text


def test_empty_text_returns_no_chunks():
    assert chunk_text("") == []


def test_short_text_returns_single_chunk():
    text = "one two three"
    assert chunk_text(text, chunk_size=500, overlap=50) == ["one two three"]


def test_long_text_splits_into_overlapping_chunks():
    words = [f"word{i}" for i in range(120)]
    text = " ".join(words)

    chunks = chunk_text(text, chunk_size=50, overlap=10)

    assert len(chunks) == 3
    assert chunks[0] == " ".join(words[0:50])
    assert chunks[1] == " ".join(words[40:90])
    assert chunks[2] == " ".join(words[80:120])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chunking.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.chunking'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/chunking.py
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks = []
    step = chunk_size - overlap
    start = 0
    while True:
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chunking.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/chunking.py tests/test_chunking.py
git commit -m "feat: add word-based text chunking"
```

---

### Task 3: sqlite-vec store module

**Files:**
- Create: `app/store.py`
- Test: `tests/test_store.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `init_db(path: str) -> sqlite3.Connection`
  - `get_doc_hash(conn, doc_id: str) -> str | None`
  - `delete_doc(conn, doc_id: str) -> None`
  - `insert_chunk(conn, doc_id: str, chunk_index: int, text: str, content_hash: str, embedding: list[float]) -> None`
  - `commit(conn) -> None`
  - `top_k(conn, embedding: list[float], k: int) -> list[tuple[str, float]]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_store.py
from app import store

EMBED_DIM = 8


def make_embedding(seed: float) -> list[float]:
    return [seed] * EMBED_DIM


def test_insert_and_top_k_returns_closest_chunk(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)

    store.insert_chunk(conn, "doc1", 0, "chunk about cats", "hash1", make_embedding(1.0))
    store.insert_chunk(conn, "doc1", 1, "chunk about dogs", "hash1", make_embedding(5.0))
    store.commit(conn)

    results = store.top_k(conn, make_embedding(1.1), k=1)

    assert len(results) == 1
    assert results[0][0] == "chunk about cats"


def test_get_doc_hash_returns_none_when_absent(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)
    assert store.get_doc_hash(conn, "missing-doc") is None


def test_delete_doc_removes_its_chunks(tmp_path):
    conn = store.init_db(str(tmp_path / "vectors.db"), dim=EMBED_DIM)
    store.insert_chunk(conn, "doc1", 0, "text a", "hash1", make_embedding(1.0))
    store.commit(conn)

    store.delete_doc(conn, "doc1")
    store.commit(conn)

    assert store.get_doc_hash(conn, "doc1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.store'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/store.py
import sqlite3
import struct

import sqlite_vec


def _serialize(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def init_db(path: str, dim: int = 768) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            UNIQUE(doc_id, chunk_index)
        )
        """
    )
    conn.execute(
        f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{dim}])"
    )
    conn.commit()
    return conn


def get_doc_hash(conn: sqlite3.Connection, doc_id: str) -> str | None:
    row = conn.execute(
        "SELECT content_hash FROM chunks WHERE doc_id = ? LIMIT 1", (doc_id,)
    ).fetchone()
    return row[0] if row else None


def delete_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    ids = [
        row[0]
        for row in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))
    ]
    for chunk_id in ids:
        conn.execute("DELETE FROM vec_chunks WHERE rowid = ?", (chunk_id,))
    conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))


def insert_chunk(
    conn: sqlite3.Connection,
    doc_id: str,
    chunk_index: int,
    text: str,
    content_hash: str,
    embedding: list[float],
) -> None:
    cursor = conn.execute(
        "INSERT INTO chunks (doc_id, chunk_index, text, content_hash) VALUES (?, ?, ?, ?)",
        (doc_id, chunk_index, text, content_hash),
    )
    chunk_id = cursor.lastrowid
    conn.execute(
        "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
        (chunk_id, _serialize(embedding)),
    )


def commit(conn: sqlite3.Connection) -> None:
    conn.commit()


def top_k(conn: sqlite3.Connection, embedding: list[float], k: int) -> list[tuple[str, float]]:
    rows = conn.execute(
        """
        SELECT c.text, v.distance
        FROM vec_chunks v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (_serialize(embedding), k),
    ).fetchall()
    return [(text, distance) for text, distance in rows]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/store.py tests/test_store.py
git commit -m "feat: add sqlite-vec backed chunk store"
```

---

### Task 4: Ollama HTTP client

**Files:**
- Create: `app/ollama_client.py`
- Test: `tests/test_ollama_client.py`

**Interfaces:**
- Consumes: `app.config.settings` (`ollama_url`, `embed_model`, `chat_model`).
- Produces:
  - `async def embed(text: str) -> list[float]`
  - `async def chat_stream(prompt: str, max_tokens: int) -> AsyncIterator[str]`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ollama_client.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ollama_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ollama_client'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/ollama_client.py
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
        response.raise_for_status()
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ollama_client.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ollama_client.py tests/test_ollama_client.py
git commit -m "feat: add async Ollama HTTP client for embeddings and chat"
```

---

### Task 5: Ingest pipeline

**Files:**
- Create: `app/ingest.py`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `chunk_text` (Task 2), `store.get_doc_hash`/`delete_doc`/`insert_chunk`/`commit` (Task 3).
- Produces: `async def ingest_docs(docs_dir: Path, conn, embed_fn: Callable[[str], Awaitable[list[float]]]) -> int` (returns count of docs (re)ingested).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ingest.py
from pathlib import Path

from app import store
from app.ingest import ingest_docs


async def fake_embed(text: str) -> list[float]:
    return [float(len(text))] * 8


async def test_ingest_embeds_new_docs(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    (docs_dir / "note.md").write_text("hello world this is a note", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)

    count = await ingest_docs(docs_dir, conn, fake_embed)

    assert count == 1
    assert store.get_doc_hash(conn, "note") is not None


async def test_ingest_skips_unchanged_docs_on_second_run(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    (docs_dir / "note.md").write_text("same content", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)

    first = await ingest_docs(docs_dir, conn, fake_embed)
    second = await ingest_docs(docs_dir, conn, fake_embed)

    assert first == 1
    assert second == 0


async def test_ingest_reembeds_changed_docs(tmp_path):
    docs_dir = tmp_path / "docs_corpus"
    docs_dir.mkdir()
    doc_path = docs_dir / "note.md"
    doc_path.write_text("version one", encoding="utf-8")

    conn = store.init_db(str(tmp_path / "vectors.db"), dim=8)
    await ingest_docs(docs_dir, conn, fake_embed)

    doc_path.write_text("version two, changed", encoding="utf-8")
    second = await ingest_docs(docs_dir, conn, fake_embed)

    assert second == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingest'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/ingest.py
import hashlib
from pathlib import Path
from typing import Awaitable, Callable

from . import store
from .chunking import chunk_text


async def ingest_docs(
    docs_dir: Path,
    conn,
    embed_fn: Callable[[str], Awaitable[list[float]]],
) -> int:
    ingested = 0
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        doc_id = path.stem

        if store.get_doc_hash(conn, doc_id) == content_hash:
            continue

        store.delete_doc(conn, doc_id)
        for index, chunk in enumerate(chunk_text(text)):
            embedding = await embed_fn(chunk)
            store.insert_chunk(conn, doc_id, index, chunk, content_hash, embedding)
        store.commit(conn)
        ingested += 1

    return ingested
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingest.py tests/test_ingest.py
git commit -m "feat: add doc ingest pipeline with hash-based re-embed skipping"
```

---

### Task 6: FastAPI app (`/health`, `/chat`)

**Files:**
- Create: `app/main.py`
- Test: `tests/test_main.py`

**Interfaces:**
- Consumes: `settings` (Task 1), `store.init_db`/`top_k` (Task 3), `ollama_client.embed`/`chat_stream`
  (Task 4), `ingest_docs` (Task 5).
- Produces: `app` (FastAPI instance) importable as `app.main.app`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main.py
from pathlib import Path

from fastapi.testclient import TestClient

from app import main, ollama_client, store


async def fake_embed(text: str) -> list[float]:
    return [0.1] * 8


async def fake_chat_stream(prompt: str, max_tokens: int):
    for token in ["Answer", " goes", " here"]:
        yield token


def make_client(tmp_path, monkeypatch):
    monkeypatch.setattr(main.settings, "app_password", "secret")
    monkeypatch.setattr(main.settings, "db_path", str(tmp_path / "vectors.db"))
    monkeypatch.setattr(main.settings, "docs_dir", str(tmp_path / "docs_corpus"))
    (tmp_path / "docs_corpus").mkdir()
    (tmp_path / "docs_corpus" / "note.md").write_text("cats are great pets", encoding="utf-8")

    monkeypatch.setattr(ollama_client, "embed", fake_embed)
    monkeypatch.setattr(ollama_client, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(store, "init_db", lambda path, dim=768: store.init_db.__wrapped__(path, dim=8))

    return TestClient(main.app)


def test_health_returns_ok(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_chat_rejects_wrong_password(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.post("/chat", json={"question": "hi", "password": "wrong"})
        assert response.status_code == 401


def test_chat_streams_answer_with_correct_password(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        response = client.post("/chat", json={"question": "hi", "password": "secret"})
        assert response.status_code == 200
        assert response.text == "Answer goes here"


def test_chat_rejects_question_over_max_length(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        monkeypatch.setattr(main.settings, "max_question_len", 5)
        response = client.post("/chat", json={"question": "way too long", "password": "secret"})
        assert response.status_code == 400
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Write minimal implementation**

Note: `store.init_db` takes a `dim` kwarg used for tests via `functools.wraps` so the
`__wrapped__` monkeypatch trick above works. Implementing `init_db` with `functools.wraps`
is unnecessary complexity, so instead expose a `DEFAULT_DIM = 768` module constant in
`store.py` used by `main.py`'s lifespan, and drop the `__wrapped__` indirection from the
test in favor of directly monkeypatching `store.DEFAULT_DIM`. Revise the test's
`make_client` to:

```python
    monkeypatch.setattr(store, "DEFAULT_DIM", 8)
```

(replacing the `monkeypatch.setattr(store, "init_db", ...)` line above), and add to
`app/store.py` from Task 3:

```python
DEFAULT_DIM = 768


def init_db(path: str, dim: int = DEFAULT_DIM) -> sqlite3.Connection:
    ...
```

(`dim: int = DEFAULT_DIM` replaces the literal `768` default from Task 3's implementation.)

```python
# app/main.py
import hmac
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import ollama_client, store
from .config import settings
from .ingest import ingest_docs


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = store.init_db(settings.db_path, dim=store.DEFAULT_DIM)
    await ingest_docs(Path(settings.docs_dir), conn, ollama_client.embed)
    app.state.conn = conn
    yield
    conn.close()


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.cors_origin],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    password: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not hmac.compare_digest(req.password, settings.app_password):
        raise HTTPException(status_code=401, detail="invalid password")
    if len(req.question) > settings.max_question_len:
        raise HTTPException(status_code=400, detail="question too long")

    embedding = await ollama_client.embed(req.question)
    matches = store.top_k(app.state.conn, embedding, settings.top_k)
    context = "\n\n".join(text for text, _ in matches)
    prompt = (
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    async def stream():
        async for token in ollama_client.chat_stream(prompt, max_tokens=settings.max_output_tokens):
            yield token

    return StreamingResponse(stream(), media_type="text/plain")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_main.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Run full test suite**

Run: `pytest -v`
Expected: All tests from Tasks 2-6 PASS.

- [ ] **Step 6: Commit**

```bash
git add app/main.py app/store.py tests/test_main.py
git commit -m "feat: add FastAPI app with /health and /chat endpoints"
```

---

### Task 7: Corpus content ("today I learn LLM" journal)

**Files:**
- Create: `docs_corpus/about-this-project.md`
- Create: `docs_corpus/how-the-rag-works.md`
- Create: `docs_corpus/homelab-setup.md`

**Interfaces:** none (data files, consumed by `ingest_docs` at runtime).

- [ ] **Step 1: Write `docs_corpus/about-this-project.md`**

```markdown
# Why I built this

I wanted to actually learn how RAG works instead of just reading about it, so I built
a small end-to-end version myself: a tiny FastAPI service, a local model running on my
own homelab box, and a vector store holding chunks of these exact notes you're reading.

It runs on a Ryzen 5 8600G with 16GB of RAM. No GPU, no cloud API keys, no credit card.
Everything you ask it gets answered by actually searching these markdown files first,
then handing the relevant pieces to the model as context. If the answer isn't in these
notes, it's supposed to say so instead of making something up.
```

- [ ] **Step 2: Write `docs_corpus/how-the-rag-works.md`**

```markdown
# How the retrieval actually works

When you ask a question, three things happen in order:

1. The question gets turned into a vector (an embedding) using Ollama's
   `nomic-embed-text` model.
2. That vector gets compared against vectors for every chunk of these notes, stored in
   a local SQLite file using the `sqlite-vec` extension. The closest few chunks win.
3. Those chunks get pasted into a prompt along with your question, and the whole thing
   gets sent to a small local chat model (`llama3.2:3b-instruct`, quantized down to fit
   comfortably in RAM alongside everything else running on the box).

I picked `sqlite-vec` over a "real" vector database on purpose. There are maybe 20-30
chunks total across these notes. Running a separate Postgres+pgvector container for
that would be solving a problem I don't have.
```

- [ ] **Step 3: Write `docs_corpus/homelab-setup.md`**

```markdown
# Where this actually runs

This service runs as two containers on my home server, managed with Podman + Quadlet
(not docker-compose). One container runs Ollama and isn't reachable from outside the
box at all. Only this API container can talk to it, over an internal network. The API
container is the only thing Caddy proxies to, and Caddy is the only thing a Cloudflare
Tunnel exposes to the internet.

The public endpoint is rate-limited per IP and sits behind a shared password, mostly to
stop random bots from running up the CPU on a box that's also hosting a few other
things for me.
```

- [ ] **Step 4: Commit**

```bash
git add docs_corpus/
git commit -m "docs: add corpus content for the RAG demo"
```

---

### Task 8: Dockerfile and CI workflow

**Files:**
- Create: `Dockerfile`
- Create: `.dockerignore`
- Create: `.github/workflows/ci.yml`

**Interfaces:** none (build/CI config).

- [ ] **Step 1: Write `Dockerfile`**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app app
COPY docs_corpus docs_corpus

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 2: Write `.dockerignore`**

```
.venv
__pycache__
*.pyc
data/
tests/
docs/
.git
```

- [ ] **Step 3: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install -r requirements.txt
      - run: pytest -v
      - name: Build image
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: podman build -t llm-rag-api:latest .
      - name: Restart service
        if: github.ref == 'refs/heads/main' && github.event_name == 'push'
        run: systemctl --user restart llm-rag-api
```

- [ ] **Step 4: Commit**

```bash
git add Dockerfile .dockerignore .github/workflows/ci.yml
git commit -m "chore: add Dockerfile and self-hosted-runner CI workflow"
```

---

### Task 9: License and README

**Files:**
- Create: `LICENSE`
- Modify: `README.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Write `LICENSE`** (MIT, current year, placeholder name replaced with the
  repo owner's GitHub username `dat-honguyen`)

```
MIT License

Copyright (c) 2026 dat-honguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 2: Rewrite `README.md`** (plain, first-person, no marketing tone)

```markdown
# Today I Learn: LLM

A small RAG chat API I built to actually understand how retrieval-augmented
generation works, instead of just reading about it.

It runs on my home server: a local model (Ollama, `llama3.2:3b-instruct`,
CPU only, no GPU) answers questions using only these notes as context.
See `docs_corpus/` for what it actually knows about. No OpenAI key, no
cloud bill, just a Ryzen APU doing its best.

## What's here

- `app/`: the FastAPI service (chunking, embeddings, a tiny sqlite-vec
  store, the `/chat` endpoint)
- `docs_corpus/`: the notes the model is grounded in
- `tests/`: pytest, mocks Ollama so tests don't need a live model

## Running it locally

You need Ollama running somewhere with `llama3.2:3b-instruct` and
`nomic-embed-text` pulled.

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in APP_PASSWORD at least
uvicorn app.main:app --reload
```

Then:

```bash
curl -X POST localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "why sqlite-vec instead of postgres?", "password": "change-me"}'
```

## Running the tests

```bash
pytest -v
```

No live model needed, Ollama calls are mocked.

## Why it's built this way

Written up in more detail in `docs/superpowers/specs/`, but the short
version: small quantized model because the box has no GPU, sqlite-vec
instead of a shared Postgres because the corpus is tiny (~20-30 chunks)
and dragging in pgvector for that would be solving a problem I don't
have, and a shared password + rate limiting instead of full SSO because
this is a public demo, not an account system.

## License

MIT, see `LICENSE`.
```

- [ ] **Step 3: Commit**

```bash
git add LICENSE README.md
git commit -m "docs: add MIT license and rewrite README"
```

---

### Task 10: Deployment reference (Quadlet units + Caddy block)

**Files:**
- Create: `deploy/ollama.container`
- Create: `deploy/llm-rag-api.container`
- Create: `deploy/llm-rag.network`
- Create: `deploy/caddy-llm-rag.snippet`
- Create: `docs/DEPLOY.md`

**Interfaces:** none (reference deployment artifacts, not applied by this plan; applying
them to the live homelab box is a separate follow-up task requiring SSH access).

- [ ] **Step 1: Write `deploy/llm-rag.network`**

```ini
[Network]
NetworkName=llm-rag
```

- [ ] **Step 2: Write `deploy/ollama.container`**

```ini
[Unit]
Description=Ollama (local LLM inference)

[Container]
Image=docker.io/ollama/ollama:latest
AutoUpdate=registry
ContainerName=ollama
Network=llm-rag.network
Volume=ollama-data:/root/.ollama

[Service]
Restart=on-failure

[Install]
WantedBy=default.target
```

- [ ] **Step 3: Write `deploy/llm-rag-api.container`**

```ini
[Unit]
Description=llm-rag-api (RAG chat service)
After=ollama.service

[Container]
Image=localhost/llm-rag-api:latest
ContainerName=llm-rag-api
Network=llm-rag.network
Volume=llm-rag-api-data:/app/data
Environment=OLLAMA_URL=http://ollama:11434
Environment=CHAT_MODEL=llama3.2:3b-instruct-q4_K_M
Environment=EMBED_MODEL=nomic-embed-text
Environment=APP_PASSWORD=%t/llm-rag-api-password
Environment=CORS_ORIGIN=https://your-portfolio-site.example
PublishPort=127.0.0.1:8010:8000

[Service]
Restart=on-failure

[Install]
WantedBy=default.target
```

Note: `Environment=APP_PASSWORD=%t/llm-rag-api-password` is a placeholder. Replace with
the actual password value (or an `EnvironmentFile=` pointing at a secrets file outside
git) when deploying; do not commit a real password.

- [ ] **Step 4: Write `deploy/caddy-llm-rag.snippet`**

```caddyfile
http://llm.datisa.dev {
	reverse_proxy llm-rag-api:8000 {
		header_up X-Forwarded-Proto https
	}
	rate_limit {
		zone llm_rag {
			key {remote_host}
			events 5
			window 1m
		}
	}
}
```

- [ ] **Step 5: Write `docs/DEPLOY.md`**

```markdown
# Deploying to the homelab box

1. Copy `deploy/llm-rag.network`, `deploy/ollama.container`, and
   `deploy/llm-rag-api.container` into `~/.config/containers/systemd/` on `homelab`.
2. Replace the `APP_PASSWORD` placeholder in `llm-rag-api.container` with a real value
   (don't commit it).
3. `systemctl --user daemon-reload`
4. `systemctl --user start ollama`
5. `podman exec ollama ollama pull llama3.2:3b-instruct-q4_K_M`
6. `podman exec ollama ollama pull nomic-embed-text`
7. Build the API image (or let the self-hosted Actions runner do it via
   `.github/workflows/ci.yml`), then `systemctl --user start llm-rag-api`.
8. Append `deploy/caddy-llm-rag.snippet`'s contents to `~/caddy/Caddyfile`, then
   `systemctl --user restart caddy`.
9. `cloudflared tunnel route dns mydebian-sv llm.datisa.dev` (check
   `/etc/cloudflared/config.yml` first, ingress rule is likely already generic).
10. Smoke test: `curl -X POST https://llm.datisa.dev/chat -d '{"question": "...", "password": "..."}'`.
```

- [ ] **Step 6: Commit**

```bash
git add deploy/ docs/DEPLOY.md
git commit -m "docs: add Quadlet deployment reference for the homelab box"
```

---

## Post-plan follow-up (not part of this plan)

Actually deploying to the `homelab` box (steps in `docs/DEPLOY.md`) requires live SSH
access and is a separate task. This plan produces a working, tested service and its
deployment reference, not a live deployment.
