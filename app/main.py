import hmac
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from . import ollama_client, store
from .config import settings
from .ingest import ingest_docs

SESSION_HISTORY_TURNS = 6
MAX_SESSIONS = 500


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = store.init_db(settings.db_path, dim=store.DEFAULT_DIM)
    await ingest_docs(Path(settings.docs_dir), conn, ollama_client.embed)
    if settings.private_docs_dir and Path(settings.private_docs_dir).is_dir():
        await ingest_docs(Path(settings.private_docs_dir), conn, ollama_client.embed)
    app.state.conn = conn
    app.state.sessions = {}
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
    session_id: str | None = None


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    if not hmac.compare_digest(req.password, settings.app_password):
        raise HTTPException(status_code=401, detail="invalid password")
    if len(req.question) > settings.max_question_len:
        raise HTTPException(status_code=400, detail="question too long")

    history = app.state.sessions.get(req.session_id) if req.session_id else None
    recap = ""
    if history:
        recap_lines = "\n".join(f"Q: {q}\nA: {a}" for q, a in history)
        recap = f"Recent conversation with this visitor:\n{recap_lines}\n\n"

    embedding = await ollama_client.embed(req.question)
    matches = store.top_k(app.state.conn, embedding, settings.top_k)
    context = "\n\n".join(text for text, _ in matches)
    prompt = (
        "You are a chat widget on Dat Ho's portfolio site. Greetings and small talk "
        "are fine, reply naturally and briefly. For anything factual about Dat, this "
        "project, or the homelab, answer using only the context below, and say you "
        "don't know if it isn't covered there. Don't make up facts that aren't in the "
        "context. Use the recent conversation only to resolve references like 'that' "
        "or follow-up questions, not as a source of facts.\n\n"
        f"{recap}"
        f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    async def stream():
        answer_parts = []
        async for token in ollama_client.chat_stream(prompt, max_tokens=settings.max_output_tokens):
            answer_parts.append(token)
            yield token
        if req.session_id:
            session_history = app.state.sessions.setdefault(
                req.session_id, deque(maxlen=SESSION_HISTORY_TURNS)
            )
            session_history.append((req.question, "".join(answer_parts)))
            if len(app.state.sessions) > MAX_SESSIONS:
                oldest_id = next(iter(app.state.sessions))
                del app.state.sessions[oldest_id]

    return StreamingResponse(stream(), media_type="text/plain")
