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
    if settings.private_docs_dir and Path(settings.private_docs_dir).is_dir():
        await ingest_docs(Path(settings.private_docs_dir), conn, ollama_client.embed)
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
        "You are a chat widget on Dat Ho's portfolio site. Greetings and small talk "
        "are fine, reply naturally and briefly. For anything factual about Dat, this "
        "project, or the homelab, answer using only the context below, and say you "
        "don't know if it isn't covered there. Don't make up facts that aren't in the "
        "context.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    async def stream():
        async for token in ollama_client.chat_stream(prompt, max_tokens=settings.max_output_tokens):
            yield token

    return StreamingResponse(stream(), media_type="text/plain")
