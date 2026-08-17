import hmac
import re
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

# Calibrated against the live corpus (2026-08-17): on-topic questions retrieve a best
# match distance of ~16-17, off-topic/greetings both land at ~22-23, with no overlap
# below ~19. The model alone was unreliable at declining off-topic questions, so
# anything with no relevantly-close match skips the LLM and gets a canned decline,
# except common greetings (which also score as "irrelevant" but should still get a
# normal reply).
OFF_TOPIC_DISTANCE_THRESHOLD = 19.0
DECLINE_MESSAGE = "I can only answer questions about Dat and this project — try asking about his work, or how this RAG service is built."
GREETING_PATTERN = re.compile(
    r"^\s*(hi|hello|hey|yo|greetings|sup|good (morning|afternoon|evening)|"
    r"how('?s| is| are) (it going|you|things)|what'?s up)\b",
    re.IGNORECASE,
)

SYSTEM_PROMPT = (
    "You are a chat widget on Dat Ho's portfolio site. You answer questions about Dat, "
    "this RAG project, and the homelab it runs on. Greetings and small talk are fine, "
    "reply naturally and briefly. For anything else, answer using only the Context given "
    "in the user's message, and only facts stated there, never your own general "
    "knowledge, training data, or assumptions. The Context is made of separate sections, "
    "each labeled with its source note in brackets, e.g. [experience]. These sections "
    "are independent notes about different topics (Dat's work history, this RAG "
    "project's own architecture, the homelab it runs on) — a section mentioning a tool "
    "or word similar to the question does not make it relevant. Only use the section(s) "
    "that actually match what's being asked, and ignore the rest, even if other sections "
    "were retrieved alongside it. If no section covers the answer, say you don't know "
    "rather than guessing or blending facts from an unrelated section. The bracket "
    "labels are only for you to tell sections apart internally — never write them, or "
    "any other mention of 'Context' or 'sections', in your reply. Just answer the "
    "question in plain prose, like a normal chat message. If a question isn't about "
    "Dat, this project, or the homelab at all (general trivia, coding help, world facts, "
    "anything unrelated), say you can only answer questions about Dat and this project, "
    "and decline. Prior turns of this conversation are shown as earlier messages below; "
    "use them only to resolve references like 'that' or 'he', never as a source of "
    "facts."
)


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

    retrieval_text = req.question
    if history:
        last_q, last_a = history[-1]
        retrieval_text = f"{last_q}\n{last_a}\n{req.question}"

    embedding = await ollama_client.embed(retrieval_text)
    matches = store.top_k(app.state.conn, embedding, settings.top_k)

    best_distance = matches[0][2] if matches else None
    has_no_relevant_match = (
        best_distance is not None and best_distance > OFF_TOPIC_DISTANCE_THRESHOLD
    )
    is_greeting = bool(GREETING_PATTERN.match(req.question))

    if has_no_relevant_match and not is_greeting:

        async def decline():
            yield DECLINE_MESSAGE

        return StreamingResponse(decline(), media_type="text/plain")

    # A greeting with no relevant retrieval match doesn't need project context — sending
    # it irrelevant chunks anyway was confusing the small model into declining hellos.
    context = (
        ""
        if has_no_relevant_match and is_greeting
        else "\n\n".join(f"[{doc_id}]\n{text}" for text, doc_id, _ in matches)
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for q, a in history or []:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append(
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {req.question}"}
    )

    async def stream():
        answer_parts = []
        async for token in ollama_client.chat_stream(messages, max_tokens=settings.max_output_tokens):
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
