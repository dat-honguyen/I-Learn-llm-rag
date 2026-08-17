"""
Runs the app's real system prompt + retrieval pipeline against a candidate model, for
the specific questions that exposed weaknesses in llama3.2:3b earlier in this project
(off-topic decline, short/casual identity questions, cross-document blending on
follow-ups). Not a substitute for full app testing — just a fast side-by-side.

Run inside the llm-rag-api container:
    podman exec llm-rag-api python3 /tmp/quality_check.py --model <name>
"""

import argparse
import asyncio

from app import store, ollama_client, main
from app.config import settings

CASES = [
    ("who is dat", None),
    ("tell me about yourself", None),
    ("what is the capital of France?", None),
    ("Where did the author work at Topicus Healthcare?", None),
]

FOLLOW_UP = ("What tools did he use there?", "Topicus Healthcare — Team Lead · Full-Stack Developer.")


async def ask(conn, model, question, history):
    retrieval_text = question
    if history:
        last_q, last_a = history[-1]
        retrieval_text = f"{last_q}\n{last_a}\n{question}"
    embedding = await ollama_client.embed(retrieval_text)
    matches = store.top_k(conn, embedding, settings.top_k)
    context = "\n\n".join(f"[{doc_id}]\n{text}" for text, doc_id, _ in matches)

    messages = [{"role": "system", "content": main.SYSTEM_PROMPT}]
    for q, a in history or []:
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})

    parts = []
    async for token in ollama_client.chat_stream(messages, max_tokens=settings.max_output_tokens):
        parts.append(token)
    return "".join(parts)


async def main_() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    settings.chat_model = args.model

    conn = store.init_db(settings.db_path, dim=store.DEFAULT_DIM)

    print(f"=== {args.model} ===\n")
    for question, _ in CASES:
        answer = await ask(conn, args.model, question, None)
        print(f"Q: {question}\nA: {answer}\n")

    q1, a1 = "Where did the author work at Topicus Healthcare?", await ask(
        conn, args.model, "Where did the author work at Topicus Healthcare?", None
    )
    print(f"Q: {q1}\nA: {a1}\n")
    q2 = "What tools did he use there?"
    a2 = await ask(conn, args.model, q2, [(q1, a1)])
    print(f"Q: {q2}\nA: {a2}\n")


if __name__ == "__main__":
    asyncio.run(main_())
