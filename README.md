# Today I Learn: LLM

A small RAG chat API I built to actually understand how retrieval-augmented
generation works, instead of just reading about it.

It runs on my home server: a local model (Ollama, `llama3.1:8b-instruct`,
CPU only, no GPU) answers questions using only these notes as context.
See `docs_corpus/` for what it actually knows about. No OpenAI key, no
cloud bill, just a Ryzen APU doing its best.

## What's here

- `app/`: the FastAPI service (chunking, embeddings, a tiny sqlite-vec
  store, the `/chat` endpoint)
- `docs_corpus/`: the notes the model is grounded in
- `tests/`: pytest, mocks Ollama so tests don't need a live model

## Running it locally

You need Ollama running somewhere with `llama3.1:8b-instruct-q4_K_M` and
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

No live model needed. Ollama calls are mocked.

## Why it's built this way

Written up in more detail in `docs/superpowers/specs/`, but the short
version: small quantized model because the box has no GPU, sqlite-vec
instead of a shared Postgres because the corpus is tiny (~20-30 chunks)
and dragging in pgvector for that would be solving a problem I don't
have, and a shared password plus rate limiting instead of full SSO
because this is a public demo, not an account system.

For the recruiter-facing version of this, what RAG actually is and why
each decision was made, see `docs/rag-architecture-writeup.md`.

## License

MIT. See `LICENSE`.
