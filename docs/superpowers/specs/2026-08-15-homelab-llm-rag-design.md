# Homelab Local LLM RAG Service - Design

Date: 2026-08-15
Status: Approved (brainstorming), pending implementation plan

## Purpose

A small, self-hosted RAG (retrieval-augmented generation) chat service running on the
homelab box (`homelab`, Ryzen 5 8600G, 16GB RAM, CPU-only inference), answering questions
grounded in the user's resume, project write-ups, and homelab runbooks. Serves two goals:

1. **Working demo**: a public, rate-limited API that a portfolio site's chat widget can
   call, letting recruiters/reviewers ask questions about the user's background and get
   answers grounded in real source documents (not hallucinated).
2. **Architecture writeup artifact**: the design itself (right-sized data store, isolated
   inference container, CI pattern reuse) is written up as a portfolio piece demonstrating
   architecture judgment, separate from the running service.

Portfolio frontend (chat widget UI) lives in the user's existing portfolio site repo.
**Out of scope** for this design. This service is API-only.

## Constraints

- Homelab box: Ryzen 5 8600G (Zen 4, Radeon 760M iGPU, no discrete GPU), 16GB RAM shared
  with other running services (Immich, Authentik, Postgres, Werewolf, Paperless, Valkey).
  CPU-bound inference. Model choice must stay small enough to be responsive and to not
  starve other services of RAM.
- Follows established homelab conventions (see `CLAUDE.md`): Podman + Quadlet (not
  docker-compose), Caddy reverse proxy on `127.0.0.1:8080`, Cloudflare Tunnel for public
  DNS, self-hosted GitHub Actions runner per self-built-image repo, shared-cache/
  shared-postgres reuse rule (with documented exception for baked-in-extension needs).

## Non-goals

- Not a general-purpose chatbot. Scope is limited to answering questions grounded in the
  ingested doc corpus.
- Not using shared-postgres. Corpus is tiny (~20-50 chunks), so a dedicated Postgres +
  pgvector instance is disproportionate. See "Vector store" decision below.
- Not building a frontend in this repo. The portfolio site owns the UI.

## Architecture

```
Recruiter/browser (portfolio site, chat widget)
     │  HTTPS, cross-origin fetch
Cloudflare Tunnel (existing tunnel "mydebian-sv")
     │
Caddy :8080 (existing container, new host block for llm.datisa.dev)
     │  rate_limit (per-IP) + CORS locked to portfolio origin
     │  header_up X-Forwarded-Proto https
llm-rag-api (new Quadlet container, self-built Python/FastAPI image)
     │  shared-password check → embed question → retrieve top-k → build grounded prompt
     ├──> sqlite-vec file (in llm-rag-api's own persistent volume)
     └──> ollama (new Quadlet container, stock image, on dedicated .network,
                   not exposed to host/LAN, reachable only by container name)
```

## Components

### `ollama` (Quadlet, stock image)
- Image: `ollama/ollama`, `AutoUpdate=registry`.
- Serves `llama3.2:3b-instruct-q4_K_M` (fallback candidate: Phi-3.5-mini-instruct if it
  benchmarks faster on the 8600G) for chat, and `nomic-embed-text` for embeddings.
- `ContainerName=ollama` set explicitly (per established gotcha: short-name DNS alias
  requires this for `llm-rag-api` to reach it by hostname).
- Joins a dedicated `.network` shared only with `llm-rag-api`. Not published to host or LAN.
- Model pulled post-deploy via `podman exec ollama ollama pull <model>` (one-time, not part
  of the image).

### `llm-rag-api` (Quadlet, self-built Python image)
- FastAPI app, own GitHub repo (`homelab-llm-rag`), own self-hosted Actions runner.
  Matches the established werewolf CI pattern (self-built images use the runner pattern,
  not `AutoUpdate=registry`).
- Startup ingest: reads `docs/*.md` (resume, project write-ups, homelab runbooks), chunks
  (~500 tokens, overlap), embeds each chunk via Ollama's embedding endpoint, stores vectors
  in a `sqlite-vec` table on a persistent volume. Re-embeds only chunks whose content hash
  changed, so redeploys are cheap.
- `POST /chat {question, password}`:
  1. Constant-time password check against an env secret (401 on mismatch, no retry hints).
  2. Embed the question, cosine-similarity top-k retrieval from sqlite-vec.
  3. Assemble a grounded prompt. System prompt instructs the model to answer only from
     the provided context and say "I don't know" otherwise, to avoid hallucinated claims
     about the user.
  4. Stream the answer back from Ollama (SSE/chunked).
- App-level guardrails (defense in depth under Caddy's rate limit): max question length,
  max output tokens.
- CORS: `Access-Control-Allow-Origin` locked to the portfolio site's origin only.
- No HTML/static frontend served by this app. API-only.

### Caddy
- New host block for `llm.datisa.dev`: `reverse_proxy llm-rag-api:<port>`, `rate_limit`
  directive (e.g. 5 req/min per IP), `header_up X-Forwarded-Proto https` (standing rule).

### Cloudflare Tunnel
- `cloudflared tunnel route dns mydebian-sv llm.datisa.dev` (existing tunnel, ingress rule
  likely already generic, verify `/etc/cloudflared/config.yml` first).

## Key decisions (with rationale, for the writeup)

- **Python over .NET**: considered leaning on the user's .NET/architecture specialization
  for portfolio differentiation, but user chose Python for its more mature LLM/RAG tooling
  ecosystem (Ollama clients, embedding libs).
- **sqlite-vec over shared-postgres**: `shared-postgres` is a stock `postgres:17` image
  without `pgvector` baked in. Per the homelab's own shared-infra exception rule (same
  reasoning as `immich-postgres`'s dedicated vectorchord build), a service needing a
  baked-in extension the shared instance lacks is allowed to go isolated. But for a
  ~20-50-chunk corpus, standing up an entire isolated Postgres just for pgvector is
  over-engineering. An embedded `sqlite-vec` file inside the app's own volume avoids a new
  container, avoids RAM pressure on `shared-postgres`, and avoids coupling to shared state
  other services depend on. This is a deliberate right-sizing decision, not a shortcut,
  and it's worth calling out explicitly in the architecture writeup.
- **CPU-only, small quantized model**: no discrete GPU on the box. A 3B-class Q4
  quantized model keeps response latency acceptable for a live demo without starving
  the other homelab services of RAM.
- **Rate limit + shared password over Authentik SSO**: gating with the existing Authentik
  SSO (the homelab's standing pattern for new OIDC-capable services) would add login
  friction for a portfolio visitor who just wants to try the demo. A shared password
  printed alongside the demo link plus Caddy-level rate limiting bounds abuse risk on the
  CPU-bound box without that friction.

## Testing

- Unit tests for chunking and retrieval logic as pure functions (mocked embeddings, no
  live Ollama dependency). Runs in CI on the self-hosted runner.
- No live-model integration test in CI (too heavy for a runner shared with other repos'
  builds). Manual smoke test against the real Ollama instance after each deploy instead.

## Deployment

- New GitHub repo `homelab-llm-rag`: FastAPI app, `Dockerfile`, tests, GitHub Actions
  workflow, self-hosted Actions runner registered on `homelab` (own directory under `~`,
  own `systemd --user` service). Same recipe as the werewolf runbook.
- `~/.config/containers/systemd/llm-rag-api.container` and `ollama.container` Quadlet
  units, plus a `.network` unit shared between them. `ContainerName=` set explicitly on
  both.
- New Caddyfile block + Cloudflare DNS route for `llm.datisa.dev`.

## Open items for the implementation plan

- Confirm exact model choice (`llama3.2:3b-instruct-q4_K_M` vs Phi-3.5-mini) via a quick
  benchmark on the box once Ollama is running.
- Confirm CORS origin (exact portfolio site domain).
- Decide the shared-password rotation/storage mechanism (env var is sufficient for v1).
