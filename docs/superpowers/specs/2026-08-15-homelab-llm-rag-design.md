# Homelab Local LLM RAG Service - Design

Date: 2026-08-15
Status: Implemented and deployed, live at `https://llm.datisa.dev`. This doc was updated
2026-08-16 to match what actually shipped, the sections below describe the as-built
system, not just the original plan. See "Deviations from the original design" near the
bottom for what changed and why.

Repo: `https://github.com/dat-honguyen/I-Learn-llm-rag`

## Purpose

A small, self-hosted RAG (retrieval-augmented generation) chat service running on the
homelab box (`homelab`, Ryzen 5 8600G, 16GB RAM, CPU-only inference), answering questions
grounded in a first-person "today I learn LLM" journal about the project itself (not
resume/personal data, see "Deviations" below). Serves two goals:

1. **Working demo**: a public, password-gated API that a portfolio site's chat widget can
   call, letting recruiters/reviewers ask questions about the project and get answers
   grounded in real source documents (not hallucinated).
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
     │  CORS locked to portfolio origin, no per-IP rate limit (see Deviations)
     │  header_up X-Forwarded-Proto https
llm-rag-api (Quadlet container, image pulled from ghcr.io, built by CI on GitHub-hosted
             ubuntu-latest, not built on the box, see Deployment)
     │  shared-password check → embed question → retrieve top-k → build grounded prompt
     ├──> sqlite-vec file (in llm-rag-api's own persistent volume)
     └──> ollama (Quadlet container, stock image, on dedicated llm-rag.network,
                   not exposed to host/LAN, reachable only by container name)

Caddy also joins llm-rag.network (same as it joins a network per service it proxies to)
so it can resolve llm-rag-api by container name.
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

### `llm-rag-api` (Quadlet, image built off-box by CI)
- FastAPI app, GitHub repo `I-Learn-llm-rag`, own self-hosted Actions runner
  (`homelab-llm-rag`). CI is two jobs, not one, see Deployment below for why.
- Startup ingest: reads `docs_corpus/*.md` (a first-person journal about the project
  itself, see "Deviations"), chunks (~500 tokens, overlap), embeds each chunk via
  Ollama's embedding endpoint, stores vectors in a `sqlite-vec` table on a persistent
  volume. Re-embeds only chunks whose content hash changed, so redeploys are cheap.
- `POST /chat {question, password}`:
  1. Constant-time password check against an env secret (401 on mismatch, no retry hints).
  2. Embed the question, cosine-similarity top-k retrieval from sqlite-vec.
  3. Assemble a grounded prompt. System prompt instructs the model to answer only from
     the provided context and say "I don't know" otherwise, to avoid hallucinated claims.
  4. Stream the answer back from Ollama (SSE/chunked).
- App-level guardrails: max question length, max output tokens. These are the only
  request-volume guardrail right now, see "Deviations" for why Caddy-level rate
  limiting isn't in place.
- CORS: `Access-Control-Allow-Origin` locked to the portfolio site's origin only (still a
  placeholder value as of this doc's last update, pending the real portfolio domain).
- No HTML/static frontend served by this app. API-only.

### Caddy
- Host block for `llm.datisa.dev`: `reverse_proxy llm-rag-api:8000`,
  `header_up X-Forwarded-Proto https` (standing rule). No `rate_limit` directive, see
  Deviations.
- Caddy's Quadlet unit needed `Network=llm-rag.network` added, same as it already joins
  a network per other service it proxies to (`db.network`, `paperless.network`, etc).
  Without it, Caddy can't resolve `llm-rag-api` by container name and every request
  502s.

### Cloudflare Tunnel
- `cloudflared tunnel route dns mydebian-sv llm.datisa.dev` adds the DNS CNAME.
- The ingress rule itself needed a manual, root-owned edit to
  `/etc/cloudflared/config.yml` (not user-editable without sudo): the ingress list on
  this tunnel is an explicit per-hostname list ending in a catch-all 404, not a wildcard.
  CLAUDE.md's homelab section assumed it might already be generic, it wasn't. Fixed by
  adding a `hostname: llm.datisa.dev` entry and `sudo systemctl restart cloudflared`.

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
- **Shared password over Authentik SSO**: gating with the existing Authentik SSO (the
  homelab's standing pattern for new OIDC-capable services) would add login friction for
  a portfolio visitor who just wants to try the demo. A shared password printed alongside
  the demo link avoids that friction. The original design paired this with Caddy-level
  rate limiting for defense in depth, that part didn't ship, see Deviations.
- **Split CI into build (hosted) and deploy (self-hosted, pull-only)**: matches the
  werewolf pattern exactly. `ubuntu-latest` builds and pushes the image to GHCR, the
  self-hosted runner on `homelab` only pulls and restarts, it never builds. Keeps the
  homelab box's CPU/RAM off the critical path for builds, and keeps the self-hosted
  runner from needing a working Python toolchain at all.

## Testing

- Unit tests for chunking and retrieval logic as pure functions (mocked embeddings, no
  live Ollama dependency). Runs in CI on `ubuntu-latest`, 15 tests, all passing.
- No live-model integration test in CI (too heavy, and the hosted runner has no access to
  the homelab's Ollama instance anyway). Manual smoke test against the real service after
  each deploy instead (`curl` against `/health` and `/chat`).

## Deployment

- GitHub repo `I-Learn-llm-rag`: FastAPI app, `Dockerfile`, tests, two-job GitHub Actions
  workflow (`.github/workflows/ci.yml`):
  - `build-and-push`, `runs-on: ubuntu-latest`: checkout, `python -m pytest -v` (note
    `python -m pytest`, not bare `pytest`, see Deviations), then on `main` pushes only,
    log in to GHCR and build+push `ghcr.io/dat-honguyen/llm-rag-api:latest` via
    `docker/build-push-action` with `provenance: false` (see Deviations).
  - `deploy`, needs `build-and-push`, `runs-on: [self-hosted, homelab]`, on `main`
    pushes only: `podman pull` the new image, `systemctl --user restart llm-rag-api`,
    confirm `systemctl --user is-active`.
  - Self-hosted Actions runner registered on `homelab` (`~/actions-runner-llm-rag`, own
    `systemd --user` service `actions-runner-llm-rag`). Same recipe as the werewolf
    runbook.
- `~/.config/containers/systemd/llm-rag-api.container` and `ollama.container` Quadlet
  units, plus `llm-rag.network`. `ContainerName=` set explicitly on both. Not
  `AutoUpdate=registry` on `llm-rag-api`, the CI deploy job is what pulls and restarts.
- Caddyfile block + Cloudflare DNS route for `llm.datisa.dev`. Live and verified working
  end to end as of this update.

## Deviations from the original design

Written up separately from "Key decisions" because these are places the as-built system
differs from what this doc originally called for, not decisions made up front.

- **No Caddy-level rate limiting.** The original design called for Caddy's `rate_limit`
  directive. The plugin that provides it (`caddy-ratelimit`) isn't in the stock
  `caddy:2-alpine` image this box runs, and none of the other services on this box use a
  custom Caddy build either. Shipped without it. Current abuse protection is
  password-only plus the app-level max-question-length/max-output-tokens caps, not
  per-IP throttling. A real gap if the password ever leaks, worth fixing later with
  either a custom Caddy build (`xcaddy` with the ratelimit plugin) or an app-level
  in-memory rate limiter in the FastAPI service itself.
- **CI build moved off the homelab box entirely.** The original design assumed the
  self-hosted runner would build the image itself (matching a looser reading of the
  werewolf pattern). Once actually building it, `actions/setup-python` failed on the
  runner: Debian 13 (trixie) is too new for `setup-python`'s prebuilt Python 3.12
  binaries. Rather than work around Debian-specific Python packaging quirks
  (`PEP 668` externally-managed environment, no `pip`/`venv` without sudo) on the
  self-hosted runner, the build got moved to `ubuntu-latest` entirely, matching what
  werewolf actually does (`build-and-push` hosted, `deploy` self-hosted pull-only). The
  self-hosted runner now never runs Python at all.
- **`python -m pytest`, not bare `pytest`.** Bare `pytest` doesn't add the repo root to
  `sys.path`, so `from app import ...` failed with `ModuleNotFoundError` even when pytest
  ran from the repo root. `python -m pytest` fixes it (the `-m` flag adds the invoking
  directory to `sys.path[0]`). Unrelated to the Debian/self-hosted-runner issues above,
  this one would have bitten on any runner.
- **`provenance: false` on the GHCR push.** `docker/build-push-action` failed with
  `ERROR: failed to build: unknown blob` pushing to GHCR, a known flakiness tied to the
  action's attestation-manifest push. Fixed by disabling provenance attestation.
- **Corpus is a project journal, not resume/personal data.** The original purpose section
  said "grounded in the user's resume, project write-ups, and homelab runbooks." What
  actually shipped is three first-person notes about the RAG project itself (why it was
  built, how retrieval works, where it runs), not real resume/personal content, since
  fabricating that would have meant either inventing false personal claims or piping in
  real personal data neither party had prepped for this. Still satisfies the portfolio
  goal (it demonstrates the RAG pipeline working end to end on real content), just
  narrower scope than originally written.
- **CORS_ORIGIN still a placeholder.** Blocked on the user picking a portfolio site
  domain and building the actual chat widget there, tracked as a follow-up, not resolved
  in this pass.
