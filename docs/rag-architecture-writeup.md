# How I built a RAG chat service, and why

This is the writeup version of `docs/superpowers/specs/2026-08-15-homelab-llm-rag-design.md`.
That doc is the engineering spec. This one is for anyone reading my portfolio who wants to
know what I actually built and why I made the choices I did, without needing to already
know what RAG is.

## What RAG is, briefly

A plain LLM only knows what it was trained on. Ask it about something specific to you
(a project, a company, your own work history) and it either says it doesn't know or, worse,
makes something up that sounds plausible.

Retrieval-Augmented Generation fixes that by adding a lookup step before the model answers:

1. Turn the question into a vector (an embedding).
2. Search a store of pre-embedded text chunks for the ones closest to that vector.
3. Paste the matching chunks into the prompt as context, then ask the model to answer
   using only that context.

The model isn't retrained on your data. It's just handed the relevant pieces at
question time. That's the whole trick, and it's why the answers stay grounded instead of
hallucinated, as long as the retrieval step actually finds the right chunks.

## Why I built one myself instead of just reading about it

I wanted to feel the actual failure modes: bad chunking, weak embeddings, a model too
small to follow instructions, retrieval returning near misses. You don't run into any of
that reading a blog post. So I built a small end-to-end version and pointed it at my own
homelab box: a Ryzen 5 8600G, 16GB RAM, no GPU, no cloud API keys.

It's live at `https://llm.datisa.dev` right now, answering questions about this project
and about me, grounded in real documents, not a canned demo.

## Architecture

```
Browser (portfolio chat widget)
    |  HTTPS
Cloudflare Tunnel
    |
Caddy (reverse proxy, TLS terminates at Cloudflare's edge)
    |
llm-rag-api (FastAPI, Podman/Quadlet container)
    |  password check -> embed question -> retrieve top-k -> build grounded prompt
    |-- sqlite-vec (vector store, own persistent volume)
    `-- ollama (separate container, chat + embedding models, not exposed outside its network)
```

Two containers, a reverse proxy, and a tunnel. No managed vector DB, no cloud inference,
no orchestration platform. Small on purpose, see the decisions below for why.

## Decisions, and the tradeoffs behind them

**Python over .NET.** My day-to-day background is .NET. I picked Python anyway because the
LLM/RAG tooling ecosystem (Ollama's client libraries, embedding helpers) is more mature
there right now. Using the language I know less well was the point: I wanted this to prove
I can pick up what the problem needs, not just replay what I already know.

**sqlite-vec instead of Postgres+pgvector.** My homelab already runs a shared Postgres
instance for other services. I didn't put this on it. The corpus here is 20 to 50 chunks
total. Standing up an isolated Postgres with pgvector baked in, just to hold that little
data, would have been solving a problem I don't have. sqlite-vec is a single file inside
the app's own container volume: no new service to run, no extra RAM pressure on shared
infrastructure, no coupling to state other services depend on. If the corpus ever grew
into the tens of thousands of chunks, I'd reconsider. It hasn't, so I didn't.

**A quantized model, CPU-only, sized by benchmarking rather than guessing.** No discrete
GPU on the box, so the model has to be small enough to answer in a reasonable time on a
CPU. Started with `llama3.2:3b-instruct-q4_K_M` for the initial build. It had a real,
specific weakness: on multi-turn follow-ups it would sometimes answer using facts from
the wrong document in the corpus (e.g. mixing up my own work history with this project's
own tech stack) — not fixable through more prompt engineering, a capability ceiling of a
3B model at 4-bit quantization. Benchmarked it against two 7-8B candidates
(`llama3.1:8b-instruct-q4_K_M`, `qwen2.5:7b-instruct-q4_K_M`) on both speed/thermals and a
fixed quality test battery covering that exact failure case. Both candidates fixed it
cleanly; picked `llama3.1:8b-instruct-q4_K_M` since it came out roughly equivalent to the
alternative on quality and was measured from a fairer, colder baseline. The real cost:
mean response time roughly doubled (11.7s to 20.9s) and peak CPU temp went up ~6°C — a
tradeoff made deliberately, after measuring it, not assumed away. Benchmark scripts and
raw results are in `benchmarks/` in this repo.

**A shared password instead of full SSO.** My homelab already has Authentik running SSO
for other services, and the standing rule there is new services should use it. I didn't,
here, on purpose: this endpoint needs to be usable by a recruiter clicking a link, not
someone willing to create an account first. A shared password gates it against random
internet traffic without adding login friction. It's a narrower kind of protection than
SSO, and I'm calling that out rather than pretending it's equivalent.

**CI split into a hosted build job and a self-hosted deploy job.** The image builds on
GitHub's `ubuntu-latest` runners and gets pushed to GHCR. A self-hosted runner on the
homelab box only pulls the new image and restarts the container, it never builds anything.
I tried building on the homelab box first and hit real friction: Debian 13 was too new for
GitHub's prebuilt Python action, and Debian's externally-managed-environment restrictions
made a local Python toolchain more trouble than it was worth. Moving the build off-box
fixed it and also means my low-power homelab CPU isn't spending cycles compiling instead
of serving requests.

## What I'd flag as gaps, not hide

- **No per-IP rate limiting.** The stock Caddy image I run doesn't include the plugin for
  it, and I haven't built a custom Caddy image just for this one endpoint. Right now abuse
  protection is password-only plus app-level caps on question length and output tokens.
  If the password ever leaked, that's a real gap. Worth fixing with either a custom Caddy
  build or an in-app rate limiter, not yet done.
- **CORS origin was a placeholder** until the frontend widget existed to give it a real
  value to lock to.

I'd rather list what's actually missing than write a spec that only describes the good
parts.

## What's grounded in the corpus

Two kinds of source documents, kept deliberately separate:

- A public corpus (`docs_corpus/`, in the git repo): project notes about how this RAG
  service itself works, why it's built the way it is, what runs where.
- A private corpus (`docs_corpus_private/`, gitignored, only on the server): my actual
  work history, pulled from the Experience section of this same portfolio site. Real
  companies, real dates, real projects, not invented. Kept out of git so my work history
  isn't sitting in a public repo's history forever, but still fully answerable by the bot
  for anyone who asks it directly.

Both get ingested into the same vector store at startup, so a question about "how does the
retrieval work" and a question about "where does Dat work now" are answered the same way:
retrieve, ground, answer, say "I don't know" if it's not in either corpus.

## Short-term memory

The first version treated every question as a one-off, so "where does he work" followed
by "what does he use there" didn't work: the second question had no idea what "there"
meant. I added a lightweight per-visitor session: the frontend generates a random session
ID once, the backend keeps the last six question/answer pairs for that ID in memory, and
folds them into the prompt as a short recap. Restarting the service clears it, which is
fine, it's meant to make one conversation coherent, not to remember a visitor forever.

## What I actually learned building this

Retrieval quality matters more than model size. A bigger model can't save you from bad
chunking or embeddings that don't capture what the chunk is actually about. Most of the
time I spent tuning this went into retrieval, not into picking a fancier model.

Constraints are useful. No GPU, no cloud budget, and an existing set of homelab conventions
to follow all forced smaller, more deliberate choices than I'd have made with an unlimited
budget. I think the result is more interesting to talk about in an interview than "I called
the OpenAI API" would have been.
