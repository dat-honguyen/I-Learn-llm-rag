# Deploying to the homelab box

1. Copy `deploy/llm-rag.network`, `deploy/ollama.container`, and
   `deploy/llm-rag-api.container` into `~/.config/containers/systemd/` on `homelab`.
2. Replace the `APP_PASSWORD` placeholder in `llm-rag-api.container` with a real value
   (don't commit it).
3. `systemctl --user daemon-reload`
4. `systemctl --user start ollama`
5. `podman exec ollama ollama pull llama3.2:3b-instruct-q4_K_M`
6. `podman exec ollama ollama pull nomic-embed-text`
7. First image pull: `podman pull ghcr.io/dat-honguyen/llm-rag-api:latest`, then
   `systemctl --user start llm-rag-api`. After that, pushes to `main` build and push the
   image on GitHub-hosted `ubuntu-latest`, then the self-hosted runner on `homelab` just
   pulls the new image and restarts the service, it never builds anything locally.
8. Append `deploy/caddy-llm-rag.snippet`'s contents to `~/caddy/Caddyfile`, then
   `systemctl --user restart caddy`.
9. `cloudflared tunnel route dns mydebian-sv llm.datisa.dev`, then add a matching
   `hostname:` entry to `/etc/cloudflared/config.yml` (root-owned, needs sudo, the
   ingress list there is an explicit per-hostname list ending in a 404 catch-all, not a
   wildcard) and `sudo systemctl restart cloudflared`.
10. Optional: extra personal/resume content the RAG bot should know about but that
    shouldn't go in the public `docs_corpus/` in this repo. Drop markdown files into
    `~/I-Learn-llm-rag-private-docs/` on the server (create the folder if it doesn't
    exist yet), then `systemctl --user restart llm-rag-api`. It's mounted read-only into
    the container at `/app/private_docs` and ingested on startup via `PRIVATE_DOCS_DIR`,
    same chunking/embedding pipeline as the committed corpus, just never touches git.
11. Smoke test: `curl -X POST https://llm.datisa.dev/chat -d '{"question": "...", "password": "..."}'`.
